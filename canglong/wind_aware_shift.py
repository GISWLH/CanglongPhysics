"""
风向感知的窗口移位模块
在Swin-Transformer原有固定移位基础上，叠加一次基于风向的额外移位

采用 Mask & Combine 方法：
1. 对整张大图预生成9种移位版本
2. 根据每个区域的风向ID选择对应的移位版本
3. 保持全图物理连续性，避免块效应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 风向ID到移位方向的映射
# 风向ID: (lat_shift, lon_shift)
WIND_SHIFT_DIRECTIONS = {
    0: (0, 0),      # 无移位
    1: (-1, 0),     # N: 向北（纬度减小）
    2: (-1, 1),     # NE: 东北
    3: (0, 1),      # E: 向东（经度增大）
    4: (1, 1),      # SE: 东南
    5: (1, 0),      # S: 向南（纬度增大）
    6: (1, -1),     # SW: 西南
    7: (0, -1),     # W: 向西（经度减小）
    8: (-1, -1),    # NW: 西北
}

WIND_DIR_NAMES = {0: 'None', 1: 'N', 2: 'NE', 3: 'E', 4: 'SE', 5: 'S', 6: 'SW', 7: 'W', 8: 'NW'}


def get_dominant_direction(wind_direction_id):
    """
    获取batch的主导风向（众数）

    参数:
        wind_direction_id (torch.Tensor): 风向ID, 形状为 (B, H, W)

    返回:
        dominant_id (int): 主导风向ID (0-8)
    """
    flat_ids = wind_direction_id.flatten().long()
    counts = torch.bincount(flat_ids, minlength=9)
    dominant_id = counts.argmax().item()
    return dominant_id


def get_regional_dominant_directions(wind_direction_id, num_regions=(4, 8)):
    """
    计算每个区域的主导风向

    参数:
        wind_direction_id (torch.Tensor): 风向ID, 形状为 (B, H, W)
        num_regions (tuple): 区域划分 (lat_regions, lon_regions)

    返回:
        regional_directions (torch.Tensor): 每个区域的主导风向 (B, lat_regions, lon_regions)
    """
    B, H, W = wind_direction_id.shape
    lat_regions, lon_regions = num_regions

    region_h = H // lat_regions
    region_w = W // lon_regions

    regional_directions = torch.zeros(B, lat_regions, lon_regions,
                                      dtype=torch.long, device=wind_direction_id.device)

    for i in range(lat_regions):
        for j in range(lon_regions):
            h_start = i * region_h
            h_end = (i + 1) * region_h if i < lat_regions - 1 else H
            w_start = j * region_w
            w_end = (j + 1) * region_w if j < lon_regions - 1 else W

            region = wind_direction_id[:, h_start:h_end, w_start:w_end]

            for b in range(B):
                flat_region = region[b].flatten().long()
                counts = torch.bincount(flat_region, minlength=9)
                regional_directions[b, i, j] = counts.argmax()

    return regional_directions


def generate_all_shifted_versions(x, shift_scale=2):
    """
    生成全图的9种移位版本

    参数:
        x (torch.Tensor): 输入张量, 形状为 (B, Pl, Lat, Lon, C)
        shift_scale (int): 移位幅度

    返回:
        shifted_versions (dict): {direction_id: shifted_tensor}
    """
    shifted_versions = {}

    for dir_id, (lat_shift, lon_shift) in WIND_SHIFT_DIRECTIONS.items():
        actual_lat_shift = lat_shift * shift_scale
        actual_lon_shift = lon_shift * shift_scale

        if actual_lat_shift == 0 and actual_lon_shift == 0:
            shifted_versions[dir_id] = x
        else:
            # 对整张图进行移位，保持物理连续性
            shifted = torch.roll(x, shifts=(0, actual_lat_shift, actual_lon_shift), dims=(1, 2, 3))
            shifted_versions[dir_id] = shifted

    return shifted_versions


def create_region_mask(shape, num_regions, region_directions, device):
    """
    创建区域掩码，用于从9种移位版本中选择

    优化版本：使用 F.interpolate 替代 Python 循环，利用 GPU 并行加速

    参数:
        shape (tuple): 输出形状 (B, Pl, Lat, Lon, C)
        num_regions (tuple): 区域划分 (lat_regions, lon_regions)
        region_directions (torch.Tensor): 各区域的风向ID (B, lat_regions, lon_regions)
        device: 设备

    返回:
        masks (dict): {direction_id: mask_tensor} 每个方向的掩码
    """
    B, Pl, Lat, Lon, C = shape

    # Step 1: 使用最近邻插值将 regional_directions 放大到像素级
    # regional_directions: (B, lat_regions, lon_regions) -> (B, 1, lat_regions, lon_regions)
    dirs_map = region_directions.unsqueeze(1).float()

    # 插值放大 -> (B, 1, Lat, Lon)
    dirs_map_up = F.interpolate(dirs_map, size=(Lat, Lon), mode='nearest')

    # Step 2: 生成9个方向的掩码（向量化操作，无Python循环）
    masks = {}
    for dir_id in range(9):
        # 生成 (B, 1, Lat, Lon) 的 mask，然后扩展为 (B, 1, Lat, Lon, 1)
        mask = (dirs_map_up == dir_id).float().unsqueeze(-1)
        masks[dir_id] = mask

    return masks


def apply_mask_and_combine(shifted_versions, masks):
    """
    使用掩码从9种移位版本中组合出最终结果

    参数:
        shifted_versions (dict): {direction_id: shifted_tensor}
        masks (dict): {direction_id: mask_tensor}

    返回:
        combined (torch.Tensor): 组合后的张量
    """
    # 获取任一张量作为模板
    template = next(iter(shifted_versions.values()))
    combined = torch.zeros_like(template)

    for dir_id in range(9):
        if dir_id in shifted_versions and dir_id in masks:
            combined = combined + shifted_versions[dir_id] * masks[dir_id]

    return combined


def apply_regional_wind_shift_v2(x, wind_direction_id, num_regions=(4, 8),
                                  wind_shift_scale=2, reverse=False):
    """
    应用分区域的风向移位（Mask & Combine 方法）

    核心思想：
    1. 对整张大图预生成9种移位版本
    2. 根据每个区域的风向ID创建掩码
    3. 使用掩码从9种版本中选择并组合

    参数:
        x (torch.Tensor): 输入张量, 形状为 (B, Pl, Lat, Lon, C)
        wind_direction_id (torch.Tensor): 风向ID, 形状为 (B, H', W')
        num_regions (tuple): 区域划分 (lat_regions, lon_regions)
        wind_shift_scale (int): 风向移位的缩放因子
        reverse (bool): 是否反向移位

    返回:
        shifted_x (torch.Tensor): 移位后的张量
        regional_dirs (torch.Tensor): 各区域使用的风向
    """
    if wind_direction_id is None:
        return x, None

    B, Pl, Lat, Lon, C = x.shape
    device = x.device

    # Step 1: 获取各区域的主导风向
    regional_dirs = get_regional_dominant_directions(wind_direction_id, num_regions)

    # 如果是反向移位，需要反转方向
    if reverse:
        # 反转映射: N<->S, E<->W, NE<->SW, NW<->SE
        reverse_map = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}
        regional_dirs_reversed = regional_dirs.clone()
        for old_id, new_id in reverse_map.items():
            regional_dirs_reversed[regional_dirs == old_id] = new_id
        regional_dirs = regional_dirs_reversed

    # Step 2: 生成9种移位版本
    shifted_versions = generate_all_shifted_versions(x, wind_shift_scale)

    # Step 3: 创建区域掩码
    masks = create_region_mask(x.shape, num_regions, regional_dirs, device)

    # Step 4: 使用掩码组合
    combined = apply_mask_and_combine(shifted_versions, masks)

    return combined, regional_dirs


# 保留旧的全局模式函数以保持兼容性
def apply_wind_extra_shift(x, wind_direction_id, wind_shift_scale=1, reverse=False):
    """
    应用基于风向的额外移位（全局模式）
    """
    if wind_direction_id is None:
        return x, 0

    dominant_id = get_dominant_direction(wind_direction_id)

    lat_shift, lon_shift = WIND_SHIFT_DIRECTIONS[dominant_id]
    lat_shift *= wind_shift_scale
    lon_shift *= wind_shift_scale

    if reverse:
        lat_shift = -lat_shift
        lon_shift = -lon_shift

    if lat_shift != 0 or lon_shift != 0:
        shifted_x = torch.roll(x, shifts=(0, lat_shift, lon_shift), dims=(1, 2, 3))
    else:
        shifted_x = x

    return shifted_x, dominant_id


class WindAwareDoubleShifter(nn.Module):
    """
    双重移位器：Swin固定移位 + 风向额外移位

    支持两种模式：
    - mode='global': 全局主导风向
    - mode='regional': 分区域独立风向（Mask & Combine方法）
    """

    def __init__(self, swin_shift_size=(1, 3, 6), wind_shift_scale=2,
                 mode='regional', num_regions=(4, 8)):
        """
        参数:
            swin_shift_size (tuple): Swin的固定移位大小 (pl, lat, lon)
            wind_shift_scale (int): 风向移位的缩放因子
            mode (str): 'global' 或 'regional'
            num_regions (tuple): 区域划分，仅在regional模式下使用
        """
        super().__init__()
        self.swin_shift_size = swin_shift_size
        self.wind_shift_scale = wind_shift_scale
        self.mode = mode
        self.num_regions = num_regions
        self.last_dominant_id = 0
        self.last_regional_dirs = None

    def forward_shift(self, x, wind_direction_id, do_swin_shift=True):
        """
        正向移位：先Swin移位，再风向移位
        """
        # Step 1: Swin固定移位
        if do_swin_shift:
            shift_pl, shift_lat, shift_lon = self.swin_shift_size
            x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))

        # Step 2: 风向额外移位
        if self.mode == 'regional':
            x, regional_dirs = apply_regional_wind_shift_v2(
                x, wind_direction_id,
                num_regions=self.num_regions,
                wind_shift_scale=self.wind_shift_scale,
                reverse=False
            )
            self.last_regional_dirs = regional_dirs
            if wind_direction_id is not None:
                self.last_dominant_id = get_dominant_direction(wind_direction_id)
        else:
            x, dominant_id = apply_wind_extra_shift(
                x, wind_direction_id,
                wind_shift_scale=self.wind_shift_scale,
                reverse=False
            )
            self.last_dominant_id = dominant_id

        return x, self.last_dominant_id

    def backward_shift(self, x, wind_direction_id, do_swin_shift=True):
        """
        反向移位：先反向风向移位，再反向Swin移位
        """
        # Step 1: 反向风向移位
        if self.mode == 'regional':
            x, _ = apply_regional_wind_shift_v2(
                x, wind_direction_id,
                num_regions=self.num_regions,
                wind_shift_scale=self.wind_shift_scale,
                reverse=True
            )
        else:
            x, _ = apply_wind_extra_shift(
                x, wind_direction_id,
                wind_shift_scale=self.wind_shift_scale,
                reverse=True
            )

        # Step 2: 反向Swin移位
        if do_swin_shift:
            shift_pl, shift_lat, shift_lon = self.swin_shift_size
            x = torch.roll(x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))

        return x

    def get_regional_stats(self):
        """获取区域风向统计信息"""
        return self.last_regional_dirs
