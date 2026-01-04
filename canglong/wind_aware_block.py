"""
风向感知的Transformer Block
在Swin-Transformer基础上，叠加风向驱动的额外移位
"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath

from canglong.earth_position import calculate_position_bias_indices
from canglong.shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from canglong.pad import calculate_padding_3d
from canglong.crop import center_crop_3d
from canglong.wind_aware_shift import WindAwareDoubleShifter


class WindAwareEarthAttention3D(nn.Module):
    """
    3D窗口注意力，支持基于风向的动态窗口交换
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros((window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                        self.type_of_windows, num_heads)
        )

        earth_position_index = calculate_position_bias_indices(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        earth_position_bias = self.earth_position_bias_table[self.earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(B_ // nLon, nLon, self.num_heads, nW_, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindAwareEarthSpecificBlock(nn.Module):
    """
    3D Transformer Block，支持基于风向的双重移位：
    1. Swin固定移位（奇数块）
    2. 风向额外移位（所有块）

    支持两种风向移位模式：
    - 'global': 全局主导风向（整个batch一个方向）
    - 'regional': 分区域独立风向（4x8=32个区域各自方向）
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_wind_aware_shift=True, wind_shift_scale=2,
                 wind_shift_mode='regional', num_regions=(4, 8)):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_wind_aware_shift = use_wind_aware_shift
        self.wind_shift_mode = wind_shift_mode

        self.norm1 = norm_layer(dim)
        padding = calculate_padding_3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.attn = WindAwareEarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        # 判断是否需要Swin移位（奇数块需要）
        shift_pl, shift_lat, shift_lon = self.shift_size
        self.do_swin_shift = (shift_pl != 0) or (shift_lat != 0) or (shift_lon != 0)

        # 双重移位器（支持全局/分区域模式）
        if self.use_wind_aware_shift:
            self.double_shifter = WindAwareDoubleShifter(
                swin_shift_size=self.shift_size,
                wind_shift_scale=wind_shift_scale,
                mode=wind_shift_mode,
                num_regions=num_regions
            )

        # 注意力掩码（仅Swin移位时需要）
        if self.do_swin_shift:
            attn_mask = create_shifted_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, wind_direction_id=None):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入特征, 形状为 (B, L, C)
            wind_direction_id (torch.Tensor, optional): 风向ID, 形状为 (B, H, W)
        """
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # Padding
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        # === 双重移位 ===
        if self.use_wind_aware_shift and wind_direction_id is not None:
            # 使用风向感知的双重移位
            x, dominant_id = self.double_shifter.forward_shift(
                x, wind_direction_id, do_swin_shift=self.do_swin_shift
            )
        elif self.do_swin_shift:
            # 仅使用Swin固定移位
            shift_pl, shift_lat, shift_lon = self.shift_size
            x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))

        # 窗口分割
        x_windows = partition_windows(x, self.window_size)
        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)

        # 注意力计算
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 窗口重组
        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)
        x = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)

        # === 反向双重移位 ===
        if self.use_wind_aware_shift and wind_direction_id is not None:
            # 反向风向感知的双重移位
            x = self.double_shifter.backward_shift(
                x, wind_direction_id, do_swin_shift=self.do_swin_shift
            )
        elif self.do_swin_shift:
            # 仅反向Swin移位
            shift_pl, shift_lat, shift_lon = self.shift_size
            x = torch.roll(x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))

        # Crop回原始尺寸
        x = center_crop_3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
