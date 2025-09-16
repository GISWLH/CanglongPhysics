import torch
import torch.nn as nn
import numpy as np


def calculate_wind_direction(u, v):
    """
    根据u,v风场分量计算风向
    
    参数:
        u (torch.Tensor): u风场分量, 形状为 (..., H, W)
        v (torch.Tensor): v风场分量, 形状为 (..., H, W)
    
    返回:
        wind_direction (torch.Tensor): 风向(角度), 形状为 (..., H, W)
    """
    # 计算风向(角度)
    wind_direction = torch.atan2(v, u) * 180 / np.pi
    # 转换为0-360度范围
    wind_direction = (wind_direction + 360) % 360
    return wind_direction


def calculate_dominant_wind_direction(wind_direction, window_size=(4, 4)):
    """
    计算每个窗口的主导风向
    
    参数:
        wind_direction (torch.Tensor): 风向(角度), 形状为 (B, H, W)
        window_size (tuple): 窗口大小 (window_h, window_w)
    
    返回:
        dominant_direction (torch.Tensor): 主导风向, 形状为 (B, H//window_h, W//window_w)
    """
    B, H, W = wind_direction.shape
    win_h, win_w = window_size
    
    # 使用平均池化处理不能整除的情况
    # 先调整张量大小以适应平均池化
    wind_direction_unsqueezed = wind_direction.unsqueeze(1)  # (B, 1, H, W)
    
    # 使用自适应平均池化调整到目标尺寸
    target_h = H // win_h
    target_w = W // win_w
    dominant_direction = torch.nn.functional.adaptive_avg_pool2d(
        wind_direction_unsqueezed, (target_h, target_w)
    ).squeeze(1)  # (B, target_h, target_w)
    
    return dominant_direction


def get_wind_direction_id(dominant_direction):
    """
    将风向转换为离散的方向ID (0-8)
    0: 不移位, 1-8: N, NE, E, SE, S, SW, W, NW
    
    参数:
        dominant_direction (torch.Tensor): 主导风向(角度), 形状为 (B, H, W)
    
    返回:
        direction_id (torch.Tensor): 方向ID (0-8), 形状为 (B, H, W)
    """
    # 将角度划分为9个区间 (每个40度)
    # 0: [340, 360]U[0, 20] (不移位)
    # 1: [20, 60] (N)
    # 2: [60, 100] (NE)
    # 3: [100, 140] (E)
    # 4: [140, 180] (SE)
    # 5: [180, 220] (S)
    # 6: [220, 260] (SW)
    # 7: [260, 300] (W)
    # 8: [300, 340] (NW)
    
    # 先处理特殊情况: [340, 360]U[0, 20] -> 0
    direction_id = torch.zeros_like(dominant_direction, dtype=torch.long)
    
    # 为每个区间分配ID
    direction_id[(dominant_direction >= 20) & (dominant_direction < 60)] = 1   # N
    direction_id[(dominant_direction >= 60) & (dominant_direction < 100)] = 2  # NE
    direction_id[(dominant_direction >= 100) & (dominant_direction < 140)] = 3 # E
    direction_id[(dominant_direction >= 140) & (dominant_direction < 180)] = 4 # SE
    direction_id[(dominant_direction >= 180) & (dominant_direction < 220)] = 5 # S
    direction_id[(dominant_direction >= 220) & (dominant_direction < 260)] = 6 # SW
    direction_id[(dominant_direction >= 260) & (dominant_direction < 300)] = 7 # W
    direction_id[(dominant_direction >= 300) & (dominant_direction < 340)] = 8 # NW
    # 0度对应不移位的情况，已初始化为0
    
    return direction_id


class WindDirectionProcessor(nn.Module):
    """
    风向处理模块，用于从输入数据中提取风向信息并计算主导风向
    """
    
    def __init__(self, window_size=(4, 4)):
        super(WindDirectionProcessor, self).__init__()
        self.window_size = window_size
    
    def forward(self, surface, upper_air):
        """
        从surface和upper_air数据中提取风向信息并计算主导风向
        
        参数:
            surface (torch.Tensor): 表面数据, 形状为 (B, 17, 2, 721, 1440)
            upper_air (torch.Tensor): 高空数据, 形状为 (B, 7, 5, 2, 721, 1440)
            
        返回:
            wind_direction_id (torch.Tensor): 风向ID, 形状为 (B, 181, 360)
        """
        B, _, _, _, H, W = upper_air.shape
        
        # 提取高空u/v风场数据 (第3,4层对应索引2,3)
        # upper_air shape: (B, 7, 5, 2, 721, 1440)
        # 提取第3,4层: upper_air[:, :, 2:4, :, :, :]
        upper_u = upper_air[:, :, 2, :, :, :]  # (B, 7, 2, 721, 1440)
        upper_v = upper_air[:, :, 3, :, :, :]  # (B, 7, 2, 721, 1440)
        
        # 提取表面10m u/v风场数据 (第5,6层对应索引4,5)
        # surface shape: (B, 17, 2, 721, 1440)
        surface_u = surface[:, 4, :, :, :]  # (B, 2, 721, 1440)
        surface_v = surface[:, 5, :, :, :]  # (B, 2, 721, 1440)
        
        # 合并高空和表面风场数据 (取时间维度的平均)
        # 对于高空数据，取所有压力层的平均
        upper_u_mean = upper_u.mean(dim=1)  # (B, 2, 721, 1440)
        upper_v_mean = upper_v.mean(dim=1)  # (B, 2, 721, 1440)
        
        # 取时间维度的平均
        upper_u_mean = upper_u_mean.mean(dim=1)  # (B, 721, 1440)
        upper_v_mean = upper_v_mean.mean(dim=1)  # (B, 721, 1440)
        surface_u_mean = surface_u.mean(dim=1)   # (B, 721, 1440)
        surface_v_mean = surface_v.mean(dim=1)   # (B, 721, 1440)
        
        # 合并高空和表面风场数据 (取平均)
        combined_u = (upper_u_mean + surface_u_mean) / 2
        combined_v = (upper_v_mean + surface_v_mean) / 2
        
        # 计算风向
        wind_direction = calculate_wind_direction(combined_u, combined_v)  # (B, 721, 1440)
        
        # 下采样到181x360 (4x4下采样)
        # 使用平均池化进行下采样
        wind_direction_downsampled = torch.nn.functional.adaptive_avg_pool2d(
            wind_direction.unsqueeze(1), (181, 360)
        ).squeeze(1)  # (B, 181, 360)
        
        # 计算主导风向
        dominant_direction = calculate_dominant_wind_direction(
            wind_direction_downsampled, self.window_size
        )  # (B, 181//win_h, 360//win_w)
        
        # 获取风向ID
        direction_id = get_wind_direction_id(dominant_direction)  # (B, 181//win_h, 360//win_w)
        
        return direction_id