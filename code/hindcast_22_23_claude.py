"""
Hindcast 2022-2023 批量回报脚本
基于run_temp.py改编，用于批量处理2022-2023年全年数据
使用预加载的输入数据，无需从Google Cloud下载
"""

import torch
import numpy as np
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.special import gamma as gamma_function

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 常量定义
forecast_weeks = 6
data_inner_steps = 24

# 变量列表
surface_var_names = [
    'large_scale_rain_rate',
    'convective_rain_rate',
    'total_column_cloud_ice_water',
    'total_cloud_cover',
    'top_net_solar_radiation_clear_sky',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'surface_latent_heat_flux',
    'surface_sensible_heat_flux',
    'surface_pressure',
    'volumetric_soil_water_layer',
    'mean_sea_level_pressure',
    'sea_ice_cover',
    'sea_surface_temperature'
]

upper_air_vars = [
    'geopotential',
    'vertical_velocity',
    'u_component_of_wind',
    'v_component_of_wind',
    'fraction_of_cloud_cover',
    'temperature',
    'specific_humidity'
]

# 变量映射和统计信息
var_mapping = {
    'large_scale_rain_rate': 'lsrr',
    'convective_rain_rate': 'crr',
    'total_column_cloud_ice_water': 'tciw',
    'total_cloud_cover': 'tcc',
    'top_net_solar_radiation_clear_sky': 'tsrc',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_dewpoint_temperature': 'd2m',
    '2m_temperature': 't2m',
    'surface_latent_heat_flux': 'surface_latent_heat_flux',
    'surface_sensible_heat_flux': 'surface_sensible_heat_flux',
    'surface_pressure': 'sp',
    'volumetric_soil_water_layer': 'swvl',
    'mean_sea_level_pressure': 'msl',
    'sea_ice_cover': 'siconc',
    'sea_surface_temperature': 'sst'
}

ordered_var_stats = {
    'lsrr': {'mean': 1.10E-05, 'std': 2.55E-05},
    'crr': {'mean': 1.29E-05, 'std': 2.97E-05},
    'tciw': {'mean': 0.022627383, 'std': 0.023428712},
    'tcc': {'mean': 0.673692584, 'std': 0.235167906},
    'tsrc': {'mean': 856148, 'std': 534222.125},
    'u10': {'mean': -0.068418466, 'std': 4.427545547},
    'v10': {'mean': 0.197138891, 'std': 3.09530735},
    'd2m': {'mean': 274.2094421, 'std': 20.45770073},
    't2m': {'mean': 278.7841187, 'std': 21.03286934},
    'surface_latent_heat_flux': {'mean': -5410301.5, 'std': 5349063.5},
    'surface_sensible_heat_flux': {'mean': -971651.375, 'std': 2276764.75},
    'sp': {'mean': 96651.14063, 'std': 9569.695313},
    'swvl': {'mean': 0.34216917, 'std': 0.5484813},
    'msl': {'mean': 100972.3438, 'std': 1191.102417},
    'siconc': {'mean': 0.785884917, 'std': 0.914535105},
    'sst': {'mean': 189.7337189, 'std': 136.1803131},

    'geopotential': {
        '300': {'mean': 13763.50879, 'std': 1403.990112},
        '500': {'mean': 28954.94531, 'std': 2085.838867},
        '700': {'mean': 54156.85547, 'std': 3300.384277},
        '850': {'mean': 89503.79688, 'std': 5027.79541}
    },
    'vertical_velocity': {
        '300': {'mean': 0.011849277, 'std': 0.126232564},
        '500': {'mean': 0.002759292, 'std': 0.097579598},
        '700': {'mean': 0.000348145, 'std': 0.072489716},
        '850': {'mean': 0.000108061, 'std': 0.049831692}
    },
    'u_component_of_wind': {
        '300': {'mean': 1.374536991, 'std': 6.700420856},
        '500': {'mean': 3.290786982, 'std': 7.666454315},
        '700': {'mean': 6.491596222, 'std': 9.875613213},
        '850': {'mean': 11.66026878, 'std': 14.00845909}
    },
    'v_component_of_wind': {
        '300': {'mean': 0.146550566, 'std': 3.75399971},
        '500': {'mean': 0.022800878, 'std': 4.179731846},
        '700': {'mean': -0.025720235, 'std': 5.324173927},
        '850': {'mean': -0.027837994, 'std': 7.523460865}
    },
    'fraction_of_cloud_cover': {
        '300': {'mean': 0.152513072, 'std': 0.15887706},
        '500': {'mean': 0.106524825, 'std': 0.144112185},
        '700': {'mean': 0.105878539, 'std': 0.112193666},
        '850': {'mean': 0.108120449, 'std': 0.108371623}
    },
    'temperature': {
        '300': {'mean': 274.8048401, 'std': 15.28209305},
        '500': {'mean': 267.6254578, 'std': 14.55300999},
        '700': {'mean': 253.1627655, 'std': 12.77071381},
        '850': {'mean': 229.0860138, 'std': 10.5536499}
    },
    'specific_humidity': {
        '300': {'mean': 0.004610791, 'std': 0.003879665},
        '500': {'mean': 0.002473272, 'std': 0.002312181},
        '700': {'mean': 0.000875093, 'std': 0.000944978},
        '850': {'mean': 0.000130984, 'std': 0.000145811}
    }
}

## Paste model here
import torch
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import sys
sys.path.append('..')  # 添加上一级目录到路径
from canglong.earth_position import calculate_position_bias_indices
from canglong.shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from canglong.embed_old import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D  
from canglong.recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
from canglong.pad import calculate_padding_3d, calculate_padding_2d
from canglong.crop import center_crop_2d, center_crop_3d
input_constant = torch.load('../constant_masks/input_tensor.pt').cuda()

class UpSample(nn.Module):
    """
    Up-sampling operation.
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: 2 * in_lat - pad_bottom, pad_left: 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
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
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1).contiguous()
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


class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = calculate_padding_3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = create_shifted_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
            x_windows = partition_windows(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = partition_windows(shifted_x, self.window_size)

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)

        if self.roll:
            shifted_x = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = torch.roll(shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
        else:
            shifted_x = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)
            x = shifted_x

        x = center_crop_3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            EarthSpecificBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                               norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
import torch
import torch.nn as nn
from canglong.helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

class Encoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(Encoder, self).__init__()
        channels = [64, 64, 64, 128, 128]
        attn_resolutions = [2]
        num_res_blocks = 1
        resolution = 256

        # 初始卷积层
        self.conv_in = nn.Conv3d(image_channels, channels[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        
        # 第一层（含残差块和注意力模块）
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks, resolution, attn_resolutions)
        
        # 下采样与第二层
        self.downsample1 = DownSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks, resolution // 2, attn_resolutions)

        # Further downsampling and third layer
        self.downsample2 = DownSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks, resolution // 4, attn_resolutions)

        # 中间层的残差块和注意力模块
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
        
        # 输出层的归一化、激活和最终卷积层
        self.norm_out = GroupNorm(channels[3])
        self.act_out = Swish()
        self.conv_out = nn.Conv3d(channels[3], latent_dim, kernel_size=3, stride=1, padding=(1,2,1))

    def _make_layer(self, in_channels, out_channels, num_res_blocks, resolution, attn_resolutions):
        layers = []
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
            if resolution in attn_resolutions:
                layers.append(NonLocalBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv_in(x)

        # 第一层，并存储跳跃连接
        x = self.layer1(x)
        skip = x  # 保存第一层输出，用于后续跳跃连接

        # 下采样，进入第二层
        x = self.downsample1(x)
        x = self.layer2(x)

        # Further downsample and third layer
        x = self.downsample2(x)
        x = self.layer3(x)

        # 中间层的残差块和注意力模块
        x = self.mid_block1(x)
        #x = self.mid_attn(x)
        x = self.mid_block2(x)
        
        # 最终的归一化、激活和卷积输出层
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)[:, :, :, :181, :360]
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, image_channels=14, latent_dim=64):
        super(Decoder, self).__init__()
        channels = [128, 128, 64, 64]  # Decoder 的通道配置
        num_res_blocks = 1  # 与 Encoder 对齐

        # 初始卷积层
        self.conv_in = nn.Conv3d(latent_dim, channels[0], kernel_size=3, stride=1, padding=1)
        
        # 第一层残差块
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks)
        
        # 上采样和第二层残差块
        self.upsample1 = UpSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks)

        self.upsample2 = UpSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks)
        
        # 中间层的残差块
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
        
        # 最终输出层
        self.norm_out = GroupNorm(channels[3])
        self.act_out = Swish()
        self.conv_out = nn.ConvTranspose3d(channels[3], image_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def _make_layer(self, in_channels, out_channels, num_res_blocks):
        # 创建指定数量的残差块
        layers = [ResidualBlock(in_channels, out_channels) for _ in range(num_res_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv_in(x)

        # 第一层残差块
        x = self.layer1(x)

        # 上采样和第二层残差块
        x = self.upsample1(x)  # 上采样后通道数保持不变

        x = self.layer2(x)     # 确保输入与 layer2 的期望通道数匹配

        x = self.upsample2(x)  # 上采样后通道数保持不变

        x = self.layer3(x)     # 确保输入与 layer2 的期望通道数匹配
        
        # 中间层的残差块
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        # 最终的归一化、激活和卷积输出层
        x = self.norm_out(x)
        x = self.act_out(x)

        x = self.conv_out(x)[:, :, :, :721, :1440]
        
        return x


class Canglong(nn.Module):
    """
    CAS Canglong PyTorch impl of: `CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction`
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.patchembed2d = ImageToPatch2D(
            img_dims=(721, 1440),
            patch_dims=(4, 4), # 8, 8
            in_channels=4,
            out_channels=embed_dim,
        )
        self.patchembed3d = ImageToPatch3D(
            img_dims=(14, 721, 1440),
            patch_dims=(1, 4, 4),
            in_channels=14,
            out_channels=embed_dim
        )
        self.patchembed4d = ImageToPatch4D(
            img_dims=(7, 4, 721, 1440),
            patch_dims=(4, 2, 4, 4),
            in_channels=7,
            out_channels=embed_dim
        )
        self.encoder3d = Encoder(image_channels=16, latent_dim=96)

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(4, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=(4, 181, 360), output_resolution=(4, 91, 180))
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(4, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(4, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim, (4, 91, 180), (4, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(4, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.patchrecovery2d = RecoveryImage2D((721, 1440), (4, 4), 2 * embed_dim, 4) #8, 8
        self.decoder3d = Decoder(image_channels=16, latent_dim=2 * 96)
        self.patchrecovery3d = RecoveryImage3D(image_size=(16, 721, 1440), 
                                               patch_size=(1, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=16) #2, 8, 8
        self.patchrecovery4d = RecoveryImage4D(image_size=(7, 4, 721, 1440), 
                                               patch_size=(4, 2, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=7)
        self.conv_constant = nn.Conv2d(in_channels=4, out_channels=96, kernel_size=5, stride=4, padding=2)
        self.input_constant = input_constant


    def forward(self, surface, upper_air):
        
        
        constant = self.conv_constant(self.input_constant)
        surface = self.encoder3d(surface)

        upper_air = self.patchembed4d(upper_air)
        

        x = torch.concat([upper_air.squeeze(3), constant.unsqueeze(2), surface], dim=2)

        B, C, Pl, Lat, Lon = x.shape

        x = x.reshape(B, C, -1).transpose(1, 2)
        
        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)

        output_surface = output[:, :, 2:, :, :]
        output_upper_air = output[:, :, 0, :, :]

        #output_surface = self.patchrecovery3d(output_surface)
        output_surface = self.decoder3d(output_surface)

        output_upper_air = self.patchrecovery4d(output_upper_air.unsqueeze(2).unsqueeze(3))

        return output_surface, output_upper_air

## End paste model
        
# 加载模型
print("加载模型...")
import sys
import canglong.embed_old as embed_old
sys.modules['canglong.embed'] = embed_old
import canglong.recovery_old as recovery_old
sys.modules['canglong.recovery'] = recovery_old

model_path = 'F:/model/weather_model_epoch_500.pt'
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("模型加载成功")

# 加载预处理的输入数据

def load_memmap_tensor(path):
    """Load large serialized tensors using memory mapping when available."""
    try:
        tensor = torch.load(path, map_location='cpu', mmap=True)
    except (TypeError, RuntimeError):
        tensor = torch.load(path, map_location='cpu')
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"File {path} does not contain a torch.Tensor (got {type(tensor)})")
    return tensor

print("加载预处理数据...")
surface_input = load_memmap_tensor('I:/ERA5_np/input_surface_norm_test_last100.pt')  # (16, 100, 721, 1440)
upper_air_input = load_memmap_tensor('I:/ERA5_np/input_upper_air_norm_test_last100.pt')  # (7, 4, 100, 721, 1440)
print(f"Surface输入形状: {surface_input.shape}")
print(f"Upper air输入形状: {upper_air_input.shape}")

# 加载气候态数据用于SPEI计算
print("加载气候态数据...")
climate = xr.open_dataset('E:/data/climate_variables_2000_2023_weekly.nc', chunks={'time': 4, 'latitude': 181, 'longitude': 360})

# 定义周号计算函数
def get_week_of_year(date):
    """用于xarray时间维度的周数计算"""
    day_of_year = date.dt.dayofyear
    return ((day_of_year - 1) // 7) + 1

def calculate_week_number(date):
    """计算单个datetime对象的周数（1-52）"""
    day_of_year = date.timetuple().tm_yday
    return ((day_of_year - 1) // 7) + 1

# SPEI计算辅助函数
def calculate_pwm(series):
    n = len(series)
    if n < 3:
        return np.nan, np.nan, np.nan
    sorted_series = np.sort(series)
    F_vals = (np.arange(1, n + 1) - 0.35) / n
    one_minus_F = 1.0 - F_vals
    W0 = np.mean(sorted_series)
    W1 = np.sum(sorted_series * one_minus_F) / n
    W2 = np.sum(sorted_series * (one_minus_F**2)) / n
    return W0, W1, W2

def calculate_loglogistic_params(W0, W1, W2):
    if np.isnan(W0) or np.isnan(W1) or np.isnan(W2):
        return np.nan, np.nan, np.nan
    numerator_beta = (2 * W1) - W0
    denominator_beta = (6 * W1) - W0 - (6 * W2)
    if np.isclose(denominator_beta, 0):
        return np.nan, np.nan, np.nan
    beta = numerator_beta / denominator_beta
    if beta <= 1.0:
        return np.nan, np.nan, np.nan
    try:
        term_gamma1 = gamma_function(1 + (1 / beta))
        term_gamma2 = gamma_function(1 - (1 / beta))
    except ValueError:
        return np.nan, np.nan, np.nan
    denominator_alpha = term_gamma1 * term_gamma2
    if np.isclose(denominator_alpha, 0):
        return np.nan, np.nan, np.nan
    alpha = ((W0 - (2 * W1)) * beta) / denominator_alpha
    if alpha <= 0:
        return np.nan, np.nan, np.nan
    gamma_param = W0 - (alpha * denominator_alpha)
    return alpha, beta, gamma_param

def loglogistic_cdf(x, alpha, beta, gamma_param):
    if np.isnan(alpha) or x <= gamma_param:
        return 1e-9
    term = (alpha / (x - gamma_param))**beta
    if np.isinf(term) or term > 1e18:
        return 1e-9
    cdf_val = 1.0 / (1.0 + term)
    return np.clip(cdf_val, 1e-9, 1.0 - 1e-9)

def cdf_to_spei(P):
    if np.isnan(P): return np.nan
    if P <= 0.0: P = 1e-9
    if P >= 1.0: P = 1.0 - 1e-9
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    if P <= 0.5:
        w = np.sqrt(-2.0 * np.log(P))
        spei = -(w - (c0 + c1 * w + c2 * w**2) / (1 + d1 * w + d2 * w**2 + d3 * w**3))
    else:
        w = np.sqrt(-2.0 * np.log(1.0 - P))
        spei = (w - (c0 + c1 * w + c2 * w**2) / (1 + d1 * w + d2 * w**2 + d3 * w**3))
    return spei

def calculate_spei_for_pixel(historical_D_series, current_D_value):
    if np.isscalar(current_D_value):
        if np.isnan(current_D_value):
            return np.nan
    else:
        if np.all(np.isnan(current_D_value)):
            return np.nan
    valid_historical_D = historical_D_series[~np.isnan(historical_D_series)]
    if len(valid_historical_D) < 10:
        return np.nan
    W0, W1, W2 = calculate_pwm(valid_historical_D)
    if np.isnan(W0):
        return np.nan
    alpha, beta, gamma_p = calculate_loglogistic_params(W0, W1, W2)
    if np.isnan(alpha):
        return np.nan
    P = loglogistic_cdf(current_D_value, alpha, beta, gamma_p)
    spei_val = cdf_to_spei(P)
    return spei_val

def denormalize_surface(normalized_surface):
    """Denormalize surface data"""
    normalized_surface = np.asarray(normalized_surface, dtype=np.float32)
    surface_means = np.array([ordered_var_stats[var_mapping[var]]['mean'] for var in surface_var_names], dtype=np.float32)
    surface_stds = np.array([ordered_var_stats[var_mapping[var]]['std'] for var in surface_var_names], dtype=np.float32)
    surface_means = surface_means.reshape(-1, 1, 1, 1)
    surface_stds = surface_stds.reshape(-1, 1, 1, 1)
    return normalized_surface * surface_stds + surface_means
def data_to_xarray(denormalized_surface, start_date, forecast_weeks=6):
    """Convert denormalized surface data to an xarray Dataset"""
    forecast_dates = [start_date + timedelta(days=(i+1)*7-1) for i in range(forecast_weeks)]
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 359.75, 1440)

    denormalized_surface = np.asarray(denormalized_surface, dtype=np.float32)

    surface_ds = xr.Dataset(coords={
        'variable': surface_var_names,
        'time': forecast_dates,
        'latitude': lat,
        'longitude': lon
    })

    surface_data_array = xr.DataArray(
        denormalized_surface,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={'variable': surface_var_names, 'time': forecast_dates, 'latitude': lat, 'longitude': lon}
    )

    surface_ds['data'] = surface_data_array
    for i, var_name in enumerate(surface_var_names):
        surface_ds[var_name] = surface_data_array.sel(variable=var_name)

    # 单位转换
    surface_ds['2m_temperature'] = (surface_ds['2m_temperature'] - 273.15).astype(np.float32)
    surface_ds['2m_dewpoint_temperature'] = (surface_ds['2m_dewpoint_temperature'] - 273.15).astype(np.float32)

    # 降水转换: m/hr -> mm/day
    m_hr_to_mm_day = 24.0 * 1000.0
    surface_ds['large_scale_rain_rate'] = (
        surface_ds['large_scale_rain_rate'].where(
            surface_ds['large_scale_rain_rate'] >= 0, 0
        ) * m_hr_to_mm_day
    ).astype(np.float32)
    surface_ds['convective_rain_rate'] = (
        surface_ds['convective_rain_rate'].where(
            surface_ds['convective_rain_rate'] >= 0, 0
        ) * m_hr_to_mm_day
    ).astype(np.float32)

    surface_ds['total_precipitation'] = (surface_ds['large_scale_rain_rate'] + surface_ds['convective_rain_rate']).astype(np.float32)

    return surface_ds

def calculate_pet_and_spei(surface_ds, climate_data, input_surface_ds=None, start_pred_idx=0):
    """计算PET和SPEI

    Args:
        surface_ds: 预报数据（6周）
        climate_data: 气候态数据
        input_surface_ds: 输入数据（2周），用于计算前3周的SPEI
        start_pred_idx: 开始计算SPEI的索引（默认0）
    """
    # 计算PET
    t2m_celsius = surface_ds['2m_temperature'].values
    d2m_celsius = surface_ds['2m_dewpoint_temperature'].values

    es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
    ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))

    ratio_ea_es = np.full_like(t2m_celsius, np.nan)
    valid_es_mask = es > 1e-9
    ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
    ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

    pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
    pet = np.maximum(pet, 0)

    surface_ds['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet)

    # 计算D (降水 - 蒸散发)
    D_pred = surface_ds['total_precipitation'] - surface_ds['potential_evapotranspiration']
    D_pred = D_pred.rename({'latitude': 'lat', 'longitude': 'lon'})

    # 如果提供了输入数据，合并以计算前3周的SPEI
    if input_surface_ds is not None:
        # 为输入数据计算PET
        t2m_input = input_surface_ds['2m_temperature'].values
        d2m_input = input_surface_ds['2m_dewpoint_temperature'].values

        es_input = 0.618 * np.exp(17.27 * t2m_input / (t2m_input + 237.3))
        ea_input = 0.618 * np.exp(17.27 * d2m_input / (d2m_input + 237.3))

        ratio_input = np.full_like(t2m_input, np.nan)
        valid_mask = es_input > 1e-9
        ratio_input[valid_mask] = ea_input[valid_mask] / es_input[valid_mask]
        ratio_input = np.clip(ratio_input, None, 1.0)

        pet_input = 4.5 * np.power((1 + t2m_input / 25.0), 2) * (1 - ratio_input)
        pet_input = np.maximum(pet_input, 0)

        input_surface_ds['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet_input)

        # 计算输入数据的D
        D_input = input_surface_ds['total_precipitation'] - input_surface_ds['potential_evapotranspiration']
        D_input = D_input.rename({'latitude': 'lat', 'longitude': 'lon'})

        # 合并输入和预报数据
        D_combined = xr.concat([D_input, D_pred], dim='time')
        start_calc_idx = 2  # 从合并数据的第3个时间点开始计算（即预报的第1周）
    else:
        D_combined = D_pred
        start_calc_idx = 3  # 从预报数据的第4个时间点开始计算

    D_hist = climate_data['tp'] - climate_data['pet']

    # 计算SPEI
    if len(D_combined.time) < 4:
        print(f"警告: 时间点不足4个（当前{len(D_combined.time)}个），无法计算SPEI")
        return surface_ds

    spei_pred_list = []
    pred_week_numbers = get_week_of_year(D_combined.time)
    hist_week_numbers = get_week_of_year(D_hist.time)

    # 从start_calc_idx + 1开始计算（确保有至少3周历史数据）
    for i in range(max(3, start_calc_idx + 1), len(D_combined.time)):
        # 累积当前及前3周的D值（从合并数据中）
        curr_week_accum = sum([D_combined.isel(time=i-j) for j in range(4) if i-j >= 0])
        curr_week_accum = curr_week_accum.load()
        curr_week_num = pred_week_numbers.isel(time=i).item()

        # 提取历史同期数据
        hist_4week_accum_list = []
        hist_years = np.unique(D_hist.time.dt.year)

        for year in hist_years:
            year_data = D_hist.where(D_hist.time.dt.year == year, drop=True)
            year_weeks = hist_week_numbers.where(D_hist.time.dt.year == year, drop=True)
            week_indices = np.where(year_weeks == curr_week_num)[0]
            if len(week_indices) > 0:
                week_idx = week_indices[0]
                if week_idx >= 3:
                    accum_D = sum([year_data.isel(time=week_idx-j) for j in range(4)])
                    hist_4week_accum_list.append(accum_D)

        if hist_4week_accum_list:
            hist_4week_accum = xr.concat(hist_4week_accum_list, dim='time')
            if hist_4week_accum.size > 0:
                hist_4week_accum = hist_4week_accum.load()
        else:
            hist_4week_accum = xr.DataArray(
                np.zeros((0,) + D_pred.isel(time=0).shape),
                coords={'time': [], **{dim: D_pred[dim] for dim in D_pred.dims if dim != 'time'}},
                dims=D_pred.dims
            )

        if len(hist_4week_accum.time) < 10:
            spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
        elif np.isnan(curr_week_accum).all():
            spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
        else:
            spei_map = xr.apply_ufunc(
                calculate_spei_for_pixel,
                hist_4week_accum,
                curr_week_accum,
                input_core_dims=[['time'], []],
                output_core_dims=[[]],
                exclude_dims=set(('time',)),
                vectorize=True,
                output_dtypes=[float],
                keep_attrs=True
            )

        spei_pred_list.append(spei_map)

    if spei_pred_list:
        spei_pred = xr.concat(spei_pred_list, dim='time')

        # 如果提供了输入数据，SPEI从合并数据的第3个点开始（对应预报的第1周）
        # 否则从第4个点开始（对应预报的第4周）
        if input_surface_ds is not None:
            # 有输入数据：计算了全部6周预报的SPEI（从合并数据的索引3开始）
            spei_pred = spei_pred.assign_coords(time=D_pred.time[:len(spei_pred_list)])
        else:
            # 无输入数据：只计算了后3周的SPEI
            spei_pred = spei_pred.assign_coords(time=D_pred.time[start_pred_idx + 3:])

        surface_ds['spei'] = spei_pred.rename({'lat': 'latitude', 'lon': 'longitude'})

    return surface_ds

# 主循环：处理2022-2023年每一周
output_dir = 'Z:/Data/hindcast_2022_2023'
os.makedirs(output_dir, exist_ok=True)

# 定义2022年的起始周（从索引4开始，即2022-02-26至03-04）
start_year = 2022
start_week = 4  # 索引4对应2022年第9周 (2.26-3.4，2022是平年)
total_weeks = 100  # 数据中有100周

print(f"\n开始批量回报 2022-2023 年...")
print(f"起始周: 索引{start_week}，日期2022-02-26至2022-03-04")
print(f"总周数: {total_weeks - start_week}")

# 记录所有处理的文件
processed_files = []

with torch.no_grad():
    for week_idx in tqdm(range(start_week, total_weeks), desc="回报进度"):
        try:
            # 计算日期 (2022-01-29是第0周)
            base_date = datetime(2022, 1, 29)
            current_date = base_date + timedelta(weeks=week_idx)
            year = current_date.year
            week_of_year = calculate_week_number(current_date)

            # 提取前两周的输入数据 (形成2周输入)
            if week_idx < 2:
                continue  # 跳过前两周，因为没有足够的历史数据

            # 提取输入数据：需要前两周（week_idx-2 和 week_idx-1）
            input_surface_tensor = surface_input[:, [week_idx-2, week_idx-1], :, :].unsqueeze(0).to(device=device, dtype=torch.float32)
            input_upper_air_tensor = upper_air_input[:, :, [week_idx-2, week_idx-1], :, :].unsqueeze(0).to(device=device, dtype=torch.float32)

            # 滚动预报6周
            current_input_surface = input_surface_tensor
            current_input_upper_air = input_upper_air_tensor

            all_surface_predictions = []
            all_upper_air_predictions = []

            for week in range(forecast_weeks):
                output_surface, output_upper_air = model(current_input_surface, current_input_upper_air)
                all_surface_predictions.append(output_surface[:, :, 0:1, :, :])
                all_upper_air_predictions.append(output_upper_air[:, :, :, 0:1, :, :])

                if week < forecast_weeks - 1:
                    new_input_surface = torch.cat([
                        current_input_surface[:, :, 1:2, :, :],
                        output_surface[:, :, 0:1, :, :]
                    ], dim=2)
                    new_input_upper_air = torch.cat([
                        current_input_upper_air[:, :, :, 1:2, :, :],
                        output_upper_air[:, :, :, 0:1, :, :]
                    ], dim=3)
                    current_input_surface = new_input_surface
                    current_input_upper_air = new_input_upper_air

            # 合并预测结果
            all_weeks_surface_predictions = torch.cat(all_surface_predictions, dim=2)

            # 反标准化预测结果
            surface_predictions_np = all_weeks_surface_predictions.cpu().numpy()
            denormalized_surface = denormalize_surface(surface_predictions_np[0])

            # 反标准化输入数据（用于SPEI计算）
            input_surface_np = input_surface_tensor.cpu().numpy()
            denormalized_input = denormalize_surface(input_surface_np[0])

            # 转换预测为xarray
            forecast_start_date = current_date + timedelta(days=1)
            surface_ds = data_to_xarray(denormalized_surface, forecast_start_date, forecast_weeks)

            # 转换输入为xarray（需要构造时间坐标）
            # current_date是当前周的开始日期，前两周分别结束于current_date-8天和current_date-1天
            input_week1_start = current_date - timedelta(days=14)  # 第一周开始日期
            input_surface_ds = data_to_xarray(denormalized_input, input_week1_start, forecast_weeks=2)

            # 计算PET和SPEI（传入输入数据）
            surface_ds = calculate_pet_and_spei(surface_ds, climate, input_surface_ds=input_surface_ds, start_pred_idx=0)

            # 只保留需要的4个变量
            output_ds = xr.Dataset({
                '2m_temperature': surface_ds['2m_temperature'],
                '2m_dewpoint_temperature': surface_ds['2m_dewpoint_temperature'],
                'total_precipitation': surface_ds['total_precipitation'],
                'spei': surface_ds['spei'] if 'spei' in surface_ds else xr.full_like(surface_ds['2m_temperature'], np.nan)
            })

            # 添加元数据
            output_ds.attrs['description'] = 'Hindcast for temperature, dewpoint, precipitation, and SPEI'
            output_ds.attrs['year'] = year
            output_ds.attrs['week_of_year'] = week_of_year
            output_ds.attrs['start_date'] = current_date.strftime('%Y-%m-%d')
            output_ds.attrs['forecast_start'] = forecast_start_date.strftime('%Y-%m-%d')

            # 保存文件
            filename = f"hindcast_{year}_week{week_of_year:02d}_surface_{current_date.strftime('%Y-%m-%d')}.nc"
            output_path = os.path.join(output_dir, filename)
            output_ds.to_netcdf(output_path)

            processed_files.append({
                'year': year,
                'week': week_of_year,
                'start_date': current_date.strftime('%Y-%m-%d'),
                'surface_file': output_path
            })

            # 每10周清理一次内存
            if (week_idx - start_week) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n处理第 {week_idx} 周 ({current_date.strftime('%Y-%m-%d')}) 失败: {str(e)}")
            continue

# 保存索引文件
index_df = pd.DataFrame(processed_files)
index_path = os.path.join(output_dir, 'hindcast_index_2022_2023.csv')
index_df.to_csv(index_path, index=False)

print(f"\n完成！")
print(f"成功处理 {len(processed_files)} 周")
print(f"文件保存在: {output_dir}")
print(f"索引文件: {index_path}")
