"""
CAS-Canglong V2 模型
在V1基础上添加风向感知的Mask & Combine窗口移位策略
"""

import os
import torch
from torch import nn
import numpy as np
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F

from .earth_position import calculate_position_bias_indices
from .shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from .embed import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D
from .recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
from .pad import calculate_padding_3d, calculate_padding_2d
from .crop import center_crop_2d, center_crop_3d
from .wind_direction import WindDirectionProcessor
from .wind_aware_mask import WindAwareAttentionMaskGenerator
from .wind_aware_block import WindAwareEarthSpecificBlock
from .helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish


class UpSample(nn.Module):
    """Up-sampling operation."""

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
    """Down-sampling operation"""

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


class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage with wind-aware shifting"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_wind_aware_shift=True,
                 wind_shift_mode='regional', num_regions=(4, 8), wind_shift_scale=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.wind_aware_mask_generator = WindAwareAttentionMaskGenerator(input_resolution, window_size) if use_wind_aware_shift else None

        self.blocks = nn.ModuleList([
            WindAwareEarthSpecificBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else (window_size[0]//2, window_size[1]//2, window_size[2]//2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_wind_aware_shift=use_wind_aware_shift,
                wind_shift_mode=wind_shift_mode,
                num_regions=num_regions,
                wind_shift_scale=wind_shift_scale
            )
            for i in range(depth)
        ])

    def forward(self, x, wind_direction_id=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, wind_direction_id)
        return x


class Encoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(Encoder, self).__init__()
        channels = [64, 64, 64, 128, 128]
        attn_resolutions = [2]
        num_res_blocks = 1
        resolution = 256

        self.conv_in = nn.Conv3d(image_channels, channels[0], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks, resolution, attn_resolutions)
        self.downsample1 = DownSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks, resolution // 2, attn_resolutions)
        self.downsample2 = DownSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks, resolution // 4, attn_resolutions)
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
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
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)[:, :, :, :181, :360]
        return x


class Decoder(nn.Module):
    def __init__(self, image_channels=14, latent_dim=64):
        super(Decoder, self).__init__()
        channels = [128, 128, 64, 64]
        num_res_blocks = 1

        self.conv_in = nn.Conv3d(latent_dim, channels[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(channels[0], channels[1], num_res_blocks)
        self.upsample1 = UpSampleBlock(channels[1])
        self.layer2 = self._make_layer(channels[1], channels[2], num_res_blocks)
        self.upsample2 = UpSampleBlock(channels[2])
        self.layer3 = self._make_layer(channels[2], channels[3], num_res_blocks)
        self.mid_block1 = ResidualBlock(channels[3], channels[3])
        self.mid_block2 = ResidualBlock(channels[3], channels[3])
        self.norm_out = GroupNorm(channels[3])
        self.act_out = Swish()
        self.conv_out = nn.ConvTranspose3d(channels[3], image_channels, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def _make_layer(self, in_channels, out_channels, num_res_blocks):
        layers = [ResidualBlock(in_channels, out_channels) for _ in range(num_res_blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.upsample1(x)
        x = self.layer2(x)
        x = self.upsample2(x)
        x = self.layer3(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)[:, :, :, :721, :1440]
        return x


class CanglongV2(nn.Module):
    """
    CAS Canglong V2: 风向感知的3D Transformer天气预测模型

    在V1基础上添加：
    - 风向感知的Mask & Combine窗口移位策略
    - 分区域(4x8=32区域)独立风向计算
    - Swin固定移位 + 风向额外移位的双重移位机制
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()

        # 风向处理器
        self.wind_direction_processor = WindDirectionProcessor(window_size=(4, 4))

        self.patchembed4d = ImageToPatch4D(
            img_dims=(10, 5, 2, 721, 1440),
            patch_dims=(2, 2, 4, 4),
            in_channels=10,
            out_channels=embed_dim
        )
        self.encoder3d = Encoder(image_channels=26, latent_dim=96)

        # 风向移位参数：分区域模式，4x8=32个区域
        wind_shift_mode = 'regional'
        num_regions = (4, 8)
        wind_shift_scale = 2

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
            use_wind_aware_shift=True,
            wind_shift_mode=wind_shift_mode,
            num_regions=num_regions,
            wind_shift_scale=wind_shift_scale
        )
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=(6, 181, 360), output_resolution=(6, 91, 180))
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
            use_wind_aware_shift=True,
            wind_shift_mode=wind_shift_mode,
            num_regions=num_regions,
            wind_shift_scale=wind_shift_scale
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
            use_wind_aware_shift=True,
            wind_shift_mode=wind_shift_mode,
            num_regions=num_regions,
            wind_shift_scale=wind_shift_scale
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim, (6, 91, 180), (6, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
            use_wind_aware_shift=True,
            wind_shift_mode=wind_shift_mode,
            num_regions=num_regions,
            wind_shift_scale=wind_shift_scale
        )
        self.decoder3d = Decoder(image_channels=26, latent_dim=2 * 96)
        self.patchrecovery4d = RecoveryImage4D(
            image_size=(10, 5, 1, 721, 1440),
            patch_size=(2, 1, 4, 4),
            input_channels=2 * embed_dim,
            output_channels=10,
            target_size=(10, 5, 1, 721, 1440)
        )

        self.conv_constant = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=4, padding=2)

        # 加载 Earth constant 数据
        earth_path = os.path.join(os.path.dirname(__file__), '..', 'constant_masks', 'Earth.pt')
        self.input_constant = torch.load(earth_path, weights_only=True)

    def forward(self, surface, upper_air, return_wind_info=False):
        """
        前向传播

        参数:
            surface: 表面数据 (B, 26, 2, 721, 1440)
            upper_air: 高空数据 (B, 10, 5, 2, 721, 1440)
            return_wind_info: 是否返回风向信息用于监控

        返回:
            output_surface, output_upper_air
            如果return_wind_info=True, 还返回wind_direction_id
        """
        # 计算风向ID
        wind_direction_id = self.wind_direction_processor(surface, upper_air)

        # 确保在正确的设备上
        if self.input_constant.device != surface.device:
            self.input_constant = self.input_constant.to(surface.device)
        constant = self.conv_constant(self.input_constant)

        surface_encoded = self.encoder3d(surface)
        upper_air_encoded = self.patchembed4d(upper_air)

        x = torch.concat([upper_air_encoded.squeeze(3),
                          surface_encoded,
                          constant.unsqueeze(2)], dim=2)

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        # 传递风向ID到各层（风向感知的双重移位）
        x = self.layer1(x, wind_direction_id)
        skip = x

        x = self.downsample(x)
        x = self.layer2(x, wind_direction_id)
        x = self.layer3(x, wind_direction_id)
        x = self.upsample(x)
        x = self.layer4(x, wind_direction_id)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 3:5, :, :]
        output_upper_air = output[:, :, 0:3, :, :]

        output_surface = self.decoder3d(output_surface)
        output_upper_air = self.patchrecovery4d(output_upper_air.unsqueeze(3))

        if return_wind_info:
            return output_surface, output_upper_air, wind_direction_id
        return output_surface, output_upper_air
