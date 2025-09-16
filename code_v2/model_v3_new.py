import torch
from torch import nn
import numpy as np
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import h5py as h5
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from canglong.earth_position import calculate_position_bias_indices
from canglong.shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from canglong.embed import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D
from canglong.recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
from canglong.pad import calculate_padding_3d, calculate_padding_2d
from canglong.crop import center_crop_2d, center_crop_3d
from canglong.wind_direction import WindDirectionProcessor
from canglong.wind_aware_mask import WindAwareAttentionMaskGenerator
from canglong.wind_aware_block import WindAwareEarthSpecificBlock
from canglong.helper import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

# 使用绝对路径加载constant_masks
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
constant_masks_path = os.path.join(project_root, 'constant_masks', 'Earth.pt')
input_constant = torch.load(constant_masks_path, weights_only=False).cuda()


class PhysicalConstraints(nn.Module):
    """
    Physical constraints module for weather prediction model.
    Implements water balance, energy balance, and hydrostatic balance constraints.
    """
    
    def __init__(self, surface_mean, surface_std, upper_mean, upper_std, delta_t=7*24*3600):
        """
        Args:
            surface_mean: Mean values for surface variables (17, 721, 1440)
            surface_std: Std values for surface variables (17, 721, 1440)
            upper_mean: Mean values for upper air variables (7, 5, 721, 1440)
            upper_std: Std values for upper air variables (7, 5, 721, 1440)
            delta_t: Time step in seconds (default: 1 week = 7*24*3600 seconds)
        """
        super().__init__()
        self.register_buffer('surface_mean', surface_mean)
        self.register_buffer('surface_std', surface_std)
        self.register_buffer('upper_mean', upper_mean)
        self.register_buffer('upper_std', upper_std)
        self.delta_t = delta_t
        
        # Physical constants
        self.L_v = 2.5e6  # Latent heat of vaporization (J/kg)
        self.R_d = 287    # Gas constant for dry air (J/(kg·K))
        self.g = 9.81     # Gravitational acceleration (m/s²)
    
    def denormalize_surface(self, normalized_data):
        """Denormalize surface data to physical units"""
        # Ensure surface_std and surface_mean are properly broadcasted
        # normalized_data shape: (B, 17, time, lat, lon)
        # surface_std/mean shape: (17, lat, lon) -> need to add batch and time dims
        surface_std = self.surface_std.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
        surface_mean = self.surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
        return normalized_data * surface_std + surface_mean
    
    def denormalize_upper(self, normalized_data):
        """Denormalize upper air data to physical units"""
        # Ensure upper_std and upper_mean are properly broadcasted
        # normalized_data shape: (B, 7, levels, time, lat, lon)
        # upper_std/mean shape: (7, levels, lat, lon) -> need to add batch and time dims
        upper_std = self.upper_std.unsqueeze(0).unsqueeze(3)  # (1, 7, levels, 1, lat, lon)
        upper_mean = self.upper_mean.unsqueeze(0).unsqueeze(3)  # (1, 7, levels, 1, lat, lon)
        return normalized_data * upper_std + upper_mean
    
    def water_balance_loss(self, input_surface, output_surface):
        """
        Calculate water balance constraint loss.
        Water balance: ΔSoil_water = P_total - E
        
        Args:
            input_surface: Input surface variables (B, 17, time, lat, lon)
            output_surface: Output surface variables (B, 17, time, lat, lon)
        """
        # Denormalize to physical units
        input_physical = self.denormalize_surface(input_surface)
        output_physical = self.denormalize_surface(output_surface)
        
        # Extract relevant variables (assuming single time output)
        # Variable indices based on CLAUDE.md:
        # 0: large_scale_rain_rate, 1: convective_rain_rate
        # 10: surface_latent_heat_flux
        # 13: volumetric_soil_water_layer
        
        # Soil water change (index 13)
        delta_soil_water = output_physical[:, 13, 0, :, :] - input_physical[:, 13, -1, :, :]
        
        # Total precipitation (indices 0 and 1) - convert from kg m^-2 s^-1 to total amount
        large_scale_rain = output_physical[:, 0, 0, :, :]
        convective_rain = output_physical[:, 1, 0, :, :]
        p_total = (large_scale_rain + convective_rain) * self.delta_t
        
        # Evaporation from latent heat flux (index 10) - convert J m^-2 to water amount
        latent_heat_flux = output_physical[:, 10, 0, :, :]
        evaporation = latent_heat_flux / self.L_v * self.delta_t
        
        # Water balance residual
        residual_water = delta_soil_water - (p_total - evaporation)
        
        # Return MSE loss
        return F.mse_loss(residual_water, torch.zeros_like(residual_water))
    
    def energy_balance_loss(self, output_surface):
        """
        Calculate energy balance constraint loss.
        Energy balance: SW_net - LW_net = SHF + LHF
        
        Args:
            output_surface: Output surface variables (B, 17, time, lat, lon)
        """
        # Denormalize to physical units
        output_physical = self.denormalize_surface(output_surface)
        
        # Extract relevant variables (assuming single time output)
        # Variable indices based on CLAUDE.md:
        # 4: top_net_solar_radiation_clear_sky (SW_net)
        # 9: mean_top_net_long_wave_radiation_flux (LW_net)
        # 10: surface_latent_heat_flux (LHF)
        # 11: surface_sensible_heat_flux (SHF)
        
        sw_net = output_physical[:, 4, 0, :, :]  # Net solar radiation (J m^-2)
        lw_net = output_physical[:, 9, 0, :, :]  # Net longwave radiation (W m^-2)
        shf = output_physical[:, 11, 0, :, :]    # Sensible heat flux (J m^-2)
        lhf = output_physical[:, 10, 0, :, :]    # Latent heat flux (J m^-2)
        
        # Energy balance residual
        # Net absorbed energy - Net released energy
        residual_energy = (sw_net - lw_net) - (shf + lhf)
        
        # Return MSE loss
        return F.mse_loss(residual_energy, torch.zeros_like(residual_energy))
    
    def hydrostatic_balance_loss(self, output_upper_air):
        """
        Calculate hydrostatic balance constraint loss.
        Hydrostatic balance: Δφ = R_d * T_avg * ln(p1/p2)
        
        Args:
            output_upper_air: Output upper air variables (B, 7, levels, time, lat, lon)
        """
        # Denormalize to physical units
        output_physical = self.denormalize_upper(output_upper_air)
        
        # Extract relevant variables (assuming single time output)
        # Variable indices based on CLAUDE.md:
        # 0: Geopotential (φ)
        # 5: Temperature (T)
        # Pressure levels: 200, 300, 500, 700, 850 hPa (indices 0-4)
        
        # Calculate between 850 hPa (index 4) and 700 hPa (index 3)
        phi_850 = output_physical[:, 0, 4, 0, :, :]  # Geopotential at 850 hPa (m^2 s^-2)
        phi_700 = output_physical[:, 0, 3, 0, :, :]  # Geopotential at 700 hPa
        temp_850 = output_physical[:, 5, 4, 0, :, :] # Temperature at 850 hPa (K)
        temp_700 = output_physical[:, 5, 3, 0, :, :] # Temperature at 700 hPa
        
        # Model-predicted geopotential thickness
        delta_phi_model = phi_700 - phi_850
        
        # Physically-calculated geopotential thickness
        temp_avg = (temp_700 + temp_850) / 2
        delta_phi_physical = self.R_d * temp_avg * torch.log(torch.tensor(850.0/700.0, device=temp_avg.device))
        
        # Hydrostatic balance residual
        residual_hydrostatic = delta_phi_model - delta_phi_physical
        
        # Return MSE loss
        return F.mse_loss(residual_hydrostatic, torch.zeros_like(residual_hydrostatic))


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


class BasicLayer(nn.Module):
    """A basic 3D Transformer layer for one stage"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_wind_aware_shift=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 风向感知的注意力掩码生成器
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
                use_wind_aware_shift=use_wind_aware_shift
            )
            for i in range(depth)
        ])

    def forward(self, x, wind_direction_id=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, wind_direction_id)
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
        self.conv_out = nn.ConvTranspose3d(channels[3], image_channels, kernel_size=(2, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

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


class CanglongV3(nn.Module):
    """
    CAS Canglong PyTorch impl of: `CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction`
    Version 3 with physical constraints (water balance, energy balance, hydrostatic balance)
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12),
                 surface_mean=None, surface_std=None, upper_mean=None, upper_std=None,
                 lambda_water=1e-11, lambda_energy=1e-12, lambda_pressure=1e-6):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        
        # Physical constraint weights
        self.lambda_water = lambda_water
        self.lambda_energy = lambda_energy
        self.lambda_pressure = lambda_pressure
        
        # 风向处理器
        self.wind_direction_processor = WindDirectionProcessor(window_size=(4, 4))
        
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
            img_dims=(7, 5, 2, 721, 1440),
            patch_dims=(2, 2, 4, 4),
            in_channels=7,
            out_channels=embed_dim
        )
        self.encoder3d = Encoder(image_channels=17, latent_dim=96)

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
            use_wind_aware_shift=True  # 启用风向感知的窗口交换
        )
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=(6, 181, 360), output_resolution=(6, 91, 180))
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
            use_wind_aware_shift=True  # 启用风向感知的窗口交换
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(6, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
            use_wind_aware_shift=True  # 启用风向感知的窗口交换
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim, (6, 91, 180), (6, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(6, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
            use_wind_aware_shift=True  # 启用风向感知的窗口交换
        )
        self.patchrecovery2d = RecoveryImage2D((721, 1440), (4, 4), 2 * embed_dim, 4) #8, 8
        self.decoder3d = Decoder(image_channels=17, latent_dim=2 * 96)
        self.patchrecovery3d = RecoveryImage3D(image_size=(16, 721, 1440), 
                                               patch_size=(1, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=16) #2, 8, 8
        self.patchrecovery4d = RecoveryImage4D(image_size=(7, 5, 1, 721, 1440), 
                                               patch_size=(2, 1, 4, 4), 
                                               input_channels=2 * embed_dim, 
                                               output_channels=7,
                                               target_size=(7, 5, 1, 721, 1440))
        

        self.conv_constant = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=4, padding=2)
        self.input_constant = input_constant
        
        # Initialize physical constraints module
        if surface_mean is not None and surface_std is not None and upper_mean is not None and upper_std is not None:
            self.physical_constraints = PhysicalConstraints(
                surface_mean, surface_std, upper_mean, upper_std
            )
        else:
            self.physical_constraints = None
            print("Warning: Physical constraints not initialized. Normalization parameters not provided.")


    def forward(self, surface, upper_air, target_surface=None, target_upper_air=None, return_losses=False):        
        
        # 计算风向ID
        wind_direction_id = self.wind_direction_processor(surface, upper_air)
        
        constant = self.conv_constant(self.input_constant)
        surface_encoded = self.encoder3d(surface)

        upper_air_encoded = self.patchembed4d(upper_air)

        
        x = torch.concat([upper_air_encoded.squeeze(3), 
                          surface_encoded, 
                          constant.unsqueeze(2)], dim=2)

        
        B, C, Pl, Lat, Lon = x.shape

        x = x.reshape(B, C, -1).transpose(1, 2)
        
        # 传递风向ID到各层
        x = self.layer1(x, wind_direction_id) #revise

        skip = x

        x = self.downsample(x)
        x = self.layer2(x, wind_direction_id)
        x = self.layer3(x, wind_direction_id)
        x = self.upsample(x)
        x = self.layer4(x, wind_direction_id)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 3:5, :, :]  #  四五层是surface
        output_upper_air = output[:, :, 0:3, :, :]  # 前三层是upper air


        output_surface = self.decoder3d(output_surface)
        output_upper_air = self.patchrecovery4d(output_upper_air.unsqueeze(3))
        
        # Calculate physical constraint losses if requested and physical constraints are initialized
        if return_losses and self.physical_constraints is not None and target_surface is not None:
            losses = {}
            
            # Calculate MSE losses
            if target_surface is not None:
                losses['mse_surface'] = F.mse_loss(output_surface, target_surface)
            if target_upper_air is not None:
                losses['mse_upper_air'] = F.mse_loss(output_upper_air, target_upper_air)
            
            # Calculate physical constraint losses
            losses['water_balance'] = self.physical_constraints.water_balance_loss(surface, output_surface)
            losses['energy_balance'] = self.physical_constraints.energy_balance_loss(output_surface)
            losses['hydrostatic_balance'] = self.physical_constraints.hydrostatic_balance_loss(output_upper_air)
            
            # Calculate total loss
            total_loss = losses.get('mse_surface', 0) + losses.get('mse_upper_air', 0)
            total_loss += self.lambda_water * losses['water_balance']
            total_loss += self.lambda_energy * losses['energy_balance']
            total_loss += self.lambda_pressure * losses['hydrostatic_balance']
            losses['total'] = total_loss
            
            return output_surface, output_upper_air, losses
        
        return output_surface, output_upper_air


def calculate_water_balance_loss(input_surface_normalized, output_surface_normalized, 
                                surface_mean, surface_std, delta_t=7*24*3600):
    """
    计算水量平衡损失
    水量平衡: ΔSoil_water = P_total - E
    
    Args:
        input_surface_normalized: 标准化的输入表面变量 (B, 17, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 17, time, lat, lon)
        surface_mean: 表面变量均值 (17, lat, lon)
        surface_std: 表面变量标准差 (17, lat, lon)
        delta_t: 时间步长（秒）
    """
    # 反标准化到物理单位
    surface_mean = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
    surface_std = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, lat, lon)
    
    input_physical = input_surface_normalized * surface_std + surface_mean
    output_physical = output_surface_normalized * surface_std + surface_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 0: large_scale_rain_rate, 1: convective_rain_rate
    # 10: surface_latent_heat_flux
    # 13: volumetric_soil_water_layer
    
    # 土壤水变化量
    delta_soil_water = output_physical[:, 13, 0, :, :] - input_physical[:, 13, -1, :, :]
    
    # 总降水量 - 从 kg m^-2 s^-1 转换为总量
    large_scale_rain = output_physical[:, 0, 0, :, :]
    convective_rain = output_physical[:, 1, 0, :, :]
    p_total = (large_scale_rain + convective_rain) * delta_t
    
    # 蒸发量 - 从 J m^-2 转换为水量
    L_v = 2.5e6  # 汽化潜热 (J/kg)
    latent_heat_flux = output_physical[:, 10, 0, :, :]
    evaporation = latent_heat_flux / L_v * delta_t
    
    # 水量平衡残差
    residual_water = delta_soil_water - (p_total - evaporation)
    
    return torch.nn.functional.mse_loss(residual_water, torch.zeros_like(residual_water))


def calculate_energy_balance_loss(output_surface_normalized, surface_mean, surface_std):
    """
    计算能量平衡损失
    能量平衡: SW_net - LW_net = SHF + LHF
    
    Args:
        output_surface_normalized: 标准化的输出表面变量 (B, 17, time, lat, lon)
        surface_mean: 表面变量均值 (17, lat, lon)
        surface_std: 表面变量标准差 (17, lat, lon)
    """
    # 反标准化到物理单位
    surface_mean = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
    surface_std = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, lat, lon)
    output_physical = output_surface_normalized * surface_std + surface_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 4: top_net_solar_radiation_clear_sky (SW_net)
    # 9: mean_top_net_long_wave_radiation_flux (LW_net)
    # 10: surface_latent_heat_flux (LHF)
    # 11: surface_sensible_heat_flux (SHF)
    
    sw_net = output_physical[:, 4, 0, :, :]  # 净太阳辐射 (J m^-2)
    lw_net = output_physical[:, 9, 0, :, :]  # 净长波辐射 (W m^-2)
    shf = output_physical[:, 11, 0, :, :]    # 感热通量 (J m^-2)
    lhf = output_physical[:, 10, 0, :, :]    # 潜热通量 (J m^-2)
    
    # 能量平衡残差
    residual_energy = (sw_net - lw_net) - (shf + lhf)
    
    return torch.nn.functional.mse_loss(residual_energy, torch.zeros_like(residual_energy))


def calculate_hydrostatic_balance_loss(output_upper_normalized, upper_mean, upper_std):
    """
    计算静力平衡损失
    静力平衡: Δφ = R_d * T_avg * ln(p1/p2)
    
    Args:
        output_upper_normalized: 标准化的输出高空变量 (B, 7, levels, time, lat, lon)
        upper_mean: 高空变量均值 (7, levels, lat, lon)
        upper_std: 高空变量标准差 (7, levels, lat, lon)
    """
    # 反标准化到物理单位
    upper_mean = upper_mean.unsqueeze(0).unsqueeze(3)  # (1, 7, levels, 1, lat, lon)
    upper_std = upper_std.unsqueeze(0).unsqueeze(3)    # (1, 7, levels, 1, lat, lon)
    output_physical = output_upper_normalized * upper_std + upper_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 0: Geopotential (φ)
    # 5: Temperature (T)
    # 压力层: 200, 300, 500, 700, 850 hPa (索引 0-4)
    
    # 计算 850 hPa (索引 4) 和 700 hPa (索引 3) 之间的静力平衡
    phi_850 = output_physical[:, 0, 4, 0, :, :]  # 850 hPa 位势 (m^2 s^-2)
    phi_700 = output_physical[:, 0, 3, 0, :, :]  # 700 hPa 位势
    temp_850 = output_physical[:, 5, 4, 0, :, :] # 850 hPa 温度 (K)
    temp_700 = output_physical[:, 5, 3, 0, :, :] # 700 hPa 温度
    
    # 模型预测的位势厚度
    delta_phi_model = phi_700 - phi_850
    
    # 物理计算的位势厚度
    R_d = 287  # 干空气气体常数 (J/(kg·K))
    temp_avg = (temp_700 + temp_850) / 2
    delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(850.0/700.0, device=temp_avg.device))
    
    # 静力平衡残差
    residual_hydrostatic = delta_phi_model - delta_phi_physical
    
    return torch.nn.functional.mse_loss(residual_hydrostatic, torch.zeros_like(residual_hydrostatic))


def load_normalization_arrays(json_path, verbose=False):
    """
    从JSON文件加载标准化参数数组
    
    Args:
        json_path: JSON文件路径
        verbose: 是否打印详细信息，默认False（静默加载）
    
    Returns:
        [surface_mean, surface_std, upper_mean, upper_std]: 四个numpy数组
        - surface_mean: (17, 721, 1440)
        - surface_std: (17, 721, 1440) 
        - upper_mean: (7, 5, 721, 1440)
        - upper_std: (7, 5, 721, 1440)
    """
    # 变量定义 - 必须与模型通道顺序严格对应
    surf_vars = ['lsrr','crr','tciw','tcc','tsrc','u10','v10','d2m',
                 't2m','avg_tnlwrf','slhf','sshf','sp','swvl','msl','siconc','sst']
    upper_vars = ['z','w','u','v','cc','t','q']
    levels = [200, 300, 500, 700, 850]
    
    def load_statistics_from_json(json_path, verbose=True):
        """加载JSON格式的统计数据"""

        with open(json_path, 'r') as f:
            stats_data = json.load(f)
        
        if verbose:
            print("JSON数据加载成功")
            print(f"Surface变量数: {len(stats_data['surface'])}")
            print(f"Upper air变量数: {len(stats_data['upper_air'])}")
        return stats_data

    def create_surface_arrays(surface_data, verbose=True):
        """创建Surface变量的mean/std数组
        
        返回:
            surface_mean: (17, 721, 1440)
            surface_std: (17, 721, 1440)
        """
        
        if verbose:
            print("创建Surface数组...")
        
        # 初始化数组
        surface_mean = np.zeros((17, 721, 1440), dtype=np.float32)
        surface_std = np.ones((17, 721, 1440), dtype=np.float32)  # 默认std为1，避免除零
        
        # 按固定顺序填充数据
        for i, var_name in enumerate(surf_vars):

            mean_val = surface_data[var_name]['mean']
            std_val = surface_data[var_name]['std']
            
            # 广播到所有空间位置
            surface_mean[i, :, :] = mean_val
            surface_std[i, :, :] = max(std_val, 1e-8)  # 避免std为0
            
            if verbose:
                print(f"  {i:2d}. {var_name:15s}: mean={mean_val:12.8f}, std={std_val:12.8f}")

        
        if verbose:
            print(f"Surface数组创建完成: {surface_mean.shape}")
        return surface_mean[None, :, None, :, :], surface_std[None, :, None, :, :]

    def create_upper_air_arrays(upper_air_data, verbose=True):
        """创建Upper air变量的mean/std数组
        
        返回:
            upper_air_mean: (7, 5, 721, 1440)
            upper_air_std: (7, 5, 721, 1440)
        """
        
        if verbose:
            print("创建Upper air数组...")
        
        # 初始化数组
        upper_air_mean = np.zeros((7, 5, 721, 1440), dtype=np.float32)
        upper_air_std = np.ones((7, 5, 721, 1440), dtype=np.float32)  # 默认std为1
        
        # 按固定顺序填充数据
        for i, var_name in enumerate(upper_vars):
            if verbose:
                print(f"  处理变量 {i}. {var_name}")
            for j, level in enumerate(levels):
                level_str = str(level)
                mean_val = upper_air_data[var_name][level_str]['mean']
                std_val = upper_air_data[var_name][level_str]['std']
                
                # 广播到所有空间位置
                upper_air_mean[i, j, :, :] = mean_val
                upper_air_std[i, j, :, :] = max(std_val, 1e-8)  # 避免std为0
                
                if verbose:
                    print(f"    {level:3d}hPa: mean={mean_val:12.6f}, std={std_val:12.6f}")

        return upper_air_mean[None, :, :, None, :, :], upper_air_std[None, :, :, None, :, :]
    
    try:
        # 1. 加载JSON数据
        stats_data = load_statistics_from_json(json_path, verbose=verbose)
        
        # 2. 创建Surface数组
        surface_mean, surface_std = create_surface_arrays(stats_data['surface'], verbose=verbose)
        
        # 3. 创建Upper air数组
        upper_mean, upper_std = create_upper_air_arrays(stats_data['upper_air'], verbose=verbose)
        
        return [surface_mean, surface_std, upper_mean, upper_std]
        
    except Exception as e:
        print(f"加载标准化数组时出错: {str(e)}")
        raise e


class WeatherDataset(Dataset):
    def __init__(self, surface_data, upper_air_data, start_idx, end_idx):
        """
        初始化气象数据集 - 按时间序列顺序划分
        
        参数:
            surface_data: 表面数据，形状为 [time, 17, 721, 1440]
            upper_air_data: 高空数据，形状为 [time, 7, 5, 721, 1440]
            start_idx: 开始索引
            end_idx: 结束索引
        """
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.length = end_idx - start_idx - 2  # 减2确保有足够的目标数据
        self.start_idx = start_idx
        
        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # 提取输入数据 (t和t+1时刻)
        input_surface = self.surface_data[actual_idx:actual_idx+2]  # [2, 17, 721, 1440]
        # 添加batch维度并调整为 [17, 2, 721, 1440]
        input_surface = np.transpose(input_surface, (1, 0, 2, 3))  # [17, 2, 721, 1440]
        
        # 提取高空数据 (t和t+1时刻)
        input_upper_air = self.upper_air_data[actual_idx:actual_idx+2]  # [2, 7, 5, 721, 1440]
        # 调整为 [7, 5, 2, 721, 1440]
        input_upper_air = np.transpose(input_upper_air, (1, 2, 0, 3, 4))  # [7, 5, 2, 721, 1440]
        
        # 提取目标数据 (t+2时刻)
        target_surface = self.surface_data[actual_idx+2:actual_idx+3]  # [1, 17, 721, 1440]
        # 调整为 [17, 1, 721, 1440]
        target_surface = np.transpose(target_surface, (1, 0, 2, 3))  # [17, 1, 721, 1440]
        
        target_upper_air = self.upper_air_data[actual_idx+2:actual_idx+3]  # [1, 7, 5, 721, 1440]
        # 调整为 [7, 5, 1, 721, 1440]
        target_upper_air = np.transpose(target_upper_air, (1, 2, 0, 3, 4))  # [7, 5, 1, 721, 1440]
        
        return input_surface, input_upper_air, target_surface, target_upper_air


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    print("Loading data...")
    h5_file = h5.File('/gz-data/ERA5_2023_weekly.h5', 'r')
    input_surface = h5_file['surface'][:]  # (52, 17, 721, 1440)
    input_upper_air = h5_file['upper_air'][:]  # (52, 7, 5, 721, 1440)
    h5_file.close()

    print(f"Surface data shape: {input_surface.shape}")
    print(f"Upper air data shape: {input_upper_air.shape}")

    # 加载标准化参数
    print("Loading normalization parameters...")
    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json'
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    # 移除额外维度并转换为张量
    surface_mean_np = surface_mean_np.squeeze(0).squeeze(1)  # (17, 721, 1440)
    surface_std_np = surface_std_np.squeeze(0).squeeze(1)
    upper_mean_np = upper_mean_np.squeeze(0).squeeze(2)  # (7, 5, 721, 1440)
    upper_std_np = upper_std_np.squeeze(0).squeeze(2)

    surface_mean = torch.from_numpy(surface_mean_np).float().to(device)
    surface_std = torch.from_numpy(surface_std_np).float().to(device)
    upper_mean = torch.from_numpy(upper_mean_np).float().to(device)
    upper_std = torch.from_numpy(upper_std_np).float().to(device)

    print(f"Surface mean shape: {surface_mean.shape}")
    print(f"Surface std shape: {surface_std.shape}")
    print(f"Upper mean shape: {upper_mean.shape}")
    print(f"Upper std shape: {upper_std.shape}")

    # 计算数据集划分点 - 按照6:2:2的时间序列划分
    total_samples = 52
    train_end = 30
    valid_end = 40

    # 创建数据集
    train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=train_end)
    valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=train_end, end_idx=valid_end)
    batch_size = 1  # 小batch size便于调试
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Created data loaders with batch size {batch_size}")

    # 创建模型
    print("Creating model...")
    # 根据观察到的损失量级调整权重：
    # Water loss ~ 1.42e+11, 需要权重 ~ 1e-11 来平衡到 ~1
    # Energy loss ~ 2.54e+12, 需要权重 ~ 1e-12 来平衡到 ~1  
    # Pressure loss ~ 1.76e+6, 需要权重 ~ 1e-6 来平衡到 ~1
    model = CanglongV3(
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_mean=upper_mean,
        upper_std=upper_std,
        lambda_water=1e-11,      # 从0.01降到1e-11
        lambda_energy=1e-12,     # 从0.001降到1e-12
        lambda_pressure=1e-6     # 从0.0001降到1e-6
    )

    # 多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 将模型移动到设备
    model.to(device)

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # 创建保存目录
    save_dir = 'checkpoints_v3'
    os.makedirs(save_dir, exist_ok=True)

    # 训练参数
    num_epochs = 2
    best_valid_loss = float('inf')

    # 物理约束权重（根据损失量级动态调整）
    # 目标：让每个物理约束贡献约1-10的损失量级
    lambda_water = 1e-11     # Water loss ~1e11 -> weight 1e-11 -> contribution ~1
    lambda_energy = 1e-12    # Energy loss ~1e12 -> weight 1e-12 -> contribution ~1  
    lambda_pressure = 1e-6   # Pressure loss ~1e6 -> weight 1e-6 -> contribution ~1

    # 训练循环
    print("Starting training with physical constraints...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        surface_loss = 0.0
        upper_air_loss = 0.0
        water_loss_total = 0.0
        energy_loss_total = 0.0
        pressure_loss_total = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for input_surface, input_upper_air, target_surface, target_upper_air in train_pbar:
            # 将数据移动到设备
            input_surface = input_surface.float().to(device)
            input_upper_air = input_upper_air.float().to(device)
            target_surface = target_surface.float().to(device)
            target_upper_air = target_upper_air.float().to(device)
            
            # 标准化输入数据
            input_surface_norm = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
            input_upper_air_norm = (input_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
            target_surface_norm = (target_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
            target_upper_air_norm = (target_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)
            
            # 计算MSE损失
            loss_surface = criterion(output_surface, target_surface_norm)
            loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
            
            # 计算物理约束损失
            loss_water = calculate_water_balance_loss(
                input_surface_norm, output_surface, 
                surface_mean, surface_std
            )
            loss_energy = calculate_energy_balance_loss(
                output_surface, surface_mean, surface_std
            )
            loss_pressure = calculate_hydrostatic_balance_loss(
                output_upper_air, upper_mean, upper_std
            )
            
            # 总损失
            loss = loss_surface + loss_upper_air + \
                   lambda_water * loss_water + \
                   lambda_energy * loss_energy + \
                   lambda_pressure * loss_pressure
            
            # 反向传播和优化
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累加损失
            batch_loss = loss.item()
            train_loss += batch_loss
            surface_loss += loss_surface.item()
            upper_air_loss += loss_upper_air.item()
            water_loss_total += loss_water.item()
            energy_loss_total += loss_energy.item()
            pressure_loss_total += loss_pressure.item()
            
            # 更新进度条
            train_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "surf": f"{loss_surface.item():.4f}",
                "upper": f"{loss_upper_air.item():.4f}",
                "water": f"{loss_water.item():.2e}",
                "energy": f"{loss_energy.item():.2e}",
                "pressure": f"{loss_pressure.item():.2e}"
            })
        
        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        surface_loss = surface_loss / len(train_loader)
        upper_air_loss = upper_air_loss / len(train_loader)
        water_loss_total = water_loss_total / len(train_loader)
        energy_loss_total = energy_loss_total / len(train_loader)
        pressure_loss_total = pressure_loss_total / len(train_loader)
        
        # 验证阶段
        model.eval()
        valid_loss = 0.0
        valid_surface_loss = 0.0
        valid_upper_air_loss = 0.0
        valid_water_loss = 0.0
        valid_energy_loss = 0.0
        valid_pressure_loss = 0.0
        
        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for input_surface, input_upper_air, target_surface, target_upper_air in valid_pbar:
                # 将数据移动到设备
                input_surface = input_surface.float().to(device)
                input_upper_air = input_upper_air.float().to(device)
                target_surface = target_surface.float().to(device)
                target_upper_air = target_upper_air.float().to(device)
                
                # 标准化输入数据
                input_surface_norm = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
                input_upper_air_norm = (input_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
                target_surface_norm = (target_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
                target_upper_air_norm = (target_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
                
                # 前向传播
                output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)
                
                # 计算MSE损失
                loss_surface = criterion(output_surface, target_surface_norm)
                loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
                
                # 计算物理约束损失
                loss_water = calculate_water_balance_loss(
                    input_surface_norm, output_surface, 
                    surface_mean, surface_std
                )
                loss_energy = calculate_energy_balance_loss(
                    output_surface, surface_mean, surface_std
                )
                loss_pressure = calculate_hydrostatic_balance_loss(
                    output_upper_air, upper_mean, upper_std
                )
                
                # 总损失
                loss = loss_surface + loss_upper_air + \
                       lambda_water * loss_water + \
                       lambda_energy * loss_energy + \
                       lambda_pressure * loss_pressure
                
                # 累加损失
                batch_loss = loss.item()
                valid_loss += batch_loss
                valid_surface_loss += loss_surface.item()
                valid_upper_air_loss += loss_upper_air.item()
                valid_water_loss += loss_water.item()
                valid_energy_loss += loss_energy.item()
                valid_pressure_loss += loss_pressure.item()
                
                # 更新进度条
                valid_pbar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "surf": f"{loss_surface.item():.4f}",
                    "upper": f"{loss_upper_air.item():.4f}",
                    "water": f"{loss_water.item():.2e}",
                    "energy": f"{loss_energy.item():.2e}",
                    "pressure": f"{loss_pressure.item():.2e}"
                })
        
        # 计算平均验证损失
        valid_loss = valid_loss / len(valid_loader)
        valid_surface_loss = valid_surface_loss / len(valid_loader)
        valid_upper_air_loss = valid_upper_air_loss / len(valid_loader)
        valid_water_loss = valid_water_loss / len(valid_loader)
        valid_energy_loss = valid_energy_loss / len(valid_loader)
        valid_pressure_loss = valid_pressure_loss / len(valid_loader)
        
        # 打印损失
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Total: {train_loss:.6f}")
        print(f"         MSE - Surface: {surface_loss:.6f}, Upper Air: {upper_air_loss:.6f}")
        print(f"         Physical Raw - Water: {water_loss_total:.2e}, Energy: {energy_loss_total:.2e}, Pressure: {pressure_loss_total:.2e}")
        print(f"         Physical Weighted - Water: {lambda_water*water_loss_total:.6f}, Energy: {lambda_energy*energy_loss_total:.6f}, Pressure: {lambda_pressure*pressure_loss_total:.6f}")
        print(f"  Valid - Total: {valid_loss:.6f}")
        print(f"         MSE - Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")
        print(f"         Physical Raw - Water: {valid_water_loss:.2e}, Energy: {valid_energy_loss:.2e}, Pressure: {valid_pressure_loss:.2e}")
        print(f"         Physical Weighted - Water: {lambda_water*valid_water_loss:.6f}, Energy: {lambda_energy*valid_energy_loss:.6f}, Pressure: {lambda_pressure*valid_pressure_loss:.6f}")
        
        # 记录最佳模型（但不保存）
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print(f"  → New best validation loss: {best_valid_loss:.6f}")
            # 不保存模型文件
            # save_path = os.path.join(save_dir, 'best_model_v3.pth')
            # torch.save(model.state_dict(), save_path)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()