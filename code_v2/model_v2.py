import torch
from torch import nn
import numpy as np
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F
import sys
sys.path.append('..')
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

input_constant = torch.load('../constant_masks/Earth.pt', weights_only=False).cuda()


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


class CanglongV2(nn.Module):
    """
    CAS Canglong PyTorch impl of: `CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global sea surface temperature prediction`
    Version 2 with wind-aware dynamic window shifting
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        
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


    def forward(self, surface, upper_air):        
        
        # 计算风向ID
        wind_direction_id = self.wind_direction_processor(surface, upper_air)
        
        constant = self.conv_constant(self.input_constant)
        surface = self.encoder3d(surface)

        upper_air = self.patchembed4d(upper_air)

        
        x = torch.concat([upper_air.squeeze(3), 
                          surface, 
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
        
        return output_surface, output_upper_air


        # 简化输出处理来验证模型架构
        return output_surface, output_upper_air  # 只取前2层surface
    

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import h5py as h5

class WeatherDataset(Dataset):
    def __init__(self, surface_data, upper_air_data, start_idx, end_idx):
        """
        初始化气象数据集 - 按时间序列顺序划分
        
        参数:
            surface_data: 表面数据，形状为 [17, 100, 721, 1440]
            upper_air_data: 高空数据，形状为 [7, 5, 100, 721, 1440]
            start_idx: 开始索引
            end_idx: 结束索引
        """
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.length = end_idx - start_idx - 2  # 减2确保有足够的目标数据
        
        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 提取输入数据 (t和t+1时刻)
        input_surface = self.surface_data[idx:idx+2]  # [1, 17, 2, 721, 1440]
        
        # 提取高空数据 (t和t+1时刻)
        input_upper_air = self.upper_air_data[idx:idx+2]  # [1, 7, 5, 2, 721, 1440]
        
        # 提取目标数据 (t+2时刻)
        target_surface = self.surface_data[idx+2]  # [1, 16, 721, 1440]
        target_upper_air = self.upper_air_data[idx+2]  # [1, 7, 4, 721, 1440]
        
        return input_surface, input_upper_air, target_surface, target_upper_air

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
print("Loading data...")
input_surface, input_upper_air = h5.File('/gz-data/ERA5_2023_weekly.h5')['surface'], h5.File('/gz-data/ERA5_2023_weekly.h5')['upper_air']
print(f"Surface data shape: {input_surface.shape}") #(52, 17, 721, 1440)
print(f"Upper air data shape: {input_upper_air.shape}") #(52, 7, 5, 721, 1440)

# 计算数据集划分点 - 按照6:2:2的时间序列划分
total_samples = 52#input_surface.shape[0]  # 假设为100
train_end = 30#int(total_samples * 0.6)  # 60
valid_end = 40#int(total_samples * 0.8)  # 80

# 创建数据集
train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=train_end)
valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=train_end, end_idx=valid_end)
batch_size = 1  # 小batch size便于调试
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # 不打乱时间顺序
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
print(f"Created data loaders with batch size {batch_size}")
import sys
sys.path.append('code_v2')
from convert_dict_to_pytorch_arrays import load_normalization_arrays


# 加载标准化参数
# 注释掉实际加载，使用模拟数据
from convert_dict_to_pytorch_arrays import load_normalization_arrays
json = '/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json'
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json)



# 创建模型
model = CanglongV2()
#model = torch.load('../model/model_v1_100.pth')
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
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# 训练参数
num_epochs = 2
best_valid_loss = float('inf')

# 训练循环
print("Starting training...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    surface_loss = 0.0
    upper_air_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for input_surface, input_upper_air, target_surface, target_upper_air in train_pbar:
        # 将数据移动到设备
        input_surface = ((input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std).to(device)
        input_upper_air = ((input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std).to(device)
        target_surface = ((target_surface.unsqueeze(2) - surface_mean) / surface_mean).to(device)
        target_upper_air = ((target_upper_air.unsqueeze(3) - upper_mean) / upper_std).to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        output_surface, output_upper_air = model(input_surface, input_upper_air)
        
        # 计算损失
        loss_surface = criterion(output_surface, target_surface)
        loss_upper_air = criterion(output_upper_air, target_upper_air)
        loss = loss_surface + loss_upper_air
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累加损失
        batch_loss = loss.item()
        train_loss += batch_loss
        surface_loss += loss_surface.item()
        upper_air_loss += loss_upper_air.item()
        
        # 更新进度条
        train_pbar.set_postfix({
            "loss": f"{batch_loss:.6f}",
            "surface": f"{loss_surface.item():.6f}",
            "upper_air": f"{loss_upper_air.item():.6f}"
        })
    
    # 计算平均训练损失
    train_loss = train_loss / len(train_loader)
    surface_loss = surface_loss / len(train_loader)
    upper_air_loss = upper_air_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    valid_loss = 0.0
    valid_surface_loss = 0.0
    valid_upper_air_loss = 0.0
    
    with torch.no_grad():
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        for input_surface, input_upper_air, target_surface, target_upper_air in valid_pbar:
            # 将数据移动到设备
            input_surface = ((input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std).to(device)
            input_upper_air = ((input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std).to(device)
            target_surface = ((target_surface.unsqueeze(2) - surface_mean) / surface_mean).to(device)
            target_upper_air = ((target_upper_air.unsqueeze(3) - upper_mean) / upper_std).to(device)
            
            # 前向传播
            output_surface, output_upper_air = model(input_surface, input_upper_air)
            
            # 计算损失
            loss_surface = criterion(output_surface, target_surface)
            loss_upper_air = criterion(output_upper_air, target_upper_air)
            loss = loss_surface + loss_upper_air
            
            # 累加损失
            batch_loss = loss.item()
            valid_loss += batch_loss
            valid_surface_loss += loss_surface.item()
            valid_upper_air_loss += loss_upper_air.item()
            
            # 更新进度条
            valid_pbar.set_postfix({
                "loss": f"{batch_loss:.6f}",
                "surface": f"{loss_surface.item():.6f}",
                "upper_air": f"{loss_upper_air.item():.6f}"
            })
    
    # 计算平均验证损失
    valid_loss = valid_loss / len(valid_loader)
    valid_surface_loss = valid_surface_loss / len(valid_loader)
    valid_upper_air_loss = valid_upper_air_loss / len(valid_loader)
    
    # 打印损失
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train - Total: {train_loss:.6f}, Surface: {surface_loss:.6f}, Upper Air: {upper_air_loss:.6f}")
    print(f"  Valid - Total: {valid_loss:.6f}, Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")