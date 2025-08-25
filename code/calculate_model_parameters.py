#!/usr/bin/env python3
"""
计算CAS-Canglong Swin Transformer模型的参数量
基于 run.py 中 Canglong 类的定义
"""

import torch
import torch.nn as nn
import numpy as np

def calculate_layer_norm_params(dim):
    """计算LayerNorm参数量"""
    return dim * 2  # weight + bias

def calculate_linear_params(in_features, out_features, bias=True):
    """计算Linear层参数量"""
    params = in_features * out_features
    if bias:
        params += out_features
    return params

def calculate_conv2d_params(in_channels, out_channels, kernel_size, bias=True):
    """计算Conv2D参数量"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    params = in_channels * out_channels * kernel_size[0] * kernel_size[1]
    if bias:
        params += out_channels
    return params

def calculate_conv3d_params(in_channels, out_channels, kernel_size, bias=True):
    """计算Conv3D参数量"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    elif len(kernel_size) == 2:
        kernel_size = (1, kernel_size[0], kernel_size[1])
    params = in_channels * out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
    if bias:
        params += out_channels
    return params

def calculate_conv4d_params(in_channels, out_channels, kernel_size, bias=True):
    """计算Conv4D参数量"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
    params = in_channels * out_channels * kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]
    if bias:
        params += out_channels
    return params

def calculate_earth_attention_params(dim, num_heads, window_size, type_of_windows):
    """计算EarthAttention3D参数量"""
    # QKV projection
    qkv_params = calculate_linear_params(dim, dim * 3, bias=True)
    
    # Output projection
    proj_params = calculate_linear_params(dim, dim, bias=False)
    
    # Earth position bias table
    earth_position_bias_params = (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1) * type_of_windows * num_heads
    
    return qkv_params + proj_params + earth_position_bias_params

def calculate_mlp_params(dim, mlp_ratio=4):
    """计算MLP参数量"""
    hidden_dim = int(dim * mlp_ratio)
    fc1_params = calculate_linear_params(dim, hidden_dim, bias=True)
    fc2_params = calculate_linear_params(hidden_dim, dim, bias=True)
    return fc1_params + fc2_params

def calculate_earth_specific_block_params(dim, input_resolution, num_heads, window_size, mlp_ratio=4):
    """计算EarthSpecificBlock参数量"""
    # LayerNorm layers
    norm1_params = calculate_layer_norm_params(dim)
    norm2_params = calculate_layer_norm_params(dim)
    
    # Earth attention
    type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
    attn_params = calculate_earth_attention_params(dim, num_heads, window_size, type_of_windows)
    
    # MLP
    mlp_params = calculate_mlp_params(dim, mlp_ratio)
    
    return norm1_params + norm2_params + attn_params + mlp_params

def calculate_basic_layer_params(dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4):
    """计算BasicLayer参数量"""
    block_params = calculate_earth_specific_block_params(dim, input_resolution, num_heads, window_size, mlp_ratio)
    return block_params * depth

def calculate_downsample_params(in_dim, input_resolution, output_resolution):
    """计算DownSample参数量 (假设为简单的线性投影)"""
    # 假设下采样层使用线性变换将维度从in_dim投影到in_dim*2
    # 实际实现可能不同，这里基于常见做法估算
    return calculate_linear_params(in_dim, in_dim * 2, bias=False)

def calculate_upsample_params(in_dim, out_dim, input_resolution, output_resolution):
    """计算UpSample参数量"""
    # 假设上采样层使用线性变换将维度从in_dim投影到out_dim
    return calculate_linear_params(in_dim, out_dim, bias=False)

def calculate_encoder_decoder_params():
    """计算Encoder和Decoder的参数量(基于代码中的定义)"""
    # Encoder参数
    encoder_params = 0
    channels = [64, 64, 64, 128, 128]
    
    # conv_in: Conv3d(16, 64, kernel_size=(1, 3, 3))
    encoder_params += calculate_conv3d_params(16, channels[0], (1, 3, 3), bias=True)
    
    # 各层参数 (简化计算，实际包含残差块和注意力机制)
    # 这里基于典型VAE编码器结构估算
    encoder_params += calculate_conv3d_params(channels[0], channels[1], 3, bias=True) * 2  # layer1
    encoder_params += calculate_conv3d_params(channels[1], channels[2], 3, bias=True) * 2  # downsample + layer2
    encoder_params += calculate_conv3d_params(channels[2], channels[3], 3, bias=True) * 2  # downsample + layer3
    encoder_params += calculate_conv3d_params(channels[3], channels[3], 3, bias=True) * 2  # mid blocks
    encoder_params += calculate_conv3d_params(channels[3], 96, 3, bias=True)  # conv_out
    
    # Decoder参数 (对称结构)
    decoder_params = 0
    decoder_channels = [192, 128, 128, 64, 64]  # latent_dim=192
    
    decoder_params += calculate_conv3d_params(192, decoder_channels[0], 3, bias=True)  # conv_in
    decoder_params += calculate_conv3d_params(decoder_channels[0], decoder_channels[1], 3, bias=True) * 2  # layer1
    decoder_params += calculate_conv3d_params(decoder_channels[1], decoder_channels[2], 3, bias=True) * 2  # upsample + layer2
    decoder_params += calculate_conv3d_params(decoder_channels[2], decoder_channels[3], 3, bias=True) * 2  # upsample + layer3
    decoder_params += calculate_conv3d_params(decoder_channels[3], decoder_channels[3], 3, bias=True) * 2  # mid blocks
    decoder_params += calculate_conv3d_params(decoder_channels[3], 16, (1, 3, 3), bias=True)  # conv_out
    
    return encoder_params + decoder_params

def calculate_canglong_model_params():
    """计算完整的CAS-Canglong模型参数量"""
    print("=== CAS-Canglong 模型参数计算 ===")
    print()
    
    # 模型配置
    embed_dim = 96
    num_heads = (8, 16, 16, 8)
    window_size = (2, 6, 12)
    mlp_ratio = 4
    
    total_params = 0
    
    # 1. Patch Embedding 层
    print("1. Patch Embedding 层:")
    
    # ImageToPatch2D: (721, 1440) -> (4, 4) patches, 4 -> 96 channels
    patch_embed_2d_params = calculate_conv2d_params(4, embed_dim, (4, 4), bias=True)
    print(f"   ImageToPatch2D: {patch_embed_2d_params:,} 参数")
    
    # ImageToPatch3D: (14, 721, 1440) -> (1, 4, 4) patches, 14 -> 96 channels
    patch_embed_3d_params = calculate_conv3d_params(14, embed_dim, (1, 4, 4), bias=True)
    print(f"   ImageToPatch3D: {patch_embed_3d_params:,} 参数")
    
    # ImageToPatch4D: (7, 4, 721, 1440) -> (4, 2, 4, 4) patches, 7 -> 96 channels
    patch_embed_4d_params = calculate_conv4d_params(7, embed_dim, (4, 2, 4, 4), bias=True)
    print(f"   ImageToPatch4D: {patch_embed_4d_params:,} 参数")
    
    patch_embed_total = patch_embed_2d_params + patch_embed_3d_params + patch_embed_4d_params
    print(f"   小计: {patch_embed_total:,} 参数")
    total_params += patch_embed_total
    
    # 2. Encoder/Decoder (VAE 部分)
    print("\n2. Encoder/Decoder (VAE):")
    encoder_decoder_params = calculate_encoder_decoder_params()
    print(f"   Encoder + Decoder: {encoder_decoder_params:,} 参数")
    total_params += encoder_decoder_params
    
    # 3. 常量卷积层
    print("\n3. 常量处理层:")
    conv_constant_params = calculate_conv2d_params(4, 96, 5, bias=True)  # stride=4, padding=2
    print(f"   Conv2d常量层: {conv_constant_params:,} 参数")
    total_params += conv_constant_params
    
    # 4. Transformer 层
    print("\n4. Swin Transformer 层:")
    
    # Layer1: BasicLayer(dim=96, depth=2, num_heads=8, input_resolution=(4, 181, 360))
    layer1_params = calculate_basic_layer_params(
        dim=embed_dim,
        input_resolution=(4, 181, 360),
        depth=2,
        num_heads=num_heads[0],
        window_size=window_size,
        mlp_ratio=mlp_ratio
    )
    print(f"   Layer1 (深度=2): {layer1_params:,} 参数")
    total_params += layer1_params
    
    # DownSample: 96 -> 192
    downsample_params = calculate_downsample_params(embed_dim, (4, 181, 360), (4, 91, 180))
    print(f"   DownSample: {downsample_params:,} 参数")
    total_params += downsample_params
    
    # Layer2: BasicLayer(dim=192, depth=6, num_heads=16, input_resolution=(4, 91, 180))
    layer2_params = calculate_basic_layer_params(
        dim=embed_dim * 2,
        input_resolution=(4, 91, 180),
        depth=6,
        num_heads=num_heads[1],
        window_size=window_size,
        mlp_ratio=mlp_ratio
    )
    print(f"   Layer2 (深度=6): {layer2_params:,} 参数")
    total_params += layer2_params
    
    # Layer3: BasicLayer(dim=192, depth=6, num_heads=16, input_resolution=(4, 91, 180))
    layer3_params = calculate_basic_layer_params(
        dim=embed_dim * 2,
        input_resolution=(4, 91, 180),
        depth=6,
        num_heads=num_heads[2],
        window_size=window_size,
        mlp_ratio=mlp_ratio
    )
    print(f"   Layer3 (深度=6): {layer3_params:,} 参数")
    total_params += layer3_params
    
    # UpSample: 192 -> 96
    upsample_params = calculate_upsample_params(embed_dim * 2, embed_dim, (4, 91, 180), (4, 181, 360))
    print(f"   UpSample: {upsample_params:,} 参数")
    total_params += upsample_params
    
    # Layer4: BasicLayer(dim=96, depth=2, num_heads=8, input_resolution=(4, 181, 360))
    layer4_params = calculate_basic_layer_params(
        dim=embed_dim,
        input_resolution=(4, 181, 360),
        depth=2,
        num_heads=num_heads[3],
        window_size=window_size,
        mlp_ratio=mlp_ratio
    )
    print(f"   Layer4 (深度=2): {layer4_params:,} 参数")
    total_params += layer4_params
    
    transformer_total = layer1_params + downsample_params + layer2_params + layer3_params + upsample_params + layer4_params
    print(f"   Transformer小计: {transformer_total:,} 参数")
    
    # 5. Recovery 层
    print("\n5. Recovery 层:")
    
    # RecoveryImage2D: (721, 1440), (4, 4), 192 -> 4 channels
    recovery_2d_params = calculate_conv2d_params(2 * embed_dim, 4, (4, 4), bias=True)  # ConvTranspose2d
    print(f"   RecoveryImage2D: {recovery_2d_params:,} 参数")
    
    # RecoveryImage3D: (16, 721, 1440), (1, 4, 4), 192 -> 16 channels  
    recovery_3d_params = calculate_conv3d_params(2 * embed_dim, 16, (1, 4, 4), bias=True)  # ConvTranspose3d
    print(f"   RecoveryImage3D: {recovery_3d_params:,} 参数")
    
    # RecoveryImage4D: (7, 4, 721, 1440), (4, 2, 4, 4), 192 -> 7 channels
    recovery_4d_params = calculate_conv4d_params(2 * embed_dim, 7, (4, 2, 4, 4), bias=True)  # ConvTranspose4d
    print(f"   RecoveryImage4D: {recovery_4d_params:,} 参数")
    
    recovery_total = recovery_2d_params + recovery_3d_params + recovery_4d_params
    print(f"   Recovery小计: {recovery_total:,} 参数")
    total_params += recovery_total
    
    # 总参数统计
    print("\n" + "="*50)
    print("参数统计汇总:")
    print(f"Patch Embedding 层:     {patch_embed_total:>15,} ({patch_embed_total/total_params*100:.1f}%)")
    print(f"Encoder/Decoder:        {encoder_decoder_params:>15,} ({encoder_decoder_params/total_params*100:.1f}%)")
    print(f"常量处理层:             {conv_constant_params:>15,} ({conv_constant_params/total_params*100:.1f}%)")
    print(f"Swin Transformer:       {transformer_total:>15,} ({transformer_total/total_params*100:.1f}%)")
    print(f"Recovery 层:            {recovery_total:>15,} ({recovery_total/total_params*100:.1f}%)")
    print("-" * 50)
    print(f"总参数量:               {total_params:>15,}")
    print(f"模型大小 (FP32):        {total_params * 4 / 1e6:.1f} MB")
    print(f"模型大小 (FP16):        {total_params * 2 / 1e6:.1f} MB")
    
    # 详细分解Transformer部分
    print("\n" + "="*50)
    print("Swin Transformer 详细分解:")
    
    # 计算单个EarthSpecificBlock的参数分布
    sample_dim = 96
    sample_resolution = (4, 181, 360)
    sample_heads = 8
    
    norm_params = calculate_layer_norm_params(sample_dim) * 2
    type_of_windows = (sample_resolution[0] // window_size[0]) * (sample_resolution[1] // window_size[1])
    attn_params = calculate_earth_attention_params(sample_dim, sample_heads, window_size, type_of_windows)
    mlp_params = calculate_mlp_params(sample_dim, mlp_ratio)
    block_total = norm_params + attn_params + mlp_params
    
    print(f"单个Block (dim={sample_dim}):")
    print(f"  LayerNorm:              {norm_params:>10,} ({norm_params/block_total*100:.1f}%)")
    print(f"  EarthAttention3D:       {attn_params:>10,} ({attn_params/block_total*100:.1f}%)")
    print(f"  MLP:                    {mlp_params:>10,} ({mlp_params/block_total*100:.1f}%)")
    print(f"  Block总计:              {block_total:>10,}")
    print(f"总Block数: 2 + 6 + 6 + 2 = 16 个")
    
    return total_params

if __name__ == "__main__":
    total_params = calculate_canglong_model_params()
    
    print("\n" + "="*50)
    print("与典型模型对比:")
    print("ResNet-50:           ~25.6M 参数")
    print("ViT-Base:            ~86.6M 参数") 
    print("Swin-Base:           ~88M 参数")
    print(f"CAS-Canglong:        ~{total_params/1e6:.1f}M 参数")
    
    if total_params > 100e6:
        print(f"\n注意: 该模型为大型模型 (>{total_params/1e6:.0f}M参数)")
    elif total_params > 50e6:
        print(f"\n注意: 该模型为中等规模模型 ({total_params/1e6:.0f}M参数)")
    else:
        print(f"\n注意: 该模型为小型模型 (<{total_params/1e6:.0f}M参数)")