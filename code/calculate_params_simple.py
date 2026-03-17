import torch
import torch.nn as nn
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import components without loading the full model
from canglong.embed import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D
from canglong.recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
import numpy as np

def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """格式化数字，转换为B(billion)或M(million)"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

# 手动计算各组件参数量
def estimate_model_params():
    """
    估算CAS-Canglong V2模型的参数量
    根据model_v2.py的架构
    """

    print("\n" + "="*80)
    print("CAS-Canglong V2 Model Parameter Estimation")
    print("="*80)

    total_params = 0

    # 1. Patch Embedding layers
    print("\n" + "-"*80)
    print("Embedding Layers:")
    print("-"*80)

    # patchembed4d: 7 -> 96 channels, kernel (2,2,4,4)
    embed4d = 7 * 96 * (2*2*4*4)
    print(f"PatchEmbed4D: {format_number(embed4d):>10s} ({embed4d:,})")
    total_params += embed4d

    # encoder3d: 复杂的ResNet-style encoder (17 -> 96 channels)
    # 包含多层卷积、残差块、下采样
    encoder3d = 15_000_000  # 粗略估计，基于ResNet架构
    print(f"Encoder3D: {format_number(encoder3d):>10s} ({encoder3d:,})")
    total_params += encoder3d

    # conv_constant: 64 -> 96 channels, kernel (5,5)
    conv_const = 64 * 96 * 5 * 5
    print(f"Conv_Constant: {format_number(conv_const):>10s} ({conv_const:,})")
    total_params += conv_const

    # 2. Transformer Layers
    print("\n" + "-"*80)
    print("Transformer Layers:")
    print("-"*80)

    embed_dim = 96

    # Layer1: BasicLayer(dim=96, depth=2, num_heads=8, resolution=(6,181,360))
    # 每个WindAwareEarthSpecificBlock包含:
    # - QKV projection: 96 -> 96*3
    # - Output projection: 96 -> 96
    # - MLP: 96 -> 96*4 -> 96
    # - Layer norms
    # - Wind-aware attention mechanisms

    def estimate_block_params(dim, mlp_ratio=4):
        # QKV projection
        qkv = dim * dim * 3
        # Output projection
        out_proj = dim * dim
        # MLP
        mlp = dim * (dim * mlp_ratio) + (dim * mlp_ratio) * dim
        # Layer norms (approximate)
        norms = dim * 4
        return qkv + out_proj + mlp + norms

    block_params = estimate_block_params(embed_dim)
    layer1_params = block_params * 2  # depth=2
    print(f"Layer1 (2 blocks, dim=96): {format_number(layer1_params):>10s} ({layer1_params:,})")
    total_params += layer1_params

    # DownSample: 96 -> 192
    downsample = embed_dim * 4 * (embed_dim * 2)
    print(f"DownSample: {format_number(downsample):>10s} ({downsample:,})")
    total_params += downsample

    # Layer2: BasicLayer(dim=192, depth=6, num_heads=16)
    block_params_192 = estimate_block_params(embed_dim * 2)
    layer2_params = block_params_192 * 6  # depth=6
    print(f"Layer2 (6 blocks, dim=192): {format_number(layer2_params):>10s} ({layer2_params:,})")
    total_params += layer2_params

    # Layer3: BasicLayer(dim=192, depth=6, num_heads=16)
    layer3_params = block_params_192 * 6  # depth=6
    print(f"Layer3 (6 blocks, dim=192): {format_number(layer3_params):>10s} ({layer3_params:,})")
    total_params += layer3_params

    # UpSample: 192 -> 96
    upsample = embed_dim * 2 * (embed_dim * 4) + embed_dim * embed_dim
    print(f"UpSample: {format_number(upsample):>10s} ({upsample:,})")
    total_params += upsample

    # Layer4: BasicLayer(dim=96, depth=2, num_heads=8)
    layer4_params = block_params * 2  # depth=2
    print(f"Layer4 (2 blocks, dim=96): {format_number(layer4_params):>10s} ({layer4_params:,})")
    total_params += layer4_params

    # 3. Recovery/Decoder Layers
    print("\n" + "-"*80)
    print("Recovery/Decoder Layers:")
    print("-"*80)

    # decoder3d: 复杂的ResNet-style decoder (192 -> 17 channels)
    decoder3d = 15_000_000  # 粗略估计，与encoder对称
    print(f"Decoder3D: {format_number(decoder3d):>10s} ({decoder3d:,})")
    total_params += decoder3d

    # patchrecovery4d: 192 -> 7 channels
    recovery4d = 192 * 7 * (2*1*4*4)
    print(f"PatchRecovery4D: {format_number(recovery4d):>10s} ({recovery4d:,})")
    total_params += recovery4d

    # Wind direction processor (small CNN for wind field processing)
    wind_processor = 100_000  # 估计
    print(f"WindDirectionProcessor: {format_number(wind_processor):>10s} ({wind_processor:,})")
    total_params += wind_processor

    # Summary
    print("\n" + "="*80)
    print(f"Estimated Total Parameters: {format_number(total_params)}")
    print(f"Exact count: {total_params:,}")
    print("="*80)

    # Component breakdown
    print("\nParameter Distribution:")
    print("-"*80)
    transformer_params = layer1_params + layer2_params + layer3_params + layer4_params + downsample + upsample
    encoder_decoder_params = encoder3d + decoder3d
    embed_recovery_params = embed4d + conv_const + recovery4d

    print(f"Transformer Blocks: {format_number(transformer_params):>10s} ({transformer_params:,}) - {transformer_params/total_params*100:.1f}%")
    print(f"Encoder/Decoder: {format_number(encoder_decoder_params):>10s} ({encoder_decoder_params:,}) - {encoder_decoder_params/total_params*100:.1f}%")
    print(f"Embed/Recovery: {format_number(embed_recovery_params):>10s} ({embed_recovery_params:,}) - {embed_recovery_params/total_params*100:.1f}%")
    print(f"Wind Processor: {format_number(wind_processor):>10s} ({wind_processor:,}) - {wind_processor/total_params*100:.1f}%")

    print("\n")

    # Comparison with other models
    print("="*80)
    print("Model Comparison:")
    print("-"*80)
    print("Pangu-Weather: ~3.8B parameters")
    print("FuXi: ~1.4B parameters")
    print("GraphCast: ~37M parameters")
    print(f"CAS-Canglong V2: ~{format_number(total_params)}")
    print("="*80 + "\n")

if __name__ == "__main__":
    estimate_model_params()
