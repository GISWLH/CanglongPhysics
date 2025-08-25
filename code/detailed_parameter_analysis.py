#!/usr/bin/env python3
"""
详细分析CAS-Canglong模型参数分布和计算验证
包括逐层参数分解和关键组件分析
"""

import torch
import torch.nn as nn
import numpy as np

def detailed_attention_analysis():
    """详细分析EarthAttention3D的参数构成"""
    print("=== EarthAttention3D 详细参数分析 ===")
    
    # 参数配置
    dim = 96
    num_heads = 8
    window_size = (2, 6, 12)
    input_resolution = (4, 181, 360)
    
    print(f"配置: dim={dim}, num_heads={num_heads}, window_size={window_size}")
    print(f"输入分辨率: {input_resolution}")
    
    # 计算type_of_windows
    type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])
    print(f"窗口类型数: ({input_resolution[0]}//{window_size[0]}) * ({input_resolution[1]}//{window_size[1]}) = {type_of_windows}")
    
    # QKV线性层参数
    qkv_params = dim * dim * 3 + dim * 3  # weight + bias
    print(f"\n1. QKV线性层: {dim} -> {dim * 3}")
    print(f"   权重: {dim} * {dim * 3} = {dim * dim * 3:,}")
    print(f"   偏置: {dim * 3} = {dim * 3}")
    print(f"   小计: {qkv_params:,}")
    
    # 输出投影层参数 (无偏置)
    proj_params = dim * dim
    print(f"\n2. 输出投影层: {dim} -> {dim} (无偏置)")
    print(f"   权重: {dim} * {dim} = {proj_params:,}")
    
    # Earth position bias table
    bias_table_size = (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1) * type_of_windows * num_heads
    print(f"\n3. Earth位置偏置表:")
    print(f"   形状: ({window_size[0]}^2) * ({window_size[1]}^2) * ({window_size[2]}*2-1) * {type_of_windows} * {num_heads}")
    print(f"   计算: {window_size[0]**2} * {window_size[1]**2} * {window_size[2]*2-1} * {type_of_windows} * {num_heads}")
    print(f"   参数量: {bias_table_size:,}")
    
    total_attn = qkv_params + proj_params + bias_table_size
    print(f"\n总EarthAttention3D参数: {total_attn:,}")
    print(f"分布: QKV({qkv_params/total_attn*100:.1f}%) + Proj({proj_params/total_attn*100:.1f}%) + PosEmb({bias_table_size/total_attn*100:.1f}%)")
    
    return total_attn

def analyze_transformer_layers():
    """分析各个Transformer层的参数分布"""
    print("\n=== Transformer层参数详细分析 ===")
    
    embed_dim = 96
    window_size = (2, 6, 12)
    mlp_ratio = 4
    
    layers_config = [
        {"name": "Layer1", "dim": embed_dim, "resolution": (4, 181, 360), "depth": 2, "heads": 8},
        {"name": "Layer2", "dim": embed_dim*2, "resolution": (4, 91, 180), "depth": 6, "heads": 16},
        {"name": "Layer3", "dim": embed_dim*2, "resolution": (4, 91, 180), "depth": 6, "heads": 16},
        {"name": "Layer4", "dim": embed_dim, "resolution": (4, 181, 360), "depth": 2, "heads": 8},
    ]
    
    total_transformer_params = 0
    
    for layer in layers_config:
        print(f"\n{layer['name']} 分析:")
        print(f"  维度: {layer['dim']}, 分辨率: {layer['resolution']}, 深度: {layer['depth']}, 头数: {layer['heads']}")
        
        # 计算单个Block的参数
        dim = layer['dim']
        
        # LayerNorm参数 (两个)
        norm_params = dim * 2 * 2  # 每个LayerNorm: weight + bias
        
        # EarthAttention参数
        type_of_windows = (layer['resolution'][0] // window_size[0]) * (layer['resolution'][1] // window_size[1])
        qkv_params = dim * dim * 3 + dim * 3
        proj_params = dim * dim
        bias_table_size = (window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1) * type_of_windows * layer['heads']
        attn_params = qkv_params + proj_params + bias_table_size
        
        # MLP参数
        hidden_dim = int(dim * mlp_ratio)
        mlp_params = dim * hidden_dim + hidden_dim + hidden_dim * dim + dim  # fc1 + fc2 (含偏置)
        
        # 单个Block总参数
        block_params = norm_params + attn_params + mlp_params
        layer_params = block_params * layer['depth']
        
        print(f"  单个Block参数:")
        print(f"    LayerNorm: {norm_params:,}")
        print(f"    EarthAttention: {attn_params:,}")
        print(f"      - QKV: {qkv_params:,}")
        print(f"      - 输出投影: {proj_params:,}")
        print(f"      - 位置偏置: {bias_table_size:,}")
        print(f"    MLP: {mlp_params:,}")
        print(f"  Block总计: {block_params:,}")
        print(f"  {layer['name']}总计 (×{layer['depth']}): {layer_params:,}")
        
        total_transformer_params += layer_params
    
    # 上下采样层
    downsample_params = embed_dim * embed_dim * 2  # 96 -> 192
    upsample_params = embed_dim * 2 * embed_dim    # 192 -> 96
    
    print(f"\n采样层参数:")
    print(f"  DownSample: {downsample_params:,}")
    print(f"  UpSample: {upsample_params:,}")
    
    total_transformer_params += downsample_params + upsample_params
    
    print(f"\nTransformer总参数: {total_transformer_params:,}")
    return total_transformer_params

def analyze_encoder_decoder():
    """分析编码器-解码器的参数"""
    print("\n=== Encoder-Decoder 详细分析 ===")
    
    encoder_params = 0
    decoder_params = 0
    
    print("Encoder参数估算:")
    # 根据代码中的结构估算
    encoder_layers = [
        ("conv_in", 16, 64, (1, 3, 3)),  # Conv3d
        ("layer1_res1", 64, 64, (3, 3, 3)),
        ("layer1_res2", 64, 64, (3, 3, 3)),
        ("downsample1", 64, 64, (2, 2, 2)),  # stride=2
        ("layer2_res1", 64, 64, (3, 3, 3)),
        ("layer2_res2", 64, 64, (3, 3, 3)),
        ("downsample2", 64, 128, (2, 2, 2)),  # stride=2
        ("layer3_res1", 128, 128, (3, 3, 3)),
        ("layer3_res2", 128, 128, (3, 3, 3)),
        ("mid_block1", 128, 128, (3, 3, 3)),
        ("mid_block2", 128, 128, (3, 3, 3)),
        ("conv_out", 128, 96, (3, 3, 3))  # 输出到latent_dim=96
    ]
    
    for name, in_ch, out_ch, kernel in encoder_layers:
        if isinstance(kernel, tuple) and len(kernel) == 3:
            params = in_ch * out_ch * kernel[0] * kernel[1] * kernel[2] + out_ch
        else:
            params = in_ch * out_ch * 3 * 3 * 3 + out_ch  # 默认3x3x3
        encoder_params += params
        print(f"  {name}: {in_ch}->{out_ch}, {params:,} 参数")
    
    print(f"Encoder总计: {encoder_params:,}")
    
    print("\nDecoder参数估算:")
    # 对称的解码器结构
    decoder_layers = [
        ("conv_in", 192, 128, (3, 3, 3)),  # latent_dim=192 -> 128
        ("layer1_res1", 128, 128, (3, 3, 3)),
        ("layer1_res2", 128, 128, (3, 3, 3)),
        ("upsample1", 128, 128, (2, 2, 2)),  # stride=2
        ("layer2_res1", 128, 64, (3, 3, 3)),
        ("layer2_res2", 64, 64, (3, 3, 3)),
        ("upsample2", 64, 64, (2, 2, 2)),  # stride=2
        ("layer3_res1", 64, 64, (3, 3, 3)),
        ("layer3_res2", 64, 64, (3, 3, 3)),
        ("mid_block1", 64, 64, (3, 3, 3)),
        ("mid_block2", 64, 64, (3, 3, 3)),
        ("conv_out", 64, 16, (1, 3, 3))  # ConvTranspose3d输出16通道
    ]
    
    for name, in_ch, out_ch, kernel in decoder_layers:
        if isinstance(kernel, tuple):
            if len(kernel) == 3:
                params = in_ch * out_ch * kernel[0] * kernel[1] * kernel[2] + out_ch
            else:
                params = in_ch * out_ch * 3 * 3 * 3 + out_ch
        else:
            params = in_ch * out_ch * 3 * 3 * 3 + out_ch
        decoder_params += params
        print(f"  {name}: {in_ch}->{out_ch}, {params:,} 参数")
    
    print(f"Decoder总计: {decoder_params:,}")
    print(f"Encoder+Decoder总计: {encoder_params + decoder_params:,}")
    
    return encoder_params + decoder_params

def memory_analysis(total_params):
    """分析模型的内存占用"""
    print(f"\n=== 内存占用分析 ===")
    print(f"参数总数: {total_params:,}")
    
    # 不同精度下的模型大小
    fp32_size = total_params * 4 / 1e6  # MB
    fp16_size = total_params * 2 / 1e6  # MB
    int8_size = total_params * 1 / 1e6  # MB
    
    print(f"模型参数存储:")
    print(f"  FP32: {fp32_size:.1f} MB")
    print(f"  FP16: {fp16_size:.1f} MB") 
    print(f"  INT8: {int8_size:.1f} MB")
    
    # 推理时的内存估算 (包括激活值)
    batch_size = 1
    sequence_length = 4 * 181 * 360  # 展平后的序列长度
    hidden_dim = 96
    
    # 主要激活值内存
    input_activation = batch_size * sequence_length * hidden_dim * 4 / 1e6  # FP32
    attention_activation = batch_size * 16 * sequence_length * sequence_length * 4 / 1e6  # 注意力矩阵
    
    print(f"\n推理时激活值内存估算 (batch_size={batch_size}):")
    print(f"  输入激活: {input_activation:.1f} MB")
    print(f"  注意力矩阵: {attention_activation:.1f} MB (可能会很大)")
    
    total_inference_memory = fp32_size + input_activation
    print(f"  推理总内存 (不含注意力): ~{total_inference_memory:.1f} MB")
    
def compare_with_baselines():
    """与基线模型对比"""
    print(f"\n=== 与基线模型对比 ===")
    
    models = {
        "CAS-Canglong": 37.6,
        "ResNet-50": 25.6,
        "ResNet-101": 44.5,
        "ViT-Base/16": 86.6,
        "ViT-Large/16": 307.4,
        "Swin-Tiny": 29.0,
        "Swin-Small": 50.0,
        "Swin-Base": 88.0,
        "PanguWeather": 200.0,  # 估算
        "FourCastNet": 100.0,   # 估算
    }
    
    print("模型规模对比 (参数量, M):")
    for model, params in sorted(models.items(), key=lambda x: x[1]):
        if model == "CAS-Canglong":
            print(f"  {model:<15}: {params:>6.1f}M ← 当前模型")
        else:
            print(f"  {model:<15}: {params:>6.1f}M")
    
    canglong_params = models["CAS-Canglong"]
    print(f"\nCAS-Canglong相对规模:")
    print(f"  vs ViT-Base: {canglong_params/models['ViT-Base/16']:.2f}x 小")
    print(f"  vs Swin-Base: {canglong_params/models['Swin-Base']:.2f}x 小")
    print(f"  vs ResNet-50: {canglong_params/models['ResNet-50']:.2f}x 大")

if __name__ == "__main__":
    print("CAS-Canglong 模型详细参数分析")
    print("=" * 60)
    
    # 详细注意力机制分析
    attn_params = detailed_attention_analysis()
    
    # Transformer层分析
    transformer_params = analyze_transformer_layers()
    
    # 编码解码器分析
    encoder_decoder_params = analyze_encoder_decoder()
    
    # 其他组件参数 (从之前的计算)
    patch_embed_params = 113952
    conv_constant_params = 9696
    recovery_params = 233499
    
    total_params = (patch_embed_params + encoder_decoder_params + 
                   conv_constant_params + transformer_params + recovery_params)
    
    print(f"\n{'='*60}")
    print("最终参数统计:")
    print(f"  Patch Embedding:    {patch_embed_params:>10,} ({patch_embed_params/total_params*100:>5.1f}%)")
    print(f"  Encoder/Decoder:     {encoder_decoder_params:>10,} ({encoder_decoder_params/total_params*100:>5.1f}%)")
    print(f"  常量处理:            {conv_constant_params:>10,} ({conv_constant_params/total_params*100:>5.1f}%)")
    print(f"  Swin Transformer:    {transformer_params:>10,} ({transformer_params/total_params*100:>5.1f}%)")
    print(f"  Recovery层:          {recovery_params:>10,} ({recovery_params/total_params*100:>5.1f}%)")
    print(f"  {'总计:':<18} {total_params:>10,}")
    
    # 内存分析
    memory_analysis(total_params)
    
    # 基线对比
    compare_with_baselines()