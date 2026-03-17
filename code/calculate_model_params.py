import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add code_v2 to path
code_v2_path = os.path.join(project_root, 'code_v2')
sys.path.insert(0, code_v2_path)

from model_v2 import CanglongV2

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

def analyze_model_structure(model):
    """详细分析模型各部分的参数量"""
    print("\n" + "="*80)
    print("CAS-Canglong Model Parameter Analysis")
    print("="*80)

    # 总参数量
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal Parameters: {format_number(total_params)} ({total_params:,})")
    print(f"Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")

    # 详细分析各模块
    print("\n" + "-"*80)
    print("Parameter breakdown by module:")
    print("-"*80)

    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
        print(f"{name:30s}: {format_number(params):>10s} ({params:>12,d})")

    # 更详细的Transformer层分析
    print("\n" + "-"*80)
    print("Detailed Transformer Layer Analysis:")
    print("-"*80)

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            layer_params = sum(p.numel() for p in layer.parameters())
            print(f"\n{layer_name}:")
            print(f"  Total: {format_number(layer_params)} ({layer_params:,})")

            # 分析每个block
            if hasattr(layer, 'blocks'):
                for i, block in enumerate(layer.blocks):
                    block_params = sum(p.numel() for p in block.parameters())
                    print(f"    Block {i}: {format_number(block_params)} ({block_params:,})")

    # Encoder/Decoder分析
    print("\n" + "-"*80)
    print("Encoder/Decoder Analysis:")
    print("-"*80)

    if hasattr(model, 'encoder3d'):
        encoder_params = sum(p.numel() for p in model.encoder3d.parameters())
        print(f"Encoder3D: {format_number(encoder_params)} ({encoder_params:,})")

    if hasattr(model, 'decoder3d'):
        decoder_params = sum(p.numel() for p in model.decoder3d.parameters())
        print(f"Decoder3D: {format_number(decoder_params)} ({decoder_params:,})")

    # 嵌入层分析
    print("\n" + "-"*80)
    print("Embedding Layer Analysis:")
    print("-"*80)

    for embed_name in ['patchembed2d', 'patchembed3d', 'patchembed4d']:
        if hasattr(model, embed_name):
            embed = getattr(model, embed_name)
            embed_params = sum(p.numel() for p in embed.parameters())
            print(f"{embed_name}: {format_number(embed_params)} ({embed_params:,})")

    # Recovery层分析
    print("\n" + "-"*80)
    print("Recovery Layer Analysis:")
    print("-"*80)

    for recovery_name in ['patchrecovery2d', 'patchrecovery3d', 'patchrecovery4d']:
        if hasattr(model, recovery_name):
            recovery = getattr(model, recovery_name)
            recovery_params = sum(p.numel() for p in recovery.parameters())
            print(f"{recovery_name}: {format_number(recovery_params)} ({recovery_params:,})")

    print("\n" + "="*80)
    print(f"Summary: CAS-Canglong V2 has {format_number(total_params)} parameters")
    print("="*80 + "\n")

if __name__ == "__main__":
    # 创建模型实例
    print("Initializing CAS-Canglong V2 model...")
    model = CanglongV2()

    # 分析模型参数
    analyze_model_structure(model)

    # 测试模型输入输出
    print("\nTesting model with sample input...")
    input_surface = torch.randn(1, 17, 2, 721, 1440)
    input_upper_air = torch.randn(1, 7, 5, 2, 721, 1440)

    print(f"Input surface shape: {input_surface.shape}")
    print(f"Input upper air shape: {input_upper_air.shape}")

    # 如果有GPU，测试GPU内存占用
    if torch.cuda.is_available():
        print("\nTesting GPU memory usage...")
        model = model.cuda()
        input_surface = input_surface.cuda()
        input_upper_air = input_upper_air.cuda()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            output_surface, output_upper_air = model(input_surface, input_upper_air)

        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"Peak GPU memory: {memory_allocated:.2f} GB")
        print(f"Output surface shape: {output_surface.shape}")
        print(f"Output upper air shape: {output_upper_air.shape}")
    else:
        print("\nNo GPU available, skipping GPU memory test")
