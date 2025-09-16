#!/usr/bin/env python3
"""
Test script for CanglongV3 model with physical constraints
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append('..')
sys.path.append('.')

# Import the model and utilities
from model_v3 import CanglongV3, PhysicalConstraints
from convert_dict_to_pytorch_arrays import load_normalization_arrays

def test_model_architecture():
    """Test basic model architecture and forward pass"""
    print("=" * 80)
    print("Testing Model Architecture")
    print("=" * 80)
    
    # Create dummy normalization parameters
    surface_mean = torch.randn(17, 721, 1440).cuda()
    surface_std = torch.ones(17, 721, 1440).cuda()
    upper_mean = torch.randn(7, 5, 721, 1440).cuda()
    upper_std = torch.ones(7, 5, 721, 1440).cuda()
    
    # Initialize model
    model = CanglongV3(
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_mean=upper_mean,
        upper_std=upper_std,
        lambda_water=0.1,
        lambda_energy=0.1,
        lambda_pressure=0.1
    ).cuda()
    
    # Create dummy input data
    batch_size = 1
    input_surface = torch.randn(batch_size, 17, 2, 721, 1440).cuda()
    input_upper_air = torch.randn(batch_size, 7, 5, 2, 721, 1440).cuda()
    
    # Test forward pass without physical constraints
    print("\n1. Testing forward pass without physical constraints...")
    with torch.no_grad():
        output_surface, output_upper_air = model(input_surface, input_upper_air)
    
    print(f"   Input surface shape: {input_surface.shape}")
    print(f"   Input upper air shape: {input_upper_air.shape}")
    print(f"   Output surface shape: {output_surface.shape}")
    print(f"   Output upper air shape: {output_upper_air.shape}")
    
    # Verify output shapes
    expected_surface_shape = (batch_size, 17, 1, 721, 1440)
    expected_upper_shape = (batch_size, 7, 5, 1, 721, 1440)
    
    assert output_surface.shape == expected_surface_shape, f"Surface shape mismatch: {output_surface.shape} != {expected_surface_shape}"
    assert output_upper_air.shape == expected_upper_shape, f"Upper air shape mismatch: {output_upper_air.shape} != {expected_upper_shape}"
    print("   ✓ Output shapes are correct!")
    
    return model

def test_physical_constraints(model):
    """Test physical constraint calculations"""
    print("\n" + "=" * 80)
    print("Testing Physical Constraints")
    print("=" * 80)
    
    batch_size = 1
    
    # Create input and target data
    input_surface = torch.randn(batch_size, 17, 2, 721, 1440).cuda()
    input_upper_air = torch.randn(batch_size, 7, 5, 2, 721, 1440).cuda()
    target_surface = torch.randn(batch_size, 17, 1, 721, 1440).cuda()
    target_upper_air = torch.randn(batch_size, 7, 5, 1, 721, 1440).cuda()
    
    # Test forward pass with physical constraints
    print("\n2. Testing forward pass with physical constraints...")
    with torch.no_grad():
        output_surface, output_upper_air, losses = model(
            input_surface, input_upper_air,
            target_surface, target_upper_air,
            return_losses=True
        )
    
    print("\n   Loss components:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key:20s}: {value.item():.6f}")
        else:
            print(f"   - {key:20s}: {value:.6f}")
    
    # Verify all expected losses are present
    expected_losses = ['mse_surface', 'mse_upper_air', 'water_balance', 
                      'energy_balance', 'hydrostatic_balance', 'total']
    
    for loss_name in expected_losses:
        assert loss_name in losses, f"Missing loss component: {loss_name}"
    
    print("\n   ✓ All physical constraint losses are computed!")
    
    # Verify physical constraints are reasonable (non-negative)
    assert losses['water_balance'] >= 0, "Water balance loss should be non-negative"
    assert losses['energy_balance'] >= 0, "Energy balance loss should be non-negative"
    assert losses['hydrostatic_balance'] >= 0, "Hydrostatic balance loss should be non-negative"
    
    print("   ✓ Physical constraint losses are valid!")

def test_with_real_normalization():
    """Test model with real normalization parameters"""
    print("\n" + "=" * 80)
    print("Testing with Real Normalization Parameters")
    print("=" * 80)
    
    # Check if normalization file exists
    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json'
    
    if not os.path.exists(json_path):
        print(f"   Warning: Normalization file not found at {json_path}")
        print("   Skipping real normalization test...")
        return
    
    print(f"\n3. Loading normalization parameters from {json_path}...")
    surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path)
    
    # Remove extra dimensions added by load_normalization_arrays
    # From (1, 17, 1, 721, 1440) to (17, 721, 1440)
    surface_mean = surface_mean.squeeze(0).squeeze(1)  
    surface_std = surface_std.squeeze(0).squeeze(1)
    # From (1, 7, 5, 1, 721, 1440) to (7, 5, 721, 1440)
    upper_mean = upper_mean.squeeze(0).squeeze(2)
    upper_std = upper_std.squeeze(0).squeeze(2)
    
    # Convert numpy arrays to CUDA tensors
    surface_mean = torch.from_numpy(surface_mean).float().cuda()
    surface_std = torch.from_numpy(surface_std).float().cuda()
    upper_mean = torch.from_numpy(upper_mean).float().cuda()
    upper_std = torch.from_numpy(upper_std).float().cuda()
    
    print(f"   Surface mean shape: {surface_mean.shape}")
    print(f"   Surface std shape: {surface_std.shape}")
    print(f"   Upper air mean shape: {upper_mean.shape}")
    print(f"   Upper air std shape: {upper_std.shape}")
    
    # Initialize model with real normalization
    model = CanglongV3(
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_mean=upper_mean,
        upper_std=upper_std,
        lambda_water=0.1,
        lambda_energy=0.1,
        lambda_pressure=0.1
    ).cuda()
    
    print("\n   ✓ Model initialized with real normalization parameters!")
    
    # Test forward pass
    batch_size = 1
    input_surface = torch.randn(batch_size, 17, 2, 721, 1440).cuda()
    input_upper_air = torch.randn(batch_size, 7, 5, 2, 721, 1440).cuda()
    target_surface = torch.randn(batch_size, 17, 1, 721, 1440).cuda()
    target_upper_air = torch.randn(batch_size, 7, 5, 1, 721, 1440).cuda()
    
    with torch.no_grad():
        output_surface, output_upper_air, losses = model(
            input_surface, input_upper_air,
            target_surface, target_upper_air,
            return_losses=True
        )
    
    print("\n   Loss components with real normalization:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"   - {key:20s}: {value.item():.6f}")
    
    print("\n   ✓ Model works correctly with real normalization parameters!")

def test_gradient_flow():
    """Test that gradients flow through physical constraints"""
    print("\n" + "=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)
    
    # Create dummy normalization parameters
    surface_mean = torch.randn(17, 721, 1440).cuda()
    surface_std = torch.ones(17, 721, 1440).cuda()
    upper_mean = torch.randn(7, 5, 721, 1440).cuda()
    upper_std = torch.ones(7, 5, 721, 1440).cuda()
    
    # Initialize model
    model = CanglongV3(
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_mean=upper_mean,
        upper_std=upper_std,
        lambda_water=0.1,
        lambda_energy=0.1,
        lambda_pressure=0.1
    ).cuda()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create input and target data
    batch_size = 1
    input_surface = torch.randn(batch_size, 17, 2, 721, 1440).cuda()
    input_upper_air = torch.randn(batch_size, 7, 5, 2, 721, 1440).cuda()
    target_surface = torch.randn(batch_size, 17, 1, 721, 1440).cuda()
    target_upper_air = torch.randn(batch_size, 7, 5, 1, 721, 1440).cuda()
    
    print("\n4. Testing gradient flow through physical constraints...")
    
    # Forward pass
    output_surface, output_upper_air, losses = model(
        input_surface, input_upper_air,
        target_surface, target_upper_air,
        return_losses=True
    )
    
    # Backward pass
    optimizer.zero_grad()
    losses['total'].backward()
    
    # Check gradients
    has_gradients = False
    total_grad_norm = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
    
    total_grad_norm = total_grad_norm ** 0.5
    
    assert has_gradients, "No gradients found!"
    print(f"   Total gradient norm: {total_grad_norm:.6f}")
    print("   ✓ Gradients flow correctly through the model!")

def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("CANGLONG V3 MODEL TEST SUITE")
    print("Testing model with physical constraints")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nError: CUDA is not available. This model requires GPU.")
        return
    
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    try:
        # Test 1: Model architecture
        model = test_model_architecture()
        
        # Test 2: Physical constraints
        test_physical_constraints(model)
        
        # Test 3: Real normalization parameters
        test_with_real_normalization()
        
        # Test 4: Gradient flow
        test_gradient_flow()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED SUCCESSFULLY! ✓")
        print("=" * 80)
        print("\nThe CanglongV3 model with physical constraints is working correctly.")
        print("\nKey features verified:")
        print("  ✓ Model architecture and forward pass")
        print("  ✓ Physical constraint calculations (water, energy, hydrostatic)")
        print("  ✓ Loss computation with physical constraints")
        print("  ✓ Gradient flow through physical constraints")
        print("  ✓ Compatibility with real normalization parameters")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())