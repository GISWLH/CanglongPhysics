#!/usr/bin/env python
"""
Test script for temperature tendency equation constraint
Tests the physical consistency and closure rates with real ERA5 data
"""

import torch
import numpy as np
import h5py as h5
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

def test_temperature_tendency():
    """Test temperature tendency constraint with real data"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Load data
    print("Loading ERA5 data...")
    with h5.File('/gz-data/ERA5_2023_weekly_new.h5', 'r') as f:
        surface_data = f['surface'][:3]  # Load 3 weeks
        upper_air_data = f['upper_air'][:3]  # Load 3 weeks

    print(f"Surface shape: {surface_data.shape}")
    print(f"Upper air shape: {upper_air_data.shape}")

    # Load normalization parameters
    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    # Convert to tensors
    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    # Prepare input data (week 0-1 as input, week 2 as output)
    input_surface = torch.from_numpy(surface_data[0:2]).float().to(device)
    input_surface = input_surface.transpose(0, 1).unsqueeze(0)  # (1, 26, 2, 721, 1440)

    input_upper_air = torch.from_numpy(upper_air_data[0:2]).float().to(device)
    input_upper_air = input_upper_air.permute(1, 2, 0, 3, 4).unsqueeze(0)  # (1, 10, 5, 2, 721, 1440)

    output_surface = torch.from_numpy(surface_data[2:3]).float().to(device)
    output_surface = output_surface.transpose(0, 1).unsqueeze(0)  # (1, 26, 1, 721, 1440)

    output_upper_air = torch.from_numpy(upper_air_data[2:3]).float().to(device)
    output_upper_air = output_upper_air.permute(1, 2, 0, 3, 4).unsqueeze(0)  # (1, 10, 5, 1, 721, 1440)

    # Normalize
    input_surface_norm = (input_surface - surface_mean) / surface_std
    input_upper_air_norm = (input_upper_air - upper_mean) / upper_std
    output_surface_norm = (output_surface - surface_mean) / surface_std
    output_upper_air_norm = (output_upper_air - upper_mean) / upper_std

    # Import and test the constraint function
    import sys
    sys.path.append('/home/CanglongPhysics')
    from train_v3 import calculate_temperature_tendency_loss

    print("\nTesting temperature tendency constraint...")
    loss = calculate_temperature_tendency_loss(
        input_upper_air_norm, output_upper_air_norm,
        input_surface_norm, output_surface_norm,
        upper_mean, upper_std, surface_mean, surface_std
    )

    print(f"Temperature tendency loss: {loss.item():.6e}")

    # Analyze components
    print("\nAnalyzing physical components...")

    # Physical constants
    R_d = 287.0  # J/(kg·K)
    c_p = 1004.0  # J/(kg·K)
    g = 9.8  # m/s²

    # Denormalize
    input_upper_physical = input_upper_air_norm * upper_std + upper_mean
    output_upper_physical = output_upper_air_norm * upper_std + upper_mean
    input_surface_physical = input_surface_norm * surface_std + surface_mean
    output_surface_physical = output_surface_norm * surface_std + surface_mean

    # Extract temperature fields
    idx_t = 2  # Temperature index
    t_t1 = input_upper_physical[0, idx_t, :, -1, :, :]  # Week 1
    t_t2 = output_upper_physical[0, idx_t, :, 0, :, :]   # Week 2

    # Time derivative
    dt = 7 * 24 * 3600  # seconds
    dT_dt = (t_t2 - t_t1) / dt  # K/s

    # Statistics by pressure level
    pressure_levels = [200, 300, 500, 700, 850]
    print("\nTemperature tendency by pressure level:")
    for i, p in enumerate(pressure_levels):
        level_tendency = dT_dt[i]
        mean_tendency = level_tendency.mean().item()
        std_tendency = level_tendency.std().item()
        max_tendency = level_tendency.abs().max().item()

        print(f"  {p} hPa:")
        print(f"    Mean dT/dt: {mean_tendency:.2e} K/s")
        print(f"    Std dT/dt:  {std_tendency:.2e} K/s")
        print(f"    Max |dT/dt|: {max_tendency:.2e} K/s")

        # Typical values should be < 1e-4 K/s
        if max_tendency > 1e-4:
            print(f"    WARNING: Large temperature tendency detected!")

    # Check horizontal advection
    idx_u, idx_v = 3, 4
    u = output_upper_physical[0, idx_u, :, 0, :, :]
    v = output_upper_physical[0, idx_v, :, 0, :, :]

    print("\nWind field statistics:")
    for i, p in enumerate(pressure_levels):
        u_mean = u[i].mean().item()
        v_mean = v[i].mean().item()
        wind_speed = torch.sqrt(u[i]**2 + v[i]**2).mean().item()

        print(f"  {p} hPa:")
        print(f"    Mean U: {u_mean:.1f} m/s")
        print(f"    Mean V: {v_mean:.1f} m/s")
        print(f"    Mean wind speed: {wind_speed:.1f} m/s")

    # Check vertical motion
    idx_w = 5
    w = output_upper_physical[0, idx_w, :, 0, :, :]  # Pa/s

    print("\nVertical velocity statistics:")
    for i, p in enumerate(pressure_levels):
        w_mean = w[i].mean().item()
        w_std = w[i].std().item()
        w_max_up = w[i].min().item()  # Negative is upward
        w_max_down = w[i].max().item()

        print(f"  {p} hPa:")
        print(f"    Mean ω: {w_mean:.3f} Pa/s")
        print(f"    Std ω:  {w_std:.3f} Pa/s")
        print(f"    Max upward: {w_max_up:.3f} Pa/s")
        print(f"    Max downward: {w_max_down:.3f} Pa/s")

    # Check diabatic heating components
    idx_lsrr, idx_crr = 4, 5
    lsrr = output_surface_physical[0, idx_lsrr, 0, :, :]
    crr = output_surface_physical[0, idx_crr, 0, :, :]
    total_precip = (lsrr + crr) * 3600 * 24 * 7  # Convert to mm/week

    print(f"\nPrecipitation statistics:")
    print(f"  Mean total precipitation: {total_precip.mean().item():.2f} mm/week")
    print(f"  Max total precipitation: {total_precip.max().item():.2f} mm/week")
    print(f"  % area with precip > 10mm/week: {(total_precip > 10).float().mean().item()*100:.1f}%")

    # Estimate closure rate
    print("\n" + "="*50)
    print("Temperature Tendency Equation Closure Analysis")
    print("="*50)

    # A perfect closure would have loss near 0
    # Estimate closure rate based on loss magnitude
    typical_tendency = 1e-5  # K/s typical magnitude
    relative_error = torch.sqrt(loss).item() / typical_tendency
    closure_rate = max(0, 1 - relative_error) * 100

    print(f"Estimated closure rate: {closure_rate:.1f}%")

    if closure_rate > 80:
        print("✓ EXCELLENT: Temperature tendency equation is well balanced")
    elif closure_rate > 60:
        print("✓ GOOD: Temperature tendency equation shows reasonable closure")
    elif closure_rate > 40:
        print("⚠ FAIR: Temperature tendency equation has moderate imbalance")
    else:
        print("⚠ POOR: Temperature tendency equation shows significant imbalance")

    print("\nNote: Lower closure is expected due to:")
    print("  - Simplified diabatic heating parameterization")
    print("  - Numerical discretization errors")
    print("  - Missing sub-grid scale processes")

    return loss.item(), closure_rate

if __name__ == "__main__":
    loss, closure = test_temperature_tendency()
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"  Temperature tendency loss: {loss:.6e}")
    print(f"  Closure rate: {closure:.1f}%")
    print(f"{'='*50}")