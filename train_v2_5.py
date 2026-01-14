"""
CAS-Canglong V2.5 Training Script
V2.5: Wider/deeper wind-aware model with per-layer Top-K wind directions.
"""

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import h5py as h5

from canglong import CanglongV2_5
from canglong.wind_aware_shift import WIND_DIR_NAMES, get_dominant_direction


class WeatherDataset(Dataset):
    def __init__(self, surface_data, upper_air_data, start_idx, end_idx):
        """Initialize weather dataset - sequential time series split"""
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.start_idx = start_idx
        self.length = end_idx - start_idx - 2

        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        input_surface = self.surface_data[actual_idx:actual_idx+2]
        input_upper_air = self.upper_air_data[actual_idx:actual_idx+2]
        target_surface = self.surface_data[actual_idx+2]
        target_upper_air = self.upper_air_data[actual_idx+2]
        return input_surface, input_upper_air, target_surface, target_upper_air


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data_path = '/gz-data/ERA5_2023_weekly_new.h5'
    input_surface = h5.File(data_path)['surface']
    input_upper_air = h5.File(data_path)['upper_air']
    print(f"Surface data shape: {input_surface.shape}")
    print(f"Upper air data shape: {input_upper_air.shape}")

    # Load normalization parameters
    print("Loading normalization parameters...")
    sys.path.append('code_v2')
    from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    # Convert to torch tensors for training normalization
    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    print(f"Surface mean shape: {surface_mean.shape}")  # (1, 26, 1, 721, 1440)
    print(f"Upper mean shape: {upper_mean.shape}")      # (1, 10, 5, 1, 721, 1440)

    # Create train and validation datasets
    total_samples = 40
    train_end = 28
    train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=train_end)
    valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=train_end, end_idx=total_samples)

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(valid_dataset)}")

    # Model configuration
    embed_dim = 192
    num_heads = (12, 24, 24, 12)
    depths = (4, 8, 8, 4)
    max_wind_dirs = 2
    max_wind_dirs_by_layer = (2, 1, 1, 1)
    use_wind_aware_shift_by_layer = (True, False, False, False)
    drop_path_max = 0.3

    # Create model with normalization parameters for wind direction calculation
    model = CanglongV2_5(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
        max_wind_dirs=max_wind_dirs,
        max_wind_dirs_by_layer=max_wind_dirs_by_layer,
        use_wind_aware_shift_by_layer=use_wind_aware_shift_by_layer,
        drop_path_max=drop_path_max,
        surface_mean=torch.from_numpy(surface_mean_np),
        surface_std=torch.from_numpy(surface_std_np),
        upper_mean=torch.from_numpy(upper_mean_np),
        upper_std=torch.from_numpy(upper_std_np),
        use_checkpoint=True
    )

    # Multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Mixed precision training
    scaler = GradScaler('cuda')
    use_amp = True
    print(f"Mixed precision training: {use_amp}")

    # Create save directory
    save_dir = 'checkpoints_v2_5'
    os.makedirs(save_dir, exist_ok=True)

    # Try to resume from existing checkpoint
    resume = True
    start_epoch = 0
    checkpoint_files = sorted(Path(save_dir).glob('model_v2_5_epoch*.pth'),
                              key=lambda p: int(p.stem.split('epoch')[1]) if 'epoch' in p.stem else -1)

    if resume and checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        try:
            state_dict = torch.load(latest_checkpoint, map_location=device)
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            start_epoch = int(latest_checkpoint.stem.split('epoch')[1])
            print(f"Resumed model weights from {latest_checkpoint} (epoch {start_epoch}).")
        except (FileNotFoundError, ValueError, KeyError) as err:
            print(f"Failed to load checkpoint {latest_checkpoint}: {err}. Starting fresh training.")
            start_epoch = 0
    else:
        print("No checkpoint found. Starting fresh training.")

    # Training parameters
    num_epochs = 50
    checkpoint_interval = 25

    print("=" * 70)
    print("Training V2.5 with Wind-Aware Per-Window Shift + Matching Attention Mask")
    print("=" * 70)
    print(f"Window size: (2, 6, 12)")
    print(f"Wind shift scale: 2")
    print(f"Wind speed threshold: 0.5 m/s")
    print(f"Top-K wind directions (by layer): {max_wind_dirs_by_layer}")
    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Depths: {depths}")
    print(f"Drop path max: {drop_path_max}")
    print("=" * 70)

    # For wind direction statistics
    wind_direction_counts = {}

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # ==================== Training Phase ====================
        model.train()
        train_loss = 0.0
        surface_loss = 0.0
        upper_air_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        first_batch = True

        for input_surface, input_upper_air, target_surface, target_upper_air in train_pbar:
            input_surface = input_surface.float().to(device)
            input_upper_air = input_upper_air.float().to(device)
            target_surface = target_surface.float().to(device)
            target_upper_air = target_upper_air.float().to(device)

            # Normalize inputs and targets
            input_surface = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
            input_upper_air = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
            target_surface = (target_surface.unsqueeze(2) - surface_mean) / surface_std
            target_upper_air = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast('cuda', enabled=use_amp):
                output_surface, output_upper_air, wind_direction_id = model(
                    input_surface, input_upper_air, return_wind_info=True
                )

                # Print wind direction info for first batch of first epoch
                if first_batch and epoch == start_epoch:
                    print(f"\n  [Wind Direction Info]:")
                    print(f"  wind_direction_id shape: {wind_direction_id.shape}")
                    unique_dirs = torch.unique(wind_direction_id).cpu().tolist()
                    print(f"  Unique directions: {[WIND_DIR_NAMES.get(int(d), str(d)) for d in unique_dirs]}")

                    # Show sample of wind directions
                    sample_dirs = wind_direction_id[0, :5, :5].cpu().numpy()
                    print(f"  Sample (5x5 windows):")
                    for row in sample_dirs:
                        print(f"    {[WIND_DIR_NAMES.get(int(d), str(d)) for d in row]}")
                    print()
                    first_batch = False

                # Statistics on dominant wind direction
                dominant_id = get_dominant_direction(wind_direction_id)
                dir_name = WIND_DIR_NAMES.get(dominant_id, 'Unknown')
                wind_direction_counts[dir_name] = wind_direction_counts.get(dir_name, 0) + 1

                # Compute loss
                loss_surface = criterion(output_surface, target_surface)
                loss_upper_air = criterion(output_upper_air, target_upper_air)
                loss = loss_surface + loss_upper_air

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update losses
            train_loss += loss.item()
            surface_loss += loss_surface.item()
            upper_air_loss += loss_upper_air.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'surf': f"{loss_surface.item():.4f}",
                'upper': f"{loss_upper_air.item():.4f}",
                'wind': dir_name
            })

        # Print training stats
        avg_train_loss = train_loss / len(train_loader)
        avg_surface_loss = surface_loss / len(train_loader)
        avg_upper_air_loss = upper_air_loss / len(train_loader)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Total: {avg_train_loss:.6f}, Surface: {avg_surface_loss:.6f}, Upper Air: {avg_upper_air_loss:.6f}")

        # ==================== Validation Phase ====================
        model.eval()
        valid_loss = 0.0
        valid_surface_loss = 0.0
        valid_upper_air_loss = 0.0

        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for input_surface, input_upper_air, target_surface, target_upper_air in valid_pbar:
                input_surface = input_surface.float().to(device)
                input_upper_air = input_upper_air.float().to(device)
                target_surface = target_surface.float().to(device)
                target_upper_air = target_upper_air.float().to(device)

                # Normalize inputs and targets
                input_surface = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
                input_upper_air = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
                target_surface = (target_surface.unsqueeze(2) - surface_mean) / surface_std
                target_upper_air = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std

                # Forward pass
                with autocast('cuda', enabled=use_amp):
                    output_surface, output_upper_air = model(input_surface, input_upper_air)
                    loss_surface = criterion(output_surface, target_surface)
                    loss_upper_air = criterion(output_upper_air, target_upper_air)
                    loss = loss_surface + loss_upper_air

                # Update losses
                valid_loss += loss.item()
                valid_surface_loss += loss_surface.item()
                valid_upper_air_loss += loss_upper_air.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_surface_loss = valid_surface_loss / len(valid_loader)
        avg_valid_upper_air_loss = valid_upper_air_loss / len(valid_loader)

        print(f"  Valid - Total: {avg_valid_loss:.6f}, Surface: {avg_valid_surface_loss:.6f}, Upper Air: {avg_valid_upper_air_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_path = os.path.join(save_dir, f"model_v2_5_epoch{epoch+1}.pth")
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  Saved checkpoint: {save_path}")

    print("Training completed!")
