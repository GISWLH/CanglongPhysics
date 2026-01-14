"""
Aligned benchmark for Canglong V2.1/V2.2.
Matches train_v2_1.py / train_v2_2.py data loading and DataLoader settings.
"""

import argparse
import time

import h5py as h5
import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from canglong import CanglongV2_1, CanglongV2_2


class WeatherDataset(Dataset):
    def __init__(self, surface_data, upper_air_data, start_idx, end_idx):
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.start_idx = start_idx
        self.length = end_idx - start_idx - 2
        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        input_surface = self.surface_data[actual_idx:actual_idx + 2]
        input_upper_air = self.upper_air_data[actual_idx:actual_idx + 2]
        target_surface = self.surface_data[actual_idx + 2]
        target_upper_air = self.upper_air_data[actual_idx + 2]
        return input_surface, input_upper_air, target_surface, target_upper_air


def normalize_batch(input_surface, input_upper_air, target_surface, target_upper_air,
                    surface_mean, surface_std, upper_mean, upper_std, device):
    input_surface = input_surface.float().to(device)
    input_upper_air = input_upper_air.float().to(device)
    target_surface = target_surface.float().to(device)
    target_upper_air = target_upper_air.float().to(device)

    input_surface = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
    input_upper_air = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
    target_surface = (target_surface.unsqueeze(2) - surface_mean) / surface_std
    target_upper_air = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std

    return input_surface, input_upper_air, target_surface, target_upper_air


def run_loader(epoch_idx, loader, model, optimizer, criterion, scaler, use_amp, device, phase):
    is_train = phase == "train"
    model.train(is_train)
    epoch_start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch_idx} [Train]" if is_train else f"Epoch {epoch_idx} [Valid]")

    for input_surface, input_upper_air, target_surface, target_upper_air in pbar:
        input_surface, input_upper_air, target_surface, target_upper_air = normalize_batch(
            input_surface, input_upper_air, target_surface, target_upper_air,
            surface_mean, surface_std, upper_mean, upper_std, device
        )

        if is_train:
            optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            output_surface, output_upper_air = model(input_surface, input_upper_air)
            loss_surface = criterion(output_surface, target_surface)
            loss_upper_air = criterion(output_upper_air, target_upper_air)
            loss = loss_surface + loss_upper_air

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    torch.cuda.synchronize()
    return time.time() - epoch_start


def parse_args():
    parser = argparse.ArgumentParser(description="Aligned benchmark for Canglong V2.1/V2.2")
    parser.add_argument("--model", choices=["v2_1", "v2_2"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-valid", action="store_true")
    parser.add_argument("--max-wind-dirs", type=int, default=2,
                        help="Max wind directions for V2.2 (top-K).")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def build_model(model_name, surface_mean_np, surface_std_np, upper_mean_np, upper_std_np, device, max_wind_dirs):
    if model_name == "v2_1":
        return CanglongV2_1(
            surface_mean=torch.from_numpy(surface_mean_np),
            surface_std=torch.from_numpy(surface_std_np),
            upper_mean=torch.from_numpy(upper_mean_np),
            upper_std=torch.from_numpy(upper_std_np),
        ).to(device)
    return CanglongV2_2(
        surface_mean=torch.from_numpy(surface_mean_np),
        surface_std=torch.from_numpy(surface_std_np),
        upper_mean=torch.from_numpy(upper_mean_np),
        upper_std=torch.from_numpy(upper_std_np),
        use_checkpoint=True,
        max_wind_dirs=max_wind_dirs,
    ).to(device)


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this benchmark.")

    print("Using device:", device)
    print("Loading data...")
    data_path = '/gz-data/ERA5_2023_weekly_new.h5'
    input_surface = h5.File(data_path)['surface']
    input_upper_air = h5.File(data_path)['upper_air']

    print("Loading normalization parameters...")
    import sys
    sys.path.append('code_v2')
    from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    total_samples = 40
    train_end = 28
    train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=train_end)
    valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=train_end, end_idx=total_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(valid_dataset)}")

    model = build_model(
        args.model,
        surface_mean_np,
        surface_std_np,
        upper_mean_np,
        upper_std_np,
        device,
        args.max_wind_dirs
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()
    use_amp = not args.no_amp
    scaler = GradScaler('cuda') if use_amp else None

    train_times = []
    valid_times = []
    train_peak_alloc = []
    train_peak_reserved = []
    valid_peak_alloc = []
    valid_peak_reserved = []

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        train_time = run_loader(epoch, train_loader, model, optimizer, criterion, scaler, use_amp, device, "train")
        train_peak_alloc.append(torch.cuda.max_memory_allocated())
        train_peak_reserved.append(torch.cuda.max_memory_reserved())
        train_times.append(train_time)

        if not args.no_valid:
            with torch.no_grad():
                torch.cuda.reset_peak_memory_stats()
                valid_time = run_loader(epoch, valid_loader, model, optimizer, criterion, scaler, use_amp, device, "valid")
                valid_peak_alloc.append(torch.cuda.max_memory_allocated())
                valid_peak_reserved.append(torch.cuda.max_memory_reserved())
            valid_times.append(valid_time)

        print(f"Epoch {epoch}/{args.epochs} train_time={train_time:.2f}s" +
              (f", valid_time={valid_time:.2f}s" if not args.no_valid else ""))

    avg_train_epoch = sum(train_times) / len(train_times)
    avg_train_step = avg_train_epoch / len(train_loader)

    report_lines = []
    report_lines.append(f"# {args.model} aligned benchmark\n")
    report_lines.append(f"- epochs: {args.epochs}\n")
    report_lines.append(f"- num_workers: {args.num_workers}\n")
    report_lines.append(f"- batch_size: {args.batch_size}\n")
    report_lines.append(f"- amp: {str(use_amp).lower()}\n")
    if args.model == "v2_2":
        report_lines.append(f"- max_wind_dirs: {args.max_wind_dirs}\n")
    report_lines.append(f"- train_avg_epoch_s: {avg_train_epoch:.2f}\n")
    report_lines.append(f"- train_avg_step_s: {avg_train_step:.2f}\n")
    if train_peak_alloc:
        report_lines.append(f"- train_peak_alloc_gb: {max(train_peak_alloc) / 1024**3:.2f}\n")
        report_lines.append(f"- train_peak_reserved_gb: {max(train_peak_reserved) / 1024**3:.2f}\n")
    if valid_times:
        avg_valid_epoch = sum(valid_times) / len(valid_times)
        report_lines.append(f"- valid_avg_epoch_s: {avg_valid_epoch:.2f}\n")
    if valid_peak_alloc:
        report_lines.append(f"- valid_peak_alloc_gb: {max(valid_peak_alloc) / 1024**3:.2f}\n")
        report_lines.append(f"- valid_peak_reserved_gb: {max(valid_peak_reserved) / 1024**3:.2f}\n")

    output_path = args.output or f"analysis/{args.model}_aligned_benchmark.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(report_lines))
    print(f"Report saved to {output_path}")
