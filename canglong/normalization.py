"""Normalization utilities for ERA5-based surface and upper-air tensors.

This module centralizes normalization loading and broadcasting logic so that
training scripts can share the same implementation instead of duplicating tensor
permutations in multiple places.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

# The conversion helper already knows how to expand the statistics to include
# singleton batch/time dimensions that match the training tensors.
from code_v2.convert_dict_to_pytorch_arrays import load_normalization_arrays

ArrayLikePath = Union[str, Path]
MaybeDevice = Optional[Union[str, torch.device]]


@dataclass
class NormalizationStats:
    """Container that keeps mean/std tensors and provides convenience helpers."""

    surface_mean: torch.Tensor
    surface_std: torch.Tensor
    upper_mean: torch.Tensor
    upper_std: torch.Tensor

    def to(self, device: MaybeDevice = None, dtype: Optional[torch.dtype] = None) -> "NormalizationStats":
        """Return a copy of the stats on the requested device/dtype."""
        kwargs = {}
        if device is not None:
            kwargs["device"] = torch.device(device)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return NormalizationStats(
            surface_mean=self.surface_mean.to(**kwargs),
            surface_std=self.surface_std.to(**kwargs),
            upper_mean=self.upper_mean.to(**kwargs),
            upper_std=self.upper_std.to(**kwargs),
        )

    def normalize_surface(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batched surface fields (B, 17, T, 721, 1440)."""
        return (tensor - self.surface_mean) / self.surface_std

    def normalize_upper(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize batched upper-air fields (B, 7, 5, T, 721, 1440)."""
        return (tensor - self.upper_mean) / self.upper_std

    def denormalize_surface(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo normalization for surface outputs shaped like (B, 17, 1, 721, 1440)."""
        return tensor * self.surface_std + self.surface_mean

    def denormalize_upper(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo normalization for upper-air outputs shaped like (B, 7, 5, 1, 721, 1440)."""
        return tensor * self.upper_std + self.upper_mean


def load_stats(json_path: ArrayLikePath, *, device: MaybeDevice = None, dtype: Optional[torch.dtype] = torch.float32,
               verbose: bool = False) -> NormalizationStats:
    """Load ERA5 normalization statistics from the shared JSON specification."""
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(
        str(json_path), verbose=verbose
    )
    stats = NormalizationStats(
        surface_mean=torch.from_numpy(surface_mean_np),
        surface_std=torch.from_numpy(surface_std_np),
        upper_mean=torch.from_numpy(upper_mean_np),
        upper_std=torch.from_numpy(upper_std_np),
    )
    return stats.to(device=device, dtype=dtype)


def normalize_batch(surface: torch.Tensor, upper: torch.Tensor, stats: NormalizationStats) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize surface and upper-air tensors with shared stats."""
    return stats.normalize_surface(surface), stats.normalize_upper(upper)


def denormalize_batch(surface: torch.Tensor, upper: torch.Tensor, stats: NormalizationStats) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denormalize surface and upper-air tensors with shared stats."""
    return stats.denormalize_surface(surface), stats.denormalize_upper(upper)
