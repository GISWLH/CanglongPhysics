import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from canglong.shift_window import partition_windows, reverse_partition
from canglong.pad import calculate_padding_3d
from canglong.crop import center_crop_3d
from canglong.wind_aware_shift import WIND_SHIFT_DIRECTIONS
from canglong.wind_aware_block import WindAwareEarthAttention3D


def create_shifted_window_mask_v2_1(resolution, window_dims, shift_dims):
    """
    Shifted window mask that supports zero shift in any dimension.
    """
    Pl, Lat, Lon = resolution
    win_pl, win_lat, win_lon = window_dims
    shift_pl, shift_lat, shift_lon = shift_dims

    def build_segments(size, win, shift):
        if shift == 0 or win <= 0:
            return [slice(0, size)]
        return [slice(0, -win), slice(-win, -shift), slice(-shift, None)]

    mask_tensor = torch.zeros((1, Pl, Lat, Lon, 1))

    pl_segments = build_segments(Pl, win_pl, shift_pl)
    lat_segments = build_segments(Lat, win_lat, shift_lat)
    lon_segments = build_segments(Lon, win_lon, shift_lon)

    counter = 0
    for pl in pl_segments:
        for lat in lat_segments:
            for lon in lon_segments:
                mask_tensor[:, pl, lat, lon, :] = counter
                counter += 1

    masked_windows = partition_windows(mask_tensor, window_dims)
    masked_windows = masked_windows.view(masked_windows.shape[0], masked_windows.shape[1], win_pl * win_lat * win_lon)
    attention_mask = masked_windows.unsqueeze(2) - masked_windows.unsqueeze(3)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0, float(0.0))
    return attention_mask


class WindAwareEarthSpecificBlockV2_1(nn.Module):
    """
    Wind-aware transformer block with per-window shifts and matching masks.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_wind_aware_shift=True, wind_shift_scale=2,
                 max_wind_dirs=None):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_wind_aware_shift = use_wind_aware_shift
        self.wind_shift_scale = wind_shift_scale
        self.max_wind_dirs = max_wind_dirs
        self._attn_mask_cache = {}

        self.norm1 = norm_layer(dim)
        padding = calculate_padding_3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])
        self.pad_resolution = tuple(pad_resolution)

        self.attn = WindAwareEarthAttention3D(
            dim=dim,
            input_resolution=self.pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def _normalize_wind_map(self, wind_direction_id, num_lat, num_lon, device):
        if wind_direction_id is None:
            return torch.zeros((1, num_lat, num_lon), device=device, dtype=torch.long)
        if isinstance(wind_direction_id, int):
            wind_map = torch.full((1, num_lat, num_lon), wind_direction_id, device=device, dtype=torch.long)
            return wind_map
        wind_map = wind_direction_id.to(device)
        if wind_map.dim() == 1:
            wind_map = wind_map.view(-1, 1, 1).expand(-1, num_lat, num_lon)
            return wind_map
        if wind_map.shape[-2:] != (num_lat, num_lon):
            wind_map = F.interpolate(wind_map.unsqueeze(1).float(), size=(num_lat, num_lon), mode='nearest')
            wind_map = wind_map.squeeze(1).long()
        return wind_map

    def _limit_wind_dirs(self, wind_map):
        if not self.max_wind_dirs or self.max_wind_dirs <= 0:
            return torch.zeros_like(wind_map)
        if self.max_wind_dirs >= 8:
            return wind_map

        flat = wind_map.view(-1)
        counts = torch.bincount(flat, minlength=9)
        nonzero_counts = counts[1:]
        if nonzero_counts.sum() == 0:
            return torch.zeros_like(wind_map)

        k = min(self.max_wind_dirs, 8)
        topk = torch.topk(nonzero_counts, k=k).indices + 1

        allowed = torch.zeros(9, device=wind_map.device, dtype=torch.bool)
        allowed[0] = True
        allowed[topk] = True

        mask = allowed[wind_map]
        wind_map = torch.where(mask, wind_map, torch.zeros_like(wind_map))
        return wind_map

    def _clamp_shift(self, shift, window):
        if window <= 1:
            return 0
        if abs(shift) >= window:
            return (window - 1) if shift > 0 else -(window - 1)
        return shift

    def _get_total_shift(self, direction_id):
        shift_pl, shift_lat, shift_lon = self.shift_size
        wind_lat = 0
        wind_lon = 0

        if self.use_wind_aware_shift and direction_id != 0:
            lat_dir, lon_dir = WIND_SHIFT_DIRECTIONS.get(direction_id, (0, 0))
            wind_lat = lat_dir * self.wind_shift_scale
            wind_lon = lon_dir * self.wind_shift_scale

        total_shift_pl = shift_pl
        total_shift_lat = shift_lat + wind_lat
        total_shift_lon = shift_lon + wind_lon

        total_shift_pl = self._clamp_shift(total_shift_pl, self.window_size[0])
        total_shift_lat = self._clamp_shift(total_shift_lat, self.window_size[1])
        total_shift_lon = self._clamp_shift(total_shift_lon, self.window_size[2])

        return total_shift_pl, total_shift_lat, total_shift_lon

    def _get_attn_mask(self, shift_sizes, device):
        shift_pl, shift_lat, shift_lon = shift_sizes
        if shift_pl == 0 and shift_lat == 0 and shift_lon == 0:
            return None
        key = (str(device), shift_pl, shift_lat, shift_lon)
        if key not in self._attn_mask_cache:
            mask = create_shifted_window_mask_v2_1(self.pad_resolution, self.window_size, shift_sizes)
            self._attn_mask_cache[key] = mask.to(device)
        return self._attn_mask_cache[key]

    def forward(self, x: torch.Tensor, wind_direction_id=None):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        assert L == Pl * Lat * Lon, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        win_pl, win_lat, win_lon = self.window_size
        num_lat = Lat_pad // win_lat
        num_lon = Lon_pad // win_lon

        if not self.use_wind_aware_shift:
            wind_direction_id = None

        wind_map = self._normalize_wind_map(wind_direction_id, num_lat, num_lon, x.device)
        wind_map = self._limit_wind_dirs(wind_map)
        if wind_map.shape[0] == 1 and B > 1:
            wind_map = wind_map.expand(B, -1, -1)

        unique_dirs = torch.unique(wind_map).tolist()
        combined = torch.zeros_like(x)

        for dir_id in unique_dirs:
            total_shift = self._get_total_shift(int(dir_id))
            shift_sizes = (abs(total_shift[0]), abs(total_shift[1]), abs(total_shift[2]))
            attn_mask = self._get_attn_mask(shift_sizes, x.device)

            x_shifted = x
            if total_shift != (0, 0, 0):
                x_shifted = torch.roll(x, shifts=(-total_shift[0], -total_shift[1], -total_shift[2]), dims=(1, 2, 3))

            x_windows = partition_windows(x_shifted, self.window_size)
            x_windows = x_windows.view(x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C)

            attn_windows = self.attn(x_windows, mask=attn_mask)

            attn_windows = attn_windows.view(attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C)
            out = reverse_partition(attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad)

            if total_shift != (0, 0, 0):
                out = torch.roll(out, shifts=(total_shift[0], total_shift[1], total_shift[2]), dims=(1, 2, 3))

            mask = (wind_map == dir_id).float()
            mask = mask.repeat_interleave(win_lat, dim=1).repeat_interleave(win_lon, dim=2)
            mask = mask.unsqueeze(1).unsqueeze(-1)
            combined = combined + out * mask

        x = combined

        x = center_crop_3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(0, 2, 3, 4, 1)

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
