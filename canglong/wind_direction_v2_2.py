import torch
import torch.nn.functional as F
import numpy as np


def _angle_to_direction_id(angle_deg):
    """
    Convert angle (0-360, 0=E, 90=N) to 8-way direction id.
    0 is reserved for calm (no shift).
    """
    angle = angle_deg % 360
    direction_id = torch.full_like(angle, 3, dtype=torch.long)  # E default

    direction_id[(angle >= 22.5) & (angle < 67.5)] = 2   # NE
    direction_id[(angle >= 67.5) & (angle < 112.5)] = 1  # N
    direction_id[(angle >= 112.5) & (angle < 157.5)] = 8 # NW
    direction_id[(angle >= 157.5) & (angle < 202.5)] = 7 # W
    direction_id[(angle >= 202.5) & (angle < 247.5)] = 6 # SW
    direction_id[(angle >= 247.5) & (angle < 292.5)] = 5 # S
    direction_id[(angle >= 292.5) & (angle < 337.5)] = 4 # SE

    return direction_id


def _calculate_padding_2d(target_size, window_size):
    """
    Calculate 2D padding to make target_size divisible by window_size.

    Returns:
        (pad_left, pad_right, pad_top, pad_bottom) for F.pad
    """
    Lat, Lon = target_size
    win_lat, win_lon = window_size

    # Latitude padding (top/bottom)
    lat_mod = Lat % win_lat
    if lat_mod:
        lat_pad_total = win_lat - lat_mod
        pad_top = lat_pad_total // 2
        pad_bottom = lat_pad_total - pad_top
    else:
        pad_top = pad_bottom = 0

    # Longitude padding (left/right)
    lon_mod = Lon % win_lon
    if lon_mod:
        lon_pad_total = win_lon - lon_mod
        pad_left = lon_pad_total // 2
        pad_right = lon_pad_total - pad_left
    else:
        pad_left = pad_right = 0

    return (pad_left, pad_right, pad_top, pad_bottom)


class WindDirectionProcessorV2_2(torch.nn.Module):
    """
    Window-level wind direction processor for V2.2.

    Notes:
    - Expects physical (denormalized) u/v values.
    - Returns per-window direction ids aligned with attention windows.
    """

    def __init__(self, window_size=(6, 12), target_size=(181, 360), wind_speed_threshold=0.5):
        super().__init__()
        self.window_size = window_size
        self.target_size = target_size
        self.wind_speed_threshold = wind_speed_threshold
        self.pad = _calculate_padding_2d(target_size, window_size)

    def forward(self, surface, upper_air,
                surface_uv_mean=None, surface_uv_std=None,
                upper_uv_mean=None, upper_uv_std=None):
        """
        Args:
            surface: (B, 26, 2, 721, 1440)
            upper_air: (B, 10, 5, 2, 721, 1440)
            surface_uv_mean/std: (1, 2, 1, 721, 1440) for u10/v10 if inputs are normalized
            upper_uv_mean/std: (1, 2, 5, 1, 721, 1440) for u/v if inputs are normalized
        Returns:
            direction_id: (B, num_lat_windows, num_lon_windows) long tensor, 0-8
        """
        upper_u = upper_air[:, 3]
        upper_v = upper_air[:, 4]
        if upper_uv_mean is not None and upper_uv_std is not None:
            upper_u = upper_u * upper_uv_std[:, 0] + upper_uv_mean[:, 0]
            upper_v = upper_v * upper_uv_std[:, 1] + upper_uv_mean[:, 1]
        upper_u = upper_u.mean(dim=1).mean(dim=1)  # (B, 721, 1440)
        upper_v = upper_v.mean(dim=1).mean(dim=1)

        surface_u = surface[:, 7]
        surface_v = surface[:, 8]
        if surface_uv_mean is not None and surface_uv_std is not None:
            surface_u = surface_u * surface_uv_std[:, 0] + surface_uv_mean[:, 0]
            surface_v = surface_v * surface_uv_std[:, 1] + surface_uv_mean[:, 1]
        surface_u = surface_u.mean(dim=1)  # (B, 721, 1440)
        surface_v = surface_v.mean(dim=1)

        combined_u = 0.5 * (upper_u + surface_u)
        combined_v = 0.5 * (upper_v + surface_v)

        u_down = F.adaptive_avg_pool2d(combined_u.unsqueeze(1), self.target_size).squeeze(1)
        v_down = F.adaptive_avg_pool2d(combined_v.unsqueeze(1), self.target_size).squeeze(1)

        pad_left, pad_right, pad_top, pad_bottom = self.pad
        u_pad = F.pad(u_down, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        v_pad = F.pad(v_down, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

        win_lat, win_lon = self.window_size
        B, H, W = u_pad.shape
        num_lat = H // win_lat
        num_lon = W // win_lon

        u_win = u_pad.view(B, num_lat, win_lat, num_lon, win_lon).mean(dim=(2, 4))
        v_win = v_pad.view(B, num_lat, win_lat, num_lon, win_lon).mean(dim=(2, 4))

        speed = torch.sqrt(u_win ** 2 + v_win ** 2)
        angle = torch.atan2(v_win, u_win) * 180.0 / np.pi
        angle = (angle + 360.0) % 360.0

        direction_id = _angle_to_direction_id(angle)
        calm_mask = speed < self.wind_speed_threshold
        direction_id = torch.where(calm_mask, torch.zeros_like(direction_id), direction_id)

        return direction_id
