"""
CAS-Canglong V2.3 model.
V2.3: Wind-aware per-window shift with matching attention mask (Top-4 directions).
"""

from .model_v2_2 import CanglongV2_2


class CanglongV2_3(CanglongV2_2):
    """
    V2.3 = V2.2 with default max_wind_dirs=4.
    """

    def __init__(self, embed_dim=96, num_heads=(8, 16, 16, 8), window_size=(2, 6, 12),
                 wind_shift_scale=2, wind_speed_threshold=0.5, max_wind_dirs=4, norm_json=None,
                 use_checkpoint=False,
                 surface_mean=None, surface_std=None, upper_mean=None, upper_std=None):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            wind_shift_scale=wind_shift_scale,
            wind_speed_threshold=wind_speed_threshold,
            max_wind_dirs=max_wind_dirs,
            norm_json=norm_json,
            use_checkpoint=use_checkpoint,
            surface_mean=surface_mean,
            surface_std=surface_std,
            upper_mean=upper_mean,
            upper_std=upper_std,
        )
