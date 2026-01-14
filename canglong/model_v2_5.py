"""
CAS-Canglong V2.5 model.
V2.5: Wider/deeper wind-aware model with Top-4 (L1/L4) and Top-2 (L2/L3) directions.
"""

from .model_v2_2 import CanglongV2_2


class CanglongV2_5(CanglongV2_2):
    """
    V2.5 = V2.2 with larger embed_dim, deeper stages, and per-layer wind Top-K.
    """

    def __init__(self, embed_dim=192, num_heads=(12, 24, 24, 12), depths=(4, 8, 8, 4),
                 window_size=(2, 6, 12), wind_shift_scale=2, wind_speed_threshold=0.5,
                 max_wind_dirs=2, max_wind_dirs_by_layer=(2, 1, 1, 1),
                 use_wind_aware_shift_by_layer=(True, False, False, False), drop_path_max=0.3,
                 norm_json=None, use_checkpoint=False,
                 surface_mean=None, surface_std=None, upper_mean=None, upper_std=None):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depths=depths,
            window_size=window_size,
            wind_shift_scale=wind_shift_scale,
            wind_speed_threshold=wind_speed_threshold,
            max_wind_dirs=max_wind_dirs,
            max_wind_dirs_by_layer=max_wind_dirs_by_layer,
            use_wind_aware_shift_by_layer=use_wind_aware_shift_by_layer,
            drop_path_max=drop_path_max,
            norm_json=norm_json,
            use_checkpoint=use_checkpoint,
            surface_mean=surface_mean,
            surface_std=surface_std,
            upper_mean=upper_mean,
            upper_std=upper_std,
        )
