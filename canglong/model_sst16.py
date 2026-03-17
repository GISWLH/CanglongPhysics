"""
CAS-Canglong SST16 Ocean Model
Sea Surface Temperature prediction model using 3D Swin-Transformer.
Input: 16 months × 8 variables → Output: 16 months × 1 variable (SST)
"""

import torch
import torch.nn as nn
import numpy as np

from .embed import ImageToPatch2D, ImageToPatch3D
from .recovery import RecoveryImage2D, RecoveryImage3D
from .model_v1 import UpSample, DownSample, BasicLayer


class CanglongSST16(nn.Module):
    """
    CAS-Canglong SST 16-month prediction model.

    Input:  (B, 8, 16, 721, 1440) — 8 variables × 16 months
        ch0: SST, ch1: MSL, ch2: SLHF, ch3: U10, ch4: V10,
        ch5: Z500, ch6: Z850, ch7: SSR
    Output: (B, 1, 16, 721, 1440) — predicted SST × 16 months
    """

    def __init__(self, embed_dim=72, num_heads=(6, 12, 12, 6), window_size=(2, 6, 12)):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.patchembed2d = ImageToPatch2D(
            img_dims=(721, 1440),
            patch_dims=(4, 4),
            in_channels=4 + 3,
            out_channels=embed_dim,
        )
        self.patchembed3d = ImageToPatch3D(
            img_dims=(13, 721, 1440),
            patch_dims=(2, 4, 4),
            in_channels=5,
            out_channels=embed_dim
        )

        self.patchembed_wlh = ImageToPatch3D(
            img_dims=(16, 721, 1440),
            patch_dims=(2, 4, 4),
            in_channels=8,
            out_channels=embed_dim
        )

        self.layer1 = BasicLayer(
            dim=embed_dim,
            input_resolution=(8, 181, 360),
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.downsample = DownSample(in_dim=embed_dim, input_resolution=(8, 181, 360), output_resolution=(8, 91, 180))
        self.layer2 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(8, 91, 180),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.layer3 = BasicLayer(
            dim=embed_dim * 2,
            input_resolution=(8, 91, 180),
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:]
        )
        self.upsample = UpSample(embed_dim * 2, embed_dim, (8, 91, 180), (8, 181, 360))
        self.layer4 = BasicLayer(
            dim=embed_dim,
            input_resolution=(8, 181, 360),
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2]
        )
        self.patchrecovery2d = RecoveryImage2D((721, 1440), (4, 4), 2 * embed_dim, 4)
        self.patchrecovery3d = RecoveryImage3D((13, 721, 1440), (2, 4, 4), 2 * embed_dim, 5)
        self.patchrecovery3d_wlh = RecoveryImage3D((16, 721, 1440), (2, 4, 4), 2 * embed_dim, 1)

    def forward(self, upper_air):
        wlh_input = self.patchembed_wlh(upper_air)
        x = wlh_input

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output = self.patchrecovery3d_wlh(output)

        return output
