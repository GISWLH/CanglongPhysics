"""
CAS-Canglong Weather Prediction Model Package
=============================================

This package contains the Canglong model for sub-seasonal to seasonal weather prediction.

Example usage:
    from canglong import Canglong

    # Create model
    model = Canglong()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Forward pass
    output_surface, output_upper_air = model(surface_data, upper_air_data)
"""

# Import the main model
from .model_v1 import Canglong
from .model_v2 import CanglongV2
from .model_v2_1 import CanglongV2_1
from .model_v3 import CanglongV3

# Import utility functions if needed
from .earth_position import calculate_position_bias_indices
from .shift_window import create_shifted_window_mask, partition_windows, reverse_partition
from .embed import ImageToPatch2D, ImageToPatch3D, ImageToPatch4D
from .recovery import RecoveryImage2D, RecoveryImage3D, RecoveryImage4D
from .pad import calculate_padding_2d, calculate_padding_3d
from .crop import center_crop_2d, center_crop_3d

# Version information
__version__ = '1.0.0'
__author__ = 'CAS-Canglong Team'

# Define what gets imported with "from canglong import *"
__all__ = [
    'Canglong',
    'CanglongV2',
    'CanglongV2_1',
    'CanglongV3',
    'calculate_position_bias_indices',
    'create_shifted_window_mask',
    'partition_windows',
    'reverse_partition',
    'ImageToPatch2D',
    'ImageToPatch3D',
    'ImageToPatch4D',
    'RecoveryImage2D',
    'RecoveryImage3D',
    'RecoveryImage4D',
    'calculate_padding_2d',
    'calculate_padding_3d',
    'center_crop_2d',
    'center_crop_3d'
]
