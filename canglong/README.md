# CAS-Canglong Model V1

A PyTorch implementation of the CAS-Canglong weather prediction model for sub-seasonal to seasonal forecasting.

## Quick Start

### One-Line Import

```python
from canglong import Canglong

# Create model
model = Canglong()
```

That's it! The model is now ready to use.

## Basic Usage

```python
import torch
from canglong import Canglong

# Initialize model
model = Canglong()

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input data shapes:
# - surface: (batch, 26, time, lat, lon) = (1, 26, 2, 721, 1440)
# - upper_air: (batch, 10, 5, time, lat, lon) = (1, 10, 5, 2, 721, 1440)

# Run inference
model.eval()
with torch.no_grad():
    output_surface, output_upper_air = model(input_surface, input_upper_air)

# Output shapes:
# - output_surface: (1, 26, 1, 721, 1440)
# - output_upper_air: (1, 10, 5, 1, 721, 1440)
```

## Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from canglong import Canglong

# Initialize model
model = Canglong()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training step
model.train()
optimizer.zero_grad()

# Forward pass
output_surface, output_upper_air = model(input_surface, input_upper_air)

# Calculate loss
loss_surface = criterion(output_surface, target_surface)
loss_upper_air = criterion(output_upper_air, target_upper_air)
loss = loss_surface + loss_upper_air

# Backward pass
loss.backward()
optimizer.step()
```

## With Normalization

```python
import sys
sys.path.append('code_v2')
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays
from canglong import Canglong

# Load normalization parameters
json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path)

# Initialize model
model = Canglong().to(device)

# Normalize inputs
input_surface_norm = (input_surface - surface_mean) / surface_std
input_upper_air_norm = (input_upper_air - upper_mean) / upper_std

# Run model
output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)

# Denormalize outputs
output_surface_physical = output_surface * surface_std + surface_mean
output_upper_air_physical = output_upper_air * upper_std + upper_mean
```

## Model Architecture

The Canglong model consists of:
- **Surface Encoder**: Conv3D + ResNet blocks for processing surface variables
- **Upper Air Encoder**: 4D patch embedding for multi-level atmospheric data
- **Earth Transformer**: 3D Swin-Transformer with Earth-specific position bias
- **Multi-scale Processing**: Downsample → Process → Upsample with skip connections
- **Decoders**: Separate decoders for surface and upper air predictions

## Input Variables

### Surface Variables (26 total)
Temperature, precipitation, wind, pressure, radiation, soil moisture, etc.

### Upper Air Variables (10 variables × 5 pressure levels)
Temperature, humidity, wind, geopotential height, etc. at 200, 300, 500, 700, 850 hPa

## Model Parameters

- **embed_dim**: Base embedding dimension (default: 96)
- **num_heads**: Attention heads in different layers (default: (8, 16, 16, 8))
- **window_size**: Window size for attention (default: (2, 6, 12))
- **Total parameters**: ~52.8M

## Requirements

- Python 3.8+
- PyTorch 1.10+
- timm
- numpy

## Files

- `model_v1.py`: Main model definition
- `__init__.py`: Package initialization
- Supporting modules in parent directory:
  - `earth_position.py`: Earth-specific position encoding
  - `shift_window.py`: Shifted window attention
  - `embed.py`: Patch embedding layers
  - `recovery.py`: Patch recovery layers
  - `helper.py`: ResNet and utility blocks

## Citation

If you use this model, please cite:
```
CAS-Canglong: A skillful 3D Transformer model for sub-seasonal to seasonal global weather prediction
```