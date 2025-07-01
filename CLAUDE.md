# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CanglongPhysics is a research project focused on adding physics information to AI weather prediction models. The main goal is to build physics-informed neural networks (PINNs) for weather forecasting that incorporate physical constraints and patterns into deep learning models.

## Key Components & Architecture

### Core Models
- **Canglong Model**: Main transformer-based weather prediction model located in `weatherlearn/models/`
- **Pangu Weather**: Reference implementation for 3D transformer weather models (`weatherlearn/models/pangu/`)
- **FuXi**: Alternative weather prediction model (`weatherlearn/models/fuxi/`)

### Data Processing Pipeline
The project processes multi-dimensional weather data from ERA5:
- **Surface Variables**: 16 variables including precipitation, temperature, pressure, wind components
- **Upper Air Variables**: 7 variables across 4 pressure levels (300, 500, 700, 850 hPa)
- **Static Data**: Topography, land cover, soil type stored in `constant_masks/`

### Custom Neural Network Components
Located in `canglong/` directory:
- **Conv4d.py**: 4D convolutional operations for spatio-temporal data
- **embed.py**: Patch embedding for 2D, 3D, and 4D data
- **recovery.py**: Patch recovery operations
- **earth_position.py**: Earth-specific position encoding for global data
- **shift_window.py**: Shifted window attention mechanisms
- **pad.py/crop.py**: Spatial padding and cropping utilities

### Physics Integration
The project implements several physics-informed approaches:
- **Data Scaling**: Learning physical scales from 40 years of ERA5 data
- **PINN Physics**: Integration of Navier-Stokes equations
- **Vector Quantization**: Codebook approach for discrete feature representation

## Data Structure & Formats

### Input Data
- ERA5 data accessed via Google Cloud Storage (`gs://gcp-public-data-arco-era5/`)
- Weekly averaged data for seasonal forecasting (6-week predictions)
- Spatial resolution: 0.25° global grid (721x1440)

### Key Notebooks
- `code/how_to_run.ipynb`: Main workflow and model execution
- `code/generate_weekly.ipynb`: Data preprocessing for weekly forecasts
- `code/model_performance.ipynb`: Model evaluation and metrics

## Running the Code

### Main Execution
The primary execution script is `code/run.py`, which contains the complete pipeline from data loading to model inference.

### Data Access
The code expects:
- ERA5 data access through xarray and zarr
- Pre-computed constant masks in `constant_masks/`
- Model checkpoints in the expected paths (referenced in notebooks)

### Dependencies
Key Python packages:
- PyTorch for deep learning
- xarray for multidimensional data
- cartopy for geospatial plotting
- salem for geographic data processing
- cmaps for meteorological color schemes

## Model Architecture Details

### Canglong Model Structure
1. **Patch Embedding**: Converts 2D/3D/4D data into tokens
2. **Earth-Specific Attention**: 3D transformer blocks with earth position bias
3. **Multi-Scale Processing**: Down/up-sampling between different resolutions
4. **Physics Integration**: Encoder-decoder architecture with VAE-like components

### Key Resolutions
- High-res: (4, 181, 360) - pressure levels, latitude, longitude
- Low-res: (4, 91, 180) - downsampled for computational efficiency

## Physics Concepts

### SPEI Calculation
The project includes Standardized Precipitation Evapotranspiration Index (SPEI) computation for drought monitoring using log-logistic distribution fitting.

### Data Normalization
Variables are normalized using statistics from 40 years of ERA5 data, with specific handling for different physical scales (e.g., precipitation vs temperature).

## Development Notes

### Model Training
- The model supports 6-week rolling forecasts
- Uses pre-trained weights for inference
- Implements teacher forcing for training

### Evaluation Metrics
- Spatial correlation analysis
- Anomaly calculations against climatology
- Comparison with ECMWF operational forecasts

This is a research codebase focused on advancing physics-informed weather prediction. The code combines traditional meteorological knowledge with modern deep learning techniques.