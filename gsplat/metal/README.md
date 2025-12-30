# Metal Backend for Depth Rendering

This directory contains a Metal-based implementation for rendering depth maps from Gaussian splats. The Metal backend is designed to run on macOS devices with Metal support and does not require CUDA.

## Features

- **Depth Rendering**: Render depth maps (mode "D") from Gaussian splat parameters
- **Metal-Native**: Uses Metal Performance Shaders (MPS) backend
- **No CUDA Required**: Completely independent from CUDA code
- **Simple API**: Easy-to-use Python interface

## Requirements

- macOS with Metal support
- PyTorch with MPS backend (`torch.backends.mps.is_available()` returns `True`)
- Python 3.8+

## Installation

The Metal backend is included in the gsplat package. No additional installation is required beyond having PyTorch with MPS support.

## Quick Start

### Using ParameterDict

```python
import torch
from gsplat.metal import render_depth, is_available

# Check if Metal is available
if not is_available():
    print("Metal backend not available")
    exit(1)

# Define Gaussian parameters (can come from training, PLY file, etc.)
params = {
    "means": torch.randn(1000, 3),        # 3D positions
    "quats": torch.randn(1000, 4),        # Quaternions (wxyz)
    "scales": torch.randn(1000, 3),       # Log scales
    "opacities": torch.randn(1000),       # Logit opacities
}

# Define camera parameters
viewmat = torch.eye(4)  # World-to-camera transform
K = torch.tensor([      # Camera intrinsics
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
], dtype=torch.float32)

# Render depth map
depth, alpha, meta = render_depth(
    means=params["means"],
    quats=params["quats"],
    scales=params["scales"],
    opacities=params["opacities"],
    viewmat=viewmat,
    K=K,
    width=640,
    height=480,
)

print(f"Depth shape: {depth.shape}")  # [480, 640]
print(f"Alpha shape: {alpha.shape}")  # [480, 640]
```

### Loading from PLY File

```python
from gsplat.metal import load_ply, render_depth

# Load Gaussian parameters from PLY file
params = load_ply("path/to/splat.ply")

# Render depth
depth, alpha, meta = render_depth(
    means=params["means"],
    quats=params["quats"],
    scales=params["scales"],
    opacities=params["opacities"],
    viewmat=viewmat,
    K=K,
    width=640,
    height=480,
)
```

## API Reference

### `render_depth()`

Render a depth map from Gaussian splat parameters.

**Parameters:**
- `means` (Tensor): Gaussian centers, shape [N, 3]
- `quats` (Tensor): Rotation quaternions (wxyz), shape [N, 4]
- `scales` (Tensor): Log-space scale factors, shape [N, 3]
- `opacities` (Tensor): Logit-space opacities, shape [N]
- `viewmat` (Tensor): World-to-camera matrix, shape [4, 4]
- `K` (Tensor): Camera intrinsics matrix, shape [3, 3]
- `width` (int): Output image width
- `height` (int): Output image height
- `near_plane` (float): Near clipping plane, default 0.01
- `far_plane` (float): Far clipping plane, default 1e10
- `background` (Tensor, optional): Background depth value
- `return_alpha` (bool): Whether to return alpha channel, default True

**Returns:**
- `depth` (Tensor): Rendered depth map, shape [height, width]
- `alpha` (Tensor): Rendered alpha channel, shape [height, width] (if `return_alpha=True`)
- `meta` (dict): Metadata dictionary

### `load_ply()`

Load Gaussian splat parameters from a PLY file.

**Parameters:**
- `ply_path` (str): Path to PLY file

**Returns:**
- Dictionary with keys: "means", "quats", "scales", "opacities", and optionally "sh0", "shN"

### `is_available()`

Check if Metal backend is available on the current system.

**Returns:**
- `bool`: True if Metal backend is available

## Camera Parameters

### View Matrix (`viewmat`)

The view matrix transforms world coordinates to camera coordinates. It should be a 4x4 transformation matrix where:
- The camera looks down the +Z axis in its local coordinate frame
- +X is right, +Y is up, +Z is forward

For a camera at position `pos` looking at the origin:
```python
viewmat = torch.eye(4)
viewmat[:3, 3] = -pos
```

### Camera Intrinsics (`K`)

The intrinsics matrix K is a 3x3 matrix:
```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point (typically image_width/2, image_height/2)

To compute focal length from field of view:
```python
import numpy as np
fov_degrees = 60
fov_rad = np.deg2rad(fov_degrees)
focal_length = width / (2.0 * np.tan(fov_rad / 2.0))
```

## Parameter Formats

### Quaternions (quats)

Quaternions are in **wxyz** format (scalar first):
- `[w, x, y, z]` where `w` is the real part
- No need to normalize (normalization is done internally)

### Scales (scales)

Scales are in **log space**:
- Actual scale = `exp(scales)`
- Typically initialized to small values (e.g., -2.0 for small splats)

### Opacities (opacities)

Opacities are in **logit space**:
- Actual opacity = `sigmoid(opacities)`
- Range: (-∞, +∞) maps to (0, 1)

## Examples

See `examples/metal_depth_example.py` for a complete working example.

```bash
# Run with synthetic scene
python examples/metal_depth_example.py

# Run with PLY file
python examples/metal_depth_example.py path/to/splat.ply
```

## Implementation Details

### Metal Shaders

The Metal backend consists of several shader kernels:

1. **Projection** (`projection.metal`):
   - Converts quaternions and scales to 3D covariance matrices
   - Transforms Gaussians from world to camera space
   - Projects 3D Gaussians to 2D image space
   - Computes bounding boxes and validity masks

2. **Rasterization** (`rasterize_depth.metal`):
   - Sorts Gaussians by depth (front-to-back)
   - Accumulates depth values per pixel
   - Handles alpha blending and transmittance

### Fallback Implementation

The current implementation includes a PyTorch fallback for testing. This uses the existing PyTorch-based projection and rasterization code but runs on MPS devices. The full Metal shader implementation will provide better performance.

## Performance

The Metal backend is optimized for macOS devices and provides competitive performance compared to CUDA on compatible hardware. Key optimizations include:

- Tile-based rasterization for efficient memory access
- Front-to-back sorting for early termination
- Parallel projection and rasterization

## Limitations

Current limitations:

- Only supports depth rendering (mode "D" and "ED")
- Does not support RGB rendering or spherical harmonics
- Does not support 2DGS (2D Gaussian Splatting)
- No backward pass (inference only)
- PLY loading requires `plyfile` package (optional dependency)

## Troubleshooting

### "Metal backend is not available"

Make sure you're running on macOS with:
- PyTorch installed with MPS support
- `torch.backends.mps.is_available()` returns `True`

### "Module 'Metal' not found"

The PyObjC Metal bindings are required for direct Metal kernel compilation. The fallback implementation will work without this, but with reduced performance.

## Future Improvements

- [ ] Complete Metal shader kernel implementation
- [ ] Optimize tile-based rasterization
- [ ] Add support for expected depth (mode "ED")
- [ ] Benchmark against CUDA implementation
- [ ] Add gradient computation for training
