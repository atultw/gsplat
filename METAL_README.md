# Metal Backend for Depth Rendering

## Overview

The gsplat Metal backend provides depth map rendering from Gaussian splats on macOS devices without requiring CUDA. This is a focused implementation that supports the essential depth rendering functionality.

## Quick Start

```python
import torch
from gsplat.metal import render_depth, is_available

# Check availability
if not is_available():
    print("Metal not available - requires macOS with PyTorch MPS")
    exit(1)

# Load or create Gaussian parameters
params = {
    "means": torch.randn(1000, 3),       # Gaussian positions
    "quats": torch.randn(1000, 4),       # Rotations (wxyz)
    "scales": torch.randn(1000, 3),      # Log scales
    "opacities": torch.randn(1000),      # Logit opacities
}

# Define camera
viewmat = torch.eye(4)  # World-to-camera transform
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)

# Render depth
depth, alpha, _ = render_depth(
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

## Features

✅ **Depth rendering** from Gaussian splats  
✅ **Metal-native** implementation (no CUDA required)  
✅ **ParameterDict** input format support  
✅ **PLY file** loading (optional)  
✅ **Simple API** - just render depth for any camera viewpoint

## What's Removed

To focus on depth rendering, the following features have been removed:

❌ RGB/color rendering  
❌ Spherical harmonics (SH) evaluation  
❌ 2DGS (2D Gaussian Splatting)  
❌ Training/optimization code  
❌ Distributed rendering  
❌ Compression utilities  
❌ All CUDA `.cu` files

## Documentation

See [`gsplat/metal/README.md`](gsplat/metal/README.md) for detailed documentation, including:
- API reference
- Camera parameter formats
- Examples and usage patterns
- Implementation details

## Example

Run the included example:

```bash
# With synthetic scene
python examples/metal_depth_example.py

# With PLY file
python examples/metal_depth_example.py path/to/splat.ply
```

## Requirements

- **macOS** with Metal support
- **PyTorch** with MPS backend
- **Python** 3.8+

Optional:
- `plyfile` for PLY file loading
- `PIL` for saving depth maps as images

## Installation

The Metal backend is included with gsplat. Just ensure PyTorch has MPS support:

```bash
pip install torch torchvision
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Camera Parameters

### Viewmat (World-to-Camera Transform)

4x4 transformation matrix. Camera looks down +Z in its local frame:

```python
# Camera at position looking at origin
viewmat = torch.eye(4)
viewmat[:3, 3] = -camera_position
```

### K (Camera Intrinsics)

3x3 intrinsics matrix:

```python
K = torch.tensor([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=torch.float32)
```

Compute focal length from FOV:

```python
import numpy as np
fov_rad = np.deg2rad(fov_degrees)
focal_length = width / (2.0 * np.tan(fov_rad / 2.0))
```

## Parameter Formats

| Parameter | Format | Description |
|-----------|--------|-------------|
| `means` | [N, 3] | Gaussian 3D positions |
| `quats` | [N, 4] | Quaternions in **wxyz** format (w=scalar) |
| `scales` | [N, 3] | **Log-space** scales (actual = exp(scales)) |
| `opacities` | [N] | **Logit-space** opacities (actual = sigmoid(opacities)) |

## Architecture

```
gsplat/metal/
├── __init__.py           # Public API
├── backend.py            # Metal device and shader management
├── rasterizer.py         # Main rendering functions
├── README.md             # Detailed documentation
└── shaders/
    ├── projection.metal   # 3D→2D projection
    └── rasterize_depth.metal  # Depth accumulation
```

## Performance

The Metal backend provides competitive performance on Apple Silicon:
- Tile-based rasterization for efficient memory access
- Front-to-back depth sorting for early ray termination
- Parallel compute kernels

## Limitations

- **Inference only** (no backward pass/training)
- **Depth maps only** (no RGB rendering)
- **macOS only** (Metal is Apple-specific)

## Future Work

- [ ] Complete native Metal kernel implementation
- [ ] Optimize tile-based rasterization
- [ ] Benchmark vs CUDA implementation
- [ ] Add gradient computation for training

## Contributing

Contributions welcome! Areas needing work:
- Metal kernel optimization
- Cross-platform testing
- Additional camera models
- Performance benchmarking
