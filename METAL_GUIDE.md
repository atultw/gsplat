# Complete Guide to Metal Backend for Depth Rendering

This guide provides complete instructions for using the Metal backend for depth map rendering from Gaussian splats on macOS.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Parameter Guide](#parameter-guide)
7. [Camera Setup](#camera-setup)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tips](#performance-tips)

## Overview

The gsplat Metal backend provides a focused, CUDA-free implementation for rendering depth maps from 3D Gaussian splats. It's designed specifically for macOS devices with Metal support.

### Key Features

- ✅ **Metal-native**: No CUDA required
- ✅ **Depth rendering**: Accurate depth map generation
- ✅ **Simple API**: Easy to use Python interface
- ✅ **ParameterDict support**: Works with standard parameter format
- ✅ **PLY loading**: Optional PLY file support

### What's Not Included

- ❌ RGB/color rendering
- ❌ Spherical harmonics evaluation
- ❌ 2D Gaussian Splatting (2DGS)
- ❌ Training/backward pass
- ❌ Distributed rendering

## Installation

### Prerequisites

1. **macOS** with Metal support (macOS 10.13+)
2. **Python** 3.8 or later
3. **PyTorch** 2.0+ with MPS backend

### Check Prerequisites

```bash
# Check Python version
python --version  # Should be 3.8+

# Check PyTorch and MPS
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

### Install PyTorch with MPS

If PyTorch doesn't have MPS support:

```bash
pip install --upgrade torch torchvision
```

### Install gsplat Metal Backend

```bash
cd gsplat
pip install -e .
```

Or with optional dependencies:

```bash
# With PLY file support
pip install -e ".[ply]"

# With visualization tools
pip install -e ".[ply,viz]"
```

### Verify Installation

```python
from gsplat.metal import is_available
print("Metal available:", is_available())
```

## Quick Start

### Minimal Example

```python
import torch
from gsplat.metal import render_depth

# Create Gaussian parameters
params = {
    "means": torch.randn(100, 3),      # Positions
    "quats": torch.randn(100, 4),      # Rotations (wxyz)
    "scales": torch.ones(100, 3) * -2,  # Log scales
    "opacities": torch.ones(100) * 2,   # Logit opacities
}

# Camera parameters
viewmat = torch.eye(4)  # Identity (camera at origin)
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)

# Render
depth, alpha, _ = render_depth(
    params["means"], params["quats"], params["scales"], params["opacities"],
    viewmat, K, width=640, height=480
)

print(f"Depth shape: {depth.shape}")  # torch.Size([480, 640])
```

### With PLY File

```python
from gsplat.metal import load_ply, render_depth

# Load from PLY
params = load_ply("splat.ply")

# Setup camera
viewmat = torch.eye(4)
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)

# Render
depth, alpha, _ = render_depth(
    params["means"], params["quats"], params["scales"], params["opacities"],
    viewmat, K, 640, 480
)
```

## API Reference

### `render_depth()`

Main function for rendering depth maps.

```python
def render_depth(
    means: Tensor,           # [N, 3]
    quats: Tensor,           # [N, 4]
    scales: Tensor,          # [N, 3]
    opacities: Tensor,       # [N]
    viewmat: Tensor,         # [4, 4]
    K: Tensor,               # [3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    background: Optional[Tensor] = None,
    return_alpha: bool = True,
) -> Tuple[Tensor, Optional[Tensor], Dict]
```

**Parameters:**

- `means`: Gaussian 3D positions in world coordinates
- `quats`: Rotation quaternions in **wxyz** format (w is scalar part)
- `scales`: **Log-space** scale factors (actual scale = exp(scales))
- `opacities`: **Logit-space** opacities (actual opacity = sigmoid(opacities))
- `viewmat`: 4x4 world-to-camera transformation matrix
- `K`: 3x3 camera intrinsics matrix
- `width`, `height`: Output image dimensions in pixels
- `near_plane`: Near clipping distance (default: 0.01)
- `far_plane`: Far clipping distance (default: 1e10)
- `background`: Background depth value (default: 0.0)
- `return_alpha`: Whether to return alpha channel (default: True)

**Returns:**

- `depth`: Depth map, shape [height, width]
- `alpha`: Alpha channel, shape [height, width] (if return_alpha=True)
- `meta`: Dictionary with rendering metadata

### `load_ply()`

Load Gaussian parameters from a PLY file.

```python
def load_ply(ply_path: str) -> Dict[str, Tensor]
```

**Parameters:**

- `ply_path`: Path to PLY file

**Returns:**

Dictionary with keys:
- `"means"`: [N, 3] positions
- `"quats"`: [N, 4] quaternions
- `"scales"`: [N, 3] scales
- `"opacities"`: [N] opacities
- `"sh0"`: [N, 1, 3] SH coefficients (if present)
- `"shN"`: [N, K, 3] higher-order SH (if present)

### `is_available()`

Check if Metal backend is available.

```python
def is_available() -> bool
```

**Returns:**

- `True` if Metal backend is available, `False` otherwise

## Examples

### Example 1: Simple Scene

```python
import torch
import numpy as np
from gsplat.metal import render_depth

# Create a line of Gaussians
N = 20
means = torch.zeros(N, 3)
means[:, 2] = torch.linspace(-10, 10, N)  # Line along Z

quats = torch.tensor([[1, 0, 0, 0]] * N, dtype=torch.float32)
scales = torch.ones(N, 3) * -2.0
opacities = torch.ones(N) * 3.0

# Camera
viewmat = torch.eye(4)
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)

# Render
depth, alpha, _ = render_depth(means, quats, scales, opacities, viewmat, K, 640, 480)
```

### Example 2: Multiple Viewpoints

```python
def create_camera_orbit(radius=10, num_views=8):
    """Create cameras in a circle around origin."""
    angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)
    viewmats = []
    
    for angle in angles:
        # Camera position on circle
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        # Look at origin
        viewmat = torch.eye(4)
        # Simplified - would need proper look-at matrix
        viewmat[0, 3] = -x
        viewmat[2, 3] = -z
        viewmats.append(viewmat)
    
    return viewmats

# Render from multiple viewpoints
viewmats = create_camera_orbit()
K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)

depths = []
for viewmat in viewmats:
    depth, _, _ = render_depth(means, quats, scales, opacities, viewmat, K, 640, 480)
    depths.append(depth)
```

### Example 3: Save Visualization

```python
from PIL import Image
import numpy as np

# Render depth
depth, alpha, _ = render_depth(means, quats, scales, opacities, viewmat, K, 640, 480)

# Normalize for visualization
depth_vis = depth.cpu().numpy()
depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
depth_vis = (depth_vis * 255).astype(np.uint8)

# Save as image
Image.fromarray(depth_vis, mode='L').save('depth.png')

# Save alpha
alpha_vis = (alpha.cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(alpha_vis, mode='L').save('alpha.png')
```

## Parameter Guide

### Quaternions (quats)

Quaternions represent 3D rotations in **wxyz** format:

```python
# Identity rotation (no rotation)
quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]

# Random rotation
quat = torch.randn(4)
quat = quat / quat.norm()  # Normalize
```

**Important:** Format is [w, x, y, z] where w is the scalar/real part.

### Scales (scales)

Scales are in **log space** for numerical stability:

```python
# Small Gaussians (radius ~0.13)
scales = torch.ones(N, 3) * -2.0

# Medium Gaussians (radius ~1.0)
scales = torch.zeros(N, 3)

# Large Gaussians (radius ~7.4)
scales = torch.ones(N, 3) * 2.0

# Anisotropic (different per axis)
scales = torch.tensor([[-1.0, 0.0, 0.5]])  # [log(sx), log(sy), log(sz)]
```

**Actual scale:** `exp(scales)`

### Opacities (opacities)

Opacities are in **logit space**:

```python
# Very transparent (~0.01)
opacity = torch.tensor([-4.6])

# Half transparent (~0.5)
opacity = torch.tensor([0.0])

# Very opaque (~0.99)
opacity = torch.tensor([4.6])

# Fully opaque (sigmoid → 1)
opacity = torch.tensor([10.0])
```

**Actual opacity:** `sigmoid(opacities) = 1 / (1 + exp(-opacities))`

### Converting from Standard Format

If you have parameters in standard (non-log/logit) format:

```python
# From standard scales to log scales
scales_log = torch.log(scales_standard)

# From standard opacities [0, 1] to logit
opacities_logit = torch.logit(opacities_standard)
# or manually: log(opacity / (1 - opacity))
```

## Camera Setup

### View Matrix (viewmat)

The view matrix transforms world coordinates to camera coordinates.

**Convention:**
- Camera looks down **+Z** axis in its local frame
- +X is right, +Y is up

```python
import torch

def create_lookat_matrix(eye, target, up):
    """Create a look-at view matrix."""
    z = (eye - target)
    z = z / z.norm()
    
    x = torch.cross(up, z)
    x = x / x.norm()
    
    y = torch.cross(z, x)
    
    viewmat = torch.eye(4)
    viewmat[0, :3] = x
    viewmat[1, :3] = y
    viewmat[2, :3] = z
    viewmat[:3, 3] = -torch.tensor([x.dot(eye), y.dot(eye), z.dot(eye)])
    
    return viewmat

# Example: Camera at (0, 0, -10) looking at origin
viewmat = create_lookat_matrix(
    eye=torch.tensor([0.0, 0.0, -10.0]),
    target=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0])
)
```

### Camera Intrinsics (K)

The intrinsics matrix defines the camera's projection:

```python
K = torch.tensor([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=torch.float32)
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point (typically width/2, height/2)

**From field of view:**

```python
import numpy as np

def fov_to_intrinsics(fov_degrees, width, height):
    """Convert field of view to intrinsics matrix."""
    fov_rad = np.deg2rad(fov_degrees)
    fx = fy = width / (2.0 * np.tan(fov_rad / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    
    return torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

K = fov_to_intrinsics(fov_degrees=60, width=640, height=480)
```

**From focal length in mm:**

```python
def focal_mm_to_pixels(focal_mm, sensor_width_mm, image_width_pixels):
    """Convert focal length from mm to pixels."""
    fx = (focal_mm / sensor_width_mm) * image_width_pixels
    return fx

# Example: 50mm lens on full-frame sensor (36mm width)
fx = focal_mm_to_pixels(50, 36, 640)  # ~889 pixels
```

## Troubleshooting

### "Metal backend not available"

**Causes:**
1. Not running on macOS
2. PyTorch doesn't have MPS support
3. Metal device creation failed

**Solutions:**

```bash
# Check system
uname -s  # Should output "Darwin" on macOS

# Check PyTorch MPS
python -c "import torch; print(torch.backends.mps.is_available())"

# Reinstall PyTorch
pip install --upgrade torch torchvision
```

### "RuntimeError: MPS backend is not available"

PyTorch MPS might not be enabled. Try:

```python
import torch

# Check if MPS is built
print("MPS built:", torch.backends.mps.is_built())

# Check if MPS is available
print("MPS available:", torch.backends.mps.is_available())

# Try creating MPS tensor
try:
    x = torch.randn(10).to("mps")
    print("MPS works!")
except:
    print("MPS not working")
```

### Depth map is all zeros

**Possible causes:**
1. All Gaussians are behind the camera
2. Gaussians are outside the view frustum
3. Opacities are too low

**Debug:**

```python
# Check Gaussian positions relative to camera
means_cam = (viewmat @ torch.cat([means, torch.ones(N, 1)], dim=1).T).T[:, :3]
print("Z depths:", means_cam[:, 2].min(), means_cam[:, 2].max())
# Positive Z = in front of camera

# Check opacities
opacities_prob = torch.sigmoid(opacities)
print("Opacity range:", opacities_prob.min(), opacities_prob.max())

# Check if any Gaussians project to image
print("Means 2D range:", means2d.min(dim=0), means2d.max(dim=0))
print("Image size:", width, height)
```

### ImportError: No module named 'plyfile'

PLY loading is optional. Install if needed:

```bash
pip install plyfile
```

Or don't use `load_ply()` - create parameters manually instead.

## Performance Tips

### 1. Batch Processing

Process multiple viewpoints efficiently:

```python
# Instead of loop
depths = [render_depth(...)[0] for viewmat in viewmats]

# Better: reuse allocations (if possible in future versions)
# Or render in parallel if independent
```

### 2. Reduce Resolution for Preview

```python
# Full resolution
depth_full = render_depth(means, quats, scales, opacities, viewmat, K, 1920, 1080)[0]

# Preview at lower resolution
depth_preview = render_depth(means, quats, scales, opacities, viewmat, K, 480, 270)[0]
```

### 3. Filter Gaussians

Remove Gaussians that won't be visible:

```python
# Simple frustum culling
means_cam = transform_points(means, viewmat)
visible = (means_cam[:, 2] > near_plane) & (means_cam[:, 2] < far_plane)

# Render only visible
depth = render_depth(
    means[visible], quats[visible], scales[visible], opacities[visible],
    viewmat, K, width, height
)[0]
```

### 4. Use Appropriate Data Types

```python
# Float32 is sufficient (don't use float64)
means = means.to(torch.float32)
```

## Advanced Topics

### Custom Camera Models

Currently only pinhole camera is supported. For other models, you'll need to modify the projection in the Metal shaders.

### Gradient Computation

The current Metal backend is inference-only. For training, you'd need to implement backward passes in the Metal kernels.

### Batch Rendering

Future versions may support batched rendering of multiple cameras simultaneously. Currently, render each view separately.

## Support and Contributing

- **Issues:** https://github.com/atultw/gsplat/issues
- **Examples:** See `examples/metal_depth_example.py`
- **Tests:** Run `pytest tests/test_metal.py`

## License

Same license as main gsplat repository.
