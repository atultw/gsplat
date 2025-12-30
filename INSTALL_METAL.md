# Metal Backend Installation

This is a Metal-only version of gsplat focused on depth rendering for macOS.

## Requirements

- macOS with Metal support
- Python 3.8+
- PyTorch 2.0+ with MPS backend

## Installation

### Option 1: Install from source (Metal-only)

```bash
pip install -e . -f setup_metal.py
```

### Option 2: Install with optional dependencies

```bash
# With PLY file loading support
pip install -e ".[ply]" -f setup_metal.py

# With visualization tools
pip install -e ".[viz]" -f setup_metal.py

# With everything
pip install -e ".[ply,viz]" -f setup_metal.py
```

## Verify Installation

```python
import torch
print("MPS available:", torch.backends.mps.is_available())

from gsplat.metal import is_available
print("Metal backend available:", is_available())
```

## Quick Test

```bash
python examples/metal_depth_example.py
```

## What's Included

This Metal-only version includes:

- ✅ Metal backend for depth rendering
- ✅ PLY file loading (optional)
- ✅ PyTorch MPS integration
- ✅ Simple Python API

## What's Removed

To focus on depth rendering, the following have been removed:

- ❌ All CUDA code and build system
- ❌ RGB/color rendering
- ❌ Spherical harmonics
- ❌ 2DGS support
- ❌ Training/optimization code
- ❌ Distributed rendering
- ❌ Compression utilities

## Documentation

See:
- `METAL_README.md` - Quick start guide
- `gsplat/metal/README.md` - Detailed API documentation
- `examples/metal_depth_example.py` - Working example

## Troubleshooting

### PyTorch MPS not available

Make sure you're using PyTorch 2.0+ on macOS:

```bash
pip install --upgrade torch torchvision
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

### Metal shaders not found

Make sure the package is installed with the shader files:

```bash
pip install -e . -f setup_metal.py
```

The shaders should be in `gsplat/metal/shaders/*.metal`.

## Development

To develop and test:

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev,ply,viz]" -f setup_metal.py

# Run example
python examples/metal_depth_example.py

# Run tests (when available)
pytest tests/test_metal.py
```

## Differences from Full gsplat

| Feature | Full gsplat | Metal-only |
|---------|-------------|------------|
| Backend | CUDA | Metal |
| Platform | Linux, Windows | macOS only |
| Rendering | RGB + Depth | Depth only |
| Training | Yes | No |
| Build | C++/CUDA compilation | No compilation |
| Dependencies | CUDA toolkit | PyTorch MPS |

## License

Same as gsplat main repository.
