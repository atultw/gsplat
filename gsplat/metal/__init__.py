# Metal backend for gsplat Gaussian splatting renderer
# Inference-only implementation for Apple Silicon GPUs

from .backend import MetalDevice, is_metal_available, get_metal_device_info
from .rendering import rasterization_metal
from .wrapper import (
    fully_fused_projection_metal,
    isect_tiles_metal,
    isect_offset_encode_metal,
    rasterize_to_pixels_metal,
    spherical_harmonics_metal,
)

__all__ = [
    # Device management
    "MetalDevice",
    "is_metal_available",
    "get_metal_device_info",
    # High-level API
    "rasterization_metal",
    # Low-level wrappers
    "fully_fused_projection_metal",
    "isect_tiles_metal",
    "isect_offset_encode_metal",
    "rasterize_to_pixels_metal",
    "spherical_harmonics_metal",
]
