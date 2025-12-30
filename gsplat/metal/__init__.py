"""Metal backend for Gaussian Splatting depth rendering."""

from .backend import is_available
from .rasterizer import load_ply, render_depth
from .add_splats import add_splats_from_image, depth_to_points, estimate_depth_from_rendered

__all__ = [
    "render_depth",
    "load_ply",
    "is_available",
    "add_splats_from_image",
    "depth_to_points",
    "estimate_depth_from_rendered",
]
