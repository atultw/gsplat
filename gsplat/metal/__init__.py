"""Metal backend for Gaussian Splatting depth rendering."""

from .backend import is_available
from .rasterizer import load_ply, render_depth

__all__ = ["render_depth", "load_ply", "is_available"]
