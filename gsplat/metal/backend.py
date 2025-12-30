"""Metal backend implementation for Gaussian Splatting."""

import os
from pathlib import Path
from typing import Optional

import torch

# Try to import Metal framework (only available on macOS)
try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None


class MetalBackend:
    """Metal backend for Gaussian Splatting rendering."""
    
    def __init__(self):
        if not METAL_AVAILABLE:
            raise RuntimeError(
                "Metal backend is not available. "
                "Metal is only supported on macOS with PyTorch MPS support."
            )
        
        # Get Metal device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Failed to create Metal device")
        
        # Create command queue
        self.command_queue = self.device.newCommandQueue()
        
        # Load and compile shaders
        self._load_shaders()
        
        # Create pipeline states
        self._create_pipelines()
    
    def _load_shaders(self):
        """Load Metal shader files."""
        shader_dir = Path(__file__).parent / "shaders"
        
        # Read shader source files
        projection_src = (shader_dir / "projection.metal").read_text()
        rasterize_src = (shader_dir / "rasterize_depth.metal").read_text()
        
        # Combine all shader sources
        self.shader_source = projection_src + "\n" + rasterize_src
        
        # Compile shader library
        options = Metal.MTLCompileOptions.new()
        self.library, error = self.device.newLibraryWithSource_options_error_(
            self.shader_source, options, None
        )
        
        if error:
            raise RuntimeError(f"Failed to compile Metal shaders: {error}")
    
    def _create_pipelines(self):
        """Create Metal compute pipeline states."""
        # Projection pipeline
        projection_fn = self.library.newFunctionWithName_("project_gaussians")
        self.projection_pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(
                projection_fn, None
            )
        )
        if error:
            raise RuntimeError(f"Failed to create projection pipeline: {error}")
        
        # Rasterization pipeline
        rasterize_fn = self.library.newFunctionWithName_("rasterize_depth_tile")
        self.rasterize_pipeline, error = (
            self.device.newComputePipelineStateWithFunction_error_(
                rasterize_fn, None
            )
        )
        if error:
            raise RuntimeError(f"Failed to create rasterization pipeline: {error}")


# Global backend instance
_backend: Optional[MetalBackend] = None


def get_backend() -> MetalBackend:
    """Get or create the Metal backend instance."""
    global _backend
    if _backend is None:
        _backend = MetalBackend()
    return _backend


def is_available() -> bool:
    """Check if Metal backend is available."""
    if not METAL_AVAILABLE:
        return False
    
    # Check if PyTorch has MPS support
    if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
        return False
    
    return True
