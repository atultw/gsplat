"""Utility functions for Metal backend."""

import platform
from typing import Dict, Any
import torch
from torch import Tensor


def is_metal_available() -> bool:
    """Check if Metal GPU backend is available.
    
    Returns:
        True if Metal is available and functional.
    """
    # Only available on macOS
    if platform.system() != "Darwin":
        return False
    
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except ImportError:
        return False
    except Exception:
        return False


def get_metal_device_info() -> Dict[str, Any]:
    """Get information about the Metal GPU device.
    
    Returns:
        Dictionary with device name, memory, capabilities, etc.
    """
    if not is_metal_available():
        return {"available": False}
    
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return {"available": False}
        
        return {
            "available": True,
            "name": device.name(),
            "max_threads_per_threadgroup": device.maxThreadsPerThreadgroup(),
            "max_buffer_length": device.maxBufferLength(),
            "has_unified_memory": device.hasUnifiedMemory(),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_mps_tensor(tensor: Tensor) -> bool:
    """Check if tensor is on MPS (Metal) device.
    
    Args:
        tensor: PyTorch tensor.
        
    Returns:
        True if tensor is on MPS device.
    """
    return tensor.device.type == "mps"


def to_mps(tensor: Tensor) -> Tensor:
    """Move tensor to MPS device if available.
    
    Args:
        tensor: Input tensor.
        
    Returns:
        Tensor on MPS device.
    """
    if torch.backends.mps.is_available():
        return tensor.to("mps")
    return tensor


def validate_render_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
) -> None:
    """Validate input tensors for rendering.
    
    Raises:
        ValueError: If inputs are invalid.
    """
    if means.ndim < 2 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape [..., N, 3], got {means.shape}")
    
    N = means.shape[-2]
    
    if quats.shape[-2:] != (N, 4):
        raise ValueError(f"quats must have shape [..., {N}, 4], got {quats.shape}")
    
    if scales.shape[-2:] != (N, 3):
        raise ValueError(f"scales must have shape [..., {N}, 3], got {scales.shape}")
    
    if opacities.shape[-1] != N:
        raise ValueError(f"opacities must have shape [..., {N}], got {opacities.shape}")
    
    if viewmats.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must have shape [..., C, 4, 4], got {viewmats.shape}")
    
    if Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Ks must have shape [..., C, 3, 3], got {Ks.shape}")


def get_supported_channel_counts():
    """Get list of natively supported color channel counts.
    
    Returns:
        List of supported channel counts.
    """
    return [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64]


def pad_channels(colors: Tensor) -> tuple:
    """Pad color tensor to next supported channel count.
    
    Args:
        colors: Color tensor [..., C].
        
    Returns:
        Tuple of (padded_colors, original_channels).
    """
    channels = colors.shape[-1]
    supported = get_supported_channel_counts()
    
    if channels in supported:
        return colors, channels
    
    # Find next supported count
    for c in supported:
        if c > channels:
            target = c
            break
    else:
        raise ValueError(f"Unsupported channel count: {channels}")
    
    padding = target - channels
    padded = torch.cat([
        colors,
        torch.zeros(*colors.shape[:-1], padding, device=colors.device, dtype=colors.dtype)
    ], dim=-1)
    
    return padded, channels
