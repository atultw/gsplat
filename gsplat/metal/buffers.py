"""Buffer management for Metal GPU tensors.

Handles conversion between PyTorch tensors and Metal buffers,
with zero-copy optimization when possible using MPS backend.
"""

from typing import Optional, Any, Tuple
import torch
from torch import Tensor

try:
    import Metal
    _METAL_AVAILABLE = True
except ImportError:
    Metal = None
    _METAL_AVAILABLE = False


def tensor_to_metal_buffer(
    tensor: Tensor,
    device: Any,
    copy: bool = False
) -> Tuple[Any, int]:
    """Convert a PyTorch tensor to a Metal buffer.
    
    When the tensor is on MPS device and contiguous, zero-copy may be possible.
    Otherwise, the data is copied to a new Metal buffer.
    
    Args:
        tensor: PyTorch tensor to convert.
        device: MetalDevice instance.
        copy: If True, always copy data even if zero-copy is possible.
        
    Returns:
        Tuple of (MTLBuffer, offset in bytes).
    """
    if not _METAL_AVAILABLE:
        raise RuntimeError("Metal is not available")
    
    # Ensure tensor is contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Get size in bytes
    size = tensor.numel() * tensor.element_size()
    
    # For MPS tensors, try to get underlying buffer
    if tensor.device.type == "mps" and not copy:
        # MPS tensors have their data on Metal already
        # We need to extract the buffer - this requires accessing private APIs
        # For now, fall back to copy
        pass
    
    # Copy path: create new buffer from tensor data
    tensor_cpu = tensor.detach().cpu()
    
    # Get raw bytes
    if tensor_cpu.dtype == torch.float32:
        data = tensor_cpu.numpy().tobytes()
    elif tensor_cpu.dtype == torch.float16:
        data = tensor_cpu.numpy().tobytes()
    elif tensor_cpu.dtype == torch.int32:
        data = tensor_cpu.numpy().tobytes()
    elif tensor_cpu.dtype == torch.int64:
        data = tensor_cpu.numpy().tobytes()
    elif tensor_cpu.dtype == torch.bool:
        data = tensor_cpu.numpy().tobytes()
    else:
        raise ValueError(f"Unsupported tensor dtype: {tensor_cpu.dtype}")
    
    # Create Metal buffer
    buffer = device.device.newBufferWithBytes_length_options_(
        data, len(data), Metal.MTLResourceStorageModeShared
    )
    
    return buffer, 0


def metal_buffer_to_tensor(
    buffer: Any,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    offset: int = 0
) -> Tensor:
    """Convert a Metal buffer to a PyTorch tensor.
    
    Args:
        buffer: MTLBuffer containing the data.
        shape: Shape of the output tensor.
        dtype: Data type of the output tensor.
        offset: Byte offset into the buffer.
        
    Returns:
        PyTorch tensor with the data.
    """
    import numpy as np
    
    # Map dtype to numpy dtype
    dtype_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    np_dtype = dtype_map[dtype]
    
    # Calculate size
    numel = 1
    for dim in shape:
        numel *= dim
    size = numel * np.dtype(np_dtype).itemsize
    
    # Read buffer contents using PyObjC's varlist.as_buffer() method
    contents = buffer.contents()
    
    if contents is None:
        raise RuntimeError("Failed to access Metal buffer contents")
    
    # Use as_buffer() to get a memoryview - this is the PyObjC way
    # to access the raw memory from an MTLBuffer
    mem = contents.as_buffer(buffer.length())
    
    # Create numpy array from memoryview (with offset support)
    np_array = np.frombuffer(mem, dtype=np_dtype, offset=offset, count=numel).reshape(shape)
    
    return torch.from_numpy(np_array.copy())


class BufferPool:
    """Pool of reusable Metal buffers to reduce allocation overhead.
    
    Buffers are cached by size and reused when available.
    """
    
    def __init__(self, device: Any, max_cached: int = 32):
        """Initialize buffer pool.
        
        Args:
            device: MetalDevice instance.
            max_cached: Maximum number of buffers to cache per size class.
        """
        self._device = device
        self._max_cached = max_cached
        self._pools: dict = {}  # size -> list of buffers
    
    def acquire(self, size: int) -> Any:
        """Acquire a buffer of at least the specified size.
        
        Args:
            size: Minimum size in bytes.
            
        Returns:
            MTLBuffer.
        """
        # Round up to power of 2 for better reuse
        size_class = 1
        while size_class < size:
            size_class *= 2
        
        if size_class in self._pools and self._pools[size_class]:
            return self._pools[size_class].pop()
        
        return self._device.create_buffer(size_class)
    
    def release(self, buffer: Any) -> None:
        """Return a buffer to the pool for reuse.
        
        Args:
            buffer: MTLBuffer to return.
        """
        size_class = buffer.length()
        
        if size_class not in self._pools:
            self._pools[size_class] = []
        
        if len(self._pools[size_class]) < self._max_cached:
            self._pools[size_class].append(buffer)
    
    def clear(self) -> None:
        """Release all cached buffers."""
        self._pools.clear()


def validate_tensor(tensor: Tensor, name: str, expected_ndim: Optional[int] = None) -> None:
    """Validate tensor properties for Metal operations.
    
    Args:
        tensor: Tensor to validate.
        name: Name for error messages.
        expected_ndim: Expected number of dimensions (optional).
        
    Raises:
        ValueError: If validation fails.
    """
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name} must be a PyTorch tensor")
    
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(f"{name} must have {expected_ndim} dimensions, got {tensor.ndim}")
    
    # Check dtype
    supported_dtypes = {torch.float32, torch.float16, torch.int32, torch.int64, torch.bool}
    if tensor.dtype not in supported_dtypes:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")


def ensure_contiguous_float32(tensor: Tensor) -> Tensor:
    """Ensure tensor is contiguous and float32.
    
    Args:
        tensor: Input tensor.
        
    Returns:
        Contiguous float32 tensor.
    """
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor
