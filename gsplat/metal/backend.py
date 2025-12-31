"""Metal GPU backend management for gsplat.

Provides device initialization, shader compilation, and pipeline management
for Apple Silicon GPUs using Metal Performance Shaders.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import weakref

# Check for Metal availability
_METAL_AVAILABLE = False
_metal_device_instance = None

try:
    import Metal
    import MetalPerformanceShaders as MPS
    _METAL_AVAILABLE = True
except ImportError:
    Metal = None
    MPS = None


def is_metal_available() -> bool:
    """Check if Metal GPU backend is available on this system."""
    if not _METAL_AVAILABLE:
        return False
    if sys.platform != "darwin":
        return False
    # Try to get a Metal device
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None
    except Exception:
        return False


def get_metal_device_info() -> Dict[str, Any]:
    """Get information about the Metal GPU device.
    
    Returns:
        Dictionary with device information including name, memory, and capabilities.
    """
    if not is_metal_available():
        return {"available": False}
    
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        return {"available": False}
    
    return {
        "available": True,
        "name": device.name(),
        "registry_id": device.registryID(),
        "max_threads_per_threadgroup": device.maxThreadsPerThreadgroup(),
        "max_buffer_length": device.maxBufferLength(),
        "has_unified_memory": device.hasUnifiedMemory(),
        "recommended_max_working_set_size": device.recommendedMaxWorkingSetSize(),
    }


class MetalDevice:
    """Manages Metal device, command queues, and shader pipelines.
    
    This class handles compilation of Metal shaders and caching of
    compute pipeline states for efficient kernel dispatch.
    
    Usage:
        device = MetalDevice()
        device.compile_shaders()
        # Use device.get_pipeline("kernel_name") to get compute pipelines
    """
    
    _instance = None
    _shaders_compiled = False
    
    def __new__(cls):
        """Singleton pattern - one device instance per process."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        if not is_metal_available():
            raise RuntimeError(
                "Metal is not available on this system. "
                "Metal requires macOS with Apple Silicon or AMD GPU."
            )
        
        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Failed to create Metal device")
        
        self._command_queue = self._device.newCommandQueue()
        if self._command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")
        
        self._library = None
        self._pipelines: Dict[str, Any] = {}
        self._initialized = True
    
    @property
    def device(self):
        """The underlying MTLDevice."""
        return self._device
    
    @property
    def command_queue(self):
        """The command queue for submitting work."""
        return self._command_queue
    
    def compile_shaders(self, force: bool = False) -> None:
        """Compile all Metal shaders.
        
        Args:
            force: If True, recompile even if already compiled.
        """
        if MetalDevice._shaders_compiled and not force:
            return
        
        shader_dir = Path(__file__).parent / "shaders"
        
        # Read header first
        header_content = ""
        header_path = shader_dir / "common.h"
        if header_path.exists():
            with open(header_path, 'r') as f:
                header_content = f.read()
            # Remove #pragma once as it's not needed when inlining
            header_content = header_content.replace('#pragma once', '// Header inlined')
        
        # Collect all shader source files
        shader_sources = []
        for shader_file in sorted(shader_dir.glob("*.metal")):
            with open(shader_file, 'r') as f:
                source = f.read()
            # Remove #include "common.h" since we'll prepend header
            source = source.replace('#include "common.h"', '// common.h inlined above')
            shader_sources.append(f"// === {shader_file.name} ===\n{source}")
        
        if not shader_sources:
            raise RuntimeError(f"No shader files found in {shader_dir}")
        
        # Combine: header first, then all shaders
        combined_source = header_content + "\n\n" + "\n\n".join(shader_sources)
        
        options = Metal.MTLCompileOptions.new()
        options.setFastMathEnabled_(True)
        
        error_ptr = None
        self._library, error_ptr = self._device.newLibraryWithSource_options_error_(
            combined_source, options, None
        )
        
        if self._library is None:
            error_msg = str(error_ptr) if error_ptr else "Unknown error"
            raise RuntimeError(f"Failed to compile Metal shaders: {error_msg}")
        
        MetalDevice._shaders_compiled = True
    
    def get_pipeline(
        self, 
        function_name: str,
        constants: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Get or create a compute pipeline for a kernel function.
        
        Args:
            function_name: Name of the kernel function in the Metal shader.
            constants: Optional function constants for specialization.
            
        Returns:
            MTLComputePipelineState for the kernel.
        """
        if self._library is None:
            self.compile_shaders()
        
        # Create cache key
        cache_key = function_name
        if constants:
            cache_key += "_" + "_".join(f"{k}={v}" for k, v in sorted(constants.items()))
        
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]
        
        # Get function
        if constants:
            constant_values = Metal.MTLFunctionConstantValues.new()
            for name, value in constants.items():
                if isinstance(value, int):
                    constant_values.setConstantValue_type_atIndex_(
                        value.to_bytes(4, 'little'),
                        Metal.MTLDataTypeUInt,
                        # Index would need to be looked up - this is simplified
                        0
                    )
            function = self._library.newFunctionWithName_constantValues_error_(
                function_name, constant_values, None
            )[0]
        else:
            function = self._library.newFunctionWithName_(function_name)
        
        if function is None:
            available = self._library.functionNames()
            raise RuntimeError(
                f"Function '{function_name}' not found. Available: {list(available)}"
            )
        
        # Create pipeline
        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for {function_name}: {error}")
        
        self._pipelines[cache_key] = pipeline
        return pipeline
    
    def create_buffer(self, size: int, options: int = 0) -> Any:
        """Create a new Metal buffer.
        
        Args:
            size: Size in bytes.
            options: MTLResourceOptions flags.
            
        Returns:
            MTLBuffer.
        """
        if options == 0:
            # Default to shared storage for unified memory
            options = Metal.MTLResourceStorageModeShared
        return self._device.newBufferWithLength_options_(size, options)
    
    def create_buffer_from_bytes(self, data: bytes, options: int = 0) -> Any:
        """Create a Metal buffer initialized with data.
        
        Args:
            data: Bytes to initialize buffer with.
            options: MTLResourceOptions flags.
            
        Returns:
            MTLBuffer.
        """
        if options == 0:
            options = Metal.MTLResourceStorageModeShared
        return self._device.newBufferWithBytes_length_options_(data, len(data), options)
    
    def synchronize(self) -> None:
        """Wait for all submitted work to complete."""
        command_buffer = self._command_queue.commandBuffer()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
