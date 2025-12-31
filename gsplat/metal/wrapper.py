"""Low-level kernel wrappers for Metal backend.

Provides Python functions that dispatch Metal compute kernels
matching the interface of gsplat.cuda._wrapper functions.
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from .backend import MetalDevice, is_metal_available
from .buffers import tensor_to_metal_buffer, metal_buffer_to_tensor, ensure_contiguous_float32

# Try to import Numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if Numba isn't available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _generate_tile_intersections_numba(
    valid_indices: np.ndarray,        # [M] indices of valid Gaussians
    tile_min_x: np.ndarray,           # [I*N] flattened
    tile_min_y: np.ndarray,           # [I*N] flattened
    tile_max_x: np.ndarray,           # [I*N] flattened  
    tile_max_y: np.ndarray,           # [I*N] flattened
    depth_bits_all: np.ndarray,       # [I*N] uint32
    tile_width: int,
    tile_n_bits: int,
    N: int,
    I: int,
    n_isects: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-JIT compiled function to generate tile intersection IDs.
    
    This is the hot inner loop - generates isect_ids and flatten_ids.
    """
    isect_ids = np.zeros(n_isects, dtype=np.int64)
    flatten_ids = np.zeros(n_isects, dtype=np.int32)
    
    cur_idx = 0
    
    for idx in valid_indices:
        i = idx // N
        g = idx % N
        
        tmin_x = tile_min_x[idx]
        tmin_y = tile_min_y[idx]
        tmax_x = tile_max_x[idx]
        tmax_y = tile_max_y[idx]
        
        depth_bits = np.int64(depth_bits_all[idx])
        iid_enc = np.int64(i) << (32 + tile_n_bits)
        flat_id = i * N + g if I > 1 else g
        
        for ty in range(tmin_y, tmax_y):
            for tx in range(tmin_x, tmax_x):
                tile_id = ty * tile_width + tx
                isect_ids[cur_idx] = iid_enc | (np.int64(tile_id) << 32) | depth_bits
                flatten_ids[cur_idx] = flat_id
                cur_idx += 1
    
    return isect_ids, flatten_ids


def fully_fused_projection_metal(
    means: Tensor,                      # [B, N, 3]
    covars: Optional[Tensor],           # [B, N, 6] or None
    quats: Optional[Tensor],            # [B, N, 4] or None
    scales: Optional[Tensor],           # [B, N, 3] or None
    viewmats: Tensor,                   # [B, C, 4, 4]
    Ks: Tensor,                         # [B, C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    opacities: Optional[Tensor] = None, # [B, N]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Project 3D Gaussians to 2D using Metal GPU.
    
    Args:
        means: Gaussian means in world space [B, N, 3]
        covars: Optional covariance matrices [B, N, 6]
        quats: Optional quaternions [B, N, 4]
        scales: Optional scales [B, N, 3]
        viewmats: Camera view matrices [B, C, 4, 4]
        Ks: Camera intrinsics [B, C, 3, 3]
        width: Image width
        height: Image height
        eps2d: Blur epsilon for numerical stability
        near_plane: Near clipping plane
        far_plane: Far clipping plane
        radius_clip: Minimum radius to keep
        calc_compensations: Whether to compute compensation terms
        opacities: Optional opacities for tighter bounds [B, N]
        
    Returns:
        radii: Bounding radii [B, C, N, 2]
        means2d: 2D means [B, C, N, 2]
        depths: Z depths [B, C, N]
        conics: Inverse covariance (upper tri) [B, C, N, 3]
        compensations: Compensation factors [B, C, N] or None
    """
    if not is_metal_available():
        raise RuntimeError("Metal is not available")
    
    device = MetalDevice()
    device.compile_shaders()
    
    # Validate inputs
    batch_dims = means.shape[:-2]
    B = math.prod(batch_dims) if batch_dims else 1
    N = means.shape[-2]
    C = viewmats.shape[-3]
    
    # Ensure contiguous float32
    means = ensure_contiguous_float32(means)
    viewmats = ensure_contiguous_float32(viewmats)
    Ks = ensure_contiguous_float32(Ks)
    
    if quats is not None:
        quats = ensure_contiguous_float32(quats)
    if scales is not None:
        scales = ensure_contiguous_float32(scales)
    if opacities is not None:
        opacities = ensure_contiguous_float32(opacities)
    
    # Allocate outputs
    out_device = means.device
    radii = torch.zeros((B, C, N, 2), dtype=torch.int32, device=out_device)
    means2d = torch.zeros((B, C, N, 2), dtype=torch.float32, device=out_device)
    depths = torch.zeros((B, C, N), dtype=torch.float32, device=out_device)
    conics = torch.zeros((B, C, N, 3), dtype=torch.float32, device=out_device)
    compensations = None
    if calc_compensations:
        compensations = torch.zeros((B, C, N), dtype=torch.float32, device=out_device)
    
    # Dispatch Metal kernel
    try:
        import Metal
        
        pipeline = device.get_pipeline("projection_ewa_3dgs_fused_fwd")
        
        # Create command buffer
        command_buffer = device.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        
        encoder.setComputePipelineState_(pipeline)
        
        # Set buffers
        means_buf, _ = tensor_to_metal_buffer(means, device)
        quats_buf, _ = tensor_to_metal_buffer(quats, device) if quats is not None else (None, 0)
        scales_buf, _ = tensor_to_metal_buffer(scales, device) if scales is not None else (None, 0)
        opacities_buf, _ = tensor_to_metal_buffer(opacities, device) if opacities is not None else (None, 0)
        viewmats_buf, _ = tensor_to_metal_buffer(viewmats, device)
        Ks_buf, _ = tensor_to_metal_buffer(Ks, device)
        
        radii_buf, _ = tensor_to_metal_buffer(radii, device)
        means2d_buf, _ = tensor_to_metal_buffer(means2d, device)
        depths_buf, _ = tensor_to_metal_buffer(depths, device)
        conics_buf, _ = tensor_to_metal_buffer(conics, device)
        comp_buf = None
        if compensations is not None:
            comp_buf, _ = tensor_to_metal_buffer(compensations, device)
        
        encoder.setBuffer_offset_atIndex_(means_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(quats_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(scales_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(opacities_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(viewmats_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(Ks_buf, 0, 5)
        encoder.setBuffer_offset_atIndex_(radii_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(means2d_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(depths_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(conics_buf, 0, 9)
        if comp_buf:
            encoder.setBuffer_offset_atIndex_(comp_buf, 0, 10)
        
        # Set parameters
        import struct
        params = struct.pack(
            'IIIIIfffff',
            B, C, N, width, height,
            eps2d, near_plane, far_plane, radius_clip, 0.0
        )
        params_buf = device.create_buffer_from_bytes(params)
        encoder.setBuffer_offset_atIndex_(params_buf, 0, 11)
        
        # Dispatch
        n_threads = B * C * N
        threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())
        n_groups = (n_threads + threads_per_group - 1) // threads_per_group
        
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(n_groups, 1, 1),
            Metal.MTLSize(threads_per_group, 1, 1)
        )
        
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Read back results
        radii = metal_buffer_to_tensor(radii_buf, (B, C, N, 2), torch.int32)
        means2d = metal_buffer_to_tensor(means2d_buf, (B, C, N, 2), torch.float32)
        depths = metal_buffer_to_tensor(depths_buf, (B, C, N), torch.float32)
        conics = metal_buffer_to_tensor(conics_buf, (B, C, N, 3), torch.float32)
        if comp_buf:
            compensations = metal_buffer_to_tensor(comp_buf, (B, C, N), torch.float32)
        
    except Exception as e:
        raise RuntimeError(f"Metal projection kernel failed: {e}")
    
    # Reshape to match input batch dims
    if batch_dims:
        radii = radii.reshape(batch_dims + (C, N, 2))
        means2d = means2d.reshape(batch_dims + (C, N, 2))
        depths = depths.reshape(batch_dims + (C, N))
        conics = conics.reshape(batch_dims + (C, N, 3))
        if compensations is not None:
            compensations = compensations.reshape(batch_dims + (C, N))
    
    return radii, means2d, depths, conics, compensations


@torch.no_grad()
def isect_tiles_metal(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    n_images: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Find tile intersections for Gaussians using vectorized NumPy.
    
    Returns:
        tiles_per_gauss: Number of tiles per Gaussian
        isect_ids: Intersection IDs for sorting
        flatten_ids: Flattened indices
    """
    import numpy as np
    import struct
    
    if not is_metal_available():
        raise RuntimeError("Metal is not available")
    
    # Convert to numpy for vectorized operations
    means2d_np = means2d.detach().cpu().numpy()
    radii_np = radii.detach().cpu().numpy()
    depths_np = depths.detach().cpu().numpy()
    
    # Handle shapes - flatten to [I, N, ...] format
    if means2d_np.ndim == 2:
        means2d_np = means2d_np[np.newaxis, ...]  # [1, N, 2]
        radii_np = radii_np[np.newaxis, ...]      # [1, N, 2]
        depths_np = depths_np[np.newaxis, ...]    # [1, N]
    
    I = means2d_np.shape[0]
    N = means2d_np.shape[1]
    
    # Vectorized tile bounds computation for all Gaussians at once
    # Shape: [I, N]
    mx = means2d_np[:, :, 0]
    my = means2d_np[:, :, 1]
    rx = radii_np[:, :, 0]
    ry = radii_np[:, :, 1]
    
    # Compute tile coordinates
    tile_x = mx / tile_size
    tile_y = my / tile_size
    tile_rx = rx / tile_size
    tile_ry = ry / tile_size
    
    # Compute tile bounds
    tile_min_x = np.maximum(0, np.floor(tile_x - tile_rx).astype(np.int32))
    tile_min_y = np.maximum(0, np.floor(tile_y - tile_ry).astype(np.int32))
    tile_max_x = np.minimum(tile_width, np.ceil(tile_x + tile_rx).astype(np.int32))
    tile_max_y = np.minimum(tile_height, np.ceil(tile_y + tile_ry).astype(np.int32))
    
    # Valid Gaussians have positive radii and at least one tile
    valid = (rx > 0) & (ry > 0) & (tile_max_x > tile_min_x) & (tile_max_y > tile_min_y)
    
    # Tiles per Gaussian
    tiles_per_gauss_np = np.where(
        valid,
        (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x),
        0
    ).astype(np.int32)
    
    tiles_per_gauss = torch.from_numpy(tiles_per_gauss_np)
    if means2d.ndim == 2:
        tiles_per_gauss = tiles_per_gauss.squeeze(0)
    
    n_isects = int(tiles_per_gauss_np.sum())
    
    if n_isects == 0:
        isect_ids = torch.empty(0, dtype=torch.int64, device=means2d.device)
        flatten_ids = torch.empty(0, dtype=torch.int32, device=means2d.device)
        return tiles_per_gauss, isect_ids, flatten_ids
    
    n_tiles = tile_width * tile_height
    tile_n_bits = int(math.floor(math.log2(n_tiles))) + 1
    
    # Get indices of valid Gaussians
    valid_indices = np.where(valid.flatten())[0].astype(np.int64)
    
    # Prepare depth bits for all Gaussians (vectorized)
    depths_flat = depths_np.flatten().astype(np.float32)
    depth_bits_all = np.frombuffer(depths_flat.tobytes(), dtype=np.uint32)
    
    # Flatten tile bounds for Numba
    tile_min_x_flat = tile_min_x.flatten().astype(np.int32)
    tile_min_y_flat = tile_min_y.flatten().astype(np.int32)
    tile_max_x_flat = tile_max_x.flatten().astype(np.int32)
    tile_max_y_flat = tile_max_y.flatten().astype(np.int32)
    
    # Use Numba JIT-compiled function for the hot loop
    isect_ids_np, flatten_ids_np = _generate_tile_intersections_numba(
        valid_indices,
        tile_min_x_flat,
        tile_min_y_flat,
        tile_max_x_flat,
        tile_max_y_flat,
        depth_bits_all,
        tile_width,
        tile_n_bits,
        N,
        I,
        n_isects,
    )
    
    # Sort by isect_ids
    sorted_idx = np.argsort(isect_ids_np)
    isect_ids_np = isect_ids_np[sorted_idx]
    flatten_ids_np = flatten_ids_np[sorted_idx]
    
    isect_ids = torch.from_numpy(isect_ids_np.copy())
    flatten_ids = torch.from_numpy(flatten_ids_np.copy())
    
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode_metal(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Encode intersection offsets from sorted IDs using vectorized NumPy.
    
    Returns:
        offsets: Tile offsets [n_images, tile_height, tile_width]
    """
    import numpy as np
    
    n_tiles = tile_width * tile_height
    tile_n_bits = int(math.floor(math.log2(n_tiles))) + 1
    
    total_tiles = n_images * n_tiles
    
    if isect_ids.numel() == 0:
        return torch.zeros(n_images, tile_height, tile_width, dtype=torch.int32, device=isect_ids.device)
    
    # Convert to numpy for vectorized ops
    isect_ids_np = isect_ids.detach().cpu().numpy()
    n_isects = len(isect_ids_np)
    
    # Extract the upper bits (image + tile ID) from all isect_ids
    # isect_id format: [image_id (high bits)] | [tile_id (middle)] | [depth (low 32 bits)]
    upper_bits = (isect_ids_np >> 32).astype(np.int64)
    
    # Extract image ID and tile ID
    iids = upper_bits >> tile_n_bits
    tids = upper_bits & ((1 << tile_n_bits) - 1)
    flat_ids = iids * n_tiles + tids
    
    # Find unique flat_ids and their first occurrence (since sorted, this gives us offsets)
    # For each tile index from 0 to total_tiles-1, find the first isect that belongs to it or later
    all_tile_indices = np.arange(total_tiles, dtype=np.int64)
    
    # searchsorted gives us the first index where flat_ids >= all_tile_indices
    offsets_np = np.searchsorted(flat_ids, all_tile_indices, side='left').astype(np.int32)
    
    # Reshape to [n_images, tile_height, tile_width]
    offsets = torch.from_numpy(offsets_np.copy()).reshape(n_images, tile_height, tile_width)
    
    return offsets


def rasterize_to_pixels_metal(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    return_last_ids: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Rasterize Gaussians to pixels using Metal.
    
    Args:
        return_last_ids: If True, also return the last contributing Gaussian index per pixel.
    
    Returns:
        render_colors: Rendered colors [I, H, W, C]
        render_alphas: Rendered alphas [I, H, W]
        last_ids: (Optional) Last contributing Gaussian index per pixel [I, H, W], 
                  indices into flatten_ids array. Only returned if return_last_ids=True.
    """
    if not is_metal_available():
        raise RuntimeError("Metal is not available")
    
    # Get dimensions
    if means2d.ndim == 2:
        # Packed mode
        I = isect_offsets.shape[0]
        N = 0  # Not used in packed mode
        channels = colors.shape[-1]
    else:
        I = means2d.shape[0]
        N = means2d.shape[1]
        channels = colors.shape[-1]
    
    # Allocate outputs
    render_colors = torch.zeros(I, image_height, image_width, channels, 
                                 dtype=torch.float32, device=means2d.device)
    render_alphas = torch.zeros(I, image_height, image_width,
                                 dtype=torch.float32, device=means2d.device)
    
    device = MetalDevice()
    device.compile_shaders()
    
    try:
        import Metal
        
        # Choose kernel based on channel count
        if channels == 3:
            kernel_name = "rasterize_to_pixels_fwd_rgb"
        elif channels == 1:
            kernel_name = "rasterize_to_pixels_fwd_depth"
        else:
            kernel_name = "rasterize_to_pixels_fwd"
        
        pipeline = device.get_pipeline(kernel_name)
        
        command_buffer = device.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        
        # Set buffers
        means2d = ensure_contiguous_float32(means2d)
        conics = ensure_contiguous_float32(conics)
        colors = ensure_contiguous_float32(colors)
        opacities = ensure_contiguous_float32(opacities)
        
        means2d_buf, _ = tensor_to_metal_buffer(means2d, device)
        conics_buf, _ = tensor_to_metal_buffer(conics, device)
        colors_buf, _ = tensor_to_metal_buffer(colors, device)
        opacities_buf, _ = tensor_to_metal_buffer(opacities, device)
        backgrounds_buf = None
        if backgrounds is not None:
            backgrounds = ensure_contiguous_float32(backgrounds)
            backgrounds_buf, _ = tensor_to_metal_buffer(backgrounds, device)
        
        offsets_buf, _ = tensor_to_metal_buffer(isect_offsets.contiguous().int(), device)
        flatten_buf, _ = tensor_to_metal_buffer(flatten_ids.contiguous().int(), device)
        
        render_colors_buf, _ = tensor_to_metal_buffer(render_colors, device)
        render_alphas_buf, _ = tensor_to_metal_buffer(render_alphas, device)
        
        # Initialize last_ids with -1
        last_ids_init = torch.full((I, image_height, image_width), -1, 
                                   dtype=torch.int32, device=means2d.device)
        last_ids_buf, _ = tensor_to_metal_buffer(last_ids_init, device)
        
        encoder.setBuffer_offset_atIndex_(means2d_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(conics_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(colors_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(opacities_buf, 0, 3)
        encoder.setBuffer_offset_atIndex_(backgrounds_buf, 0, 4)
        encoder.setBuffer_offset_atIndex_(None, 0, 5)  # masks
        encoder.setBuffer_offset_atIndex_(offsets_buf, 0, 6)
        encoder.setBuffer_offset_atIndex_(flatten_buf, 0, 7)
        encoder.setBuffer_offset_atIndex_(render_colors_buf, 0, 8)
        encoder.setBuffer_offset_atIndex_(render_alphas_buf, 0, 9)
        encoder.setBuffer_offset_atIndex_(last_ids_buf, 0, 10)
        
        # Parameters
        tile_width = math.ceil(image_width / tile_size)
        tile_height = math.ceil(image_height / tile_size)
        n_isects = flatten_ids.shape[0]
        
        import struct
        params = struct.pack(
            'IIIIIIIIII',
            I, N, n_isects, image_width, image_height,
            tile_size, tile_width, tile_height, channels, 
            1 if means2d.ndim == 2 else 0  # packed
        )
        params_buf = device.create_buffer_from_bytes(params)
        encoder.setBuffer_offset_atIndex_(params_buf, 0, 11)
        
        # Dispatch - one threadgroup per tile
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(I, tile_height, tile_width),
            Metal.MTLSize(tile_size, tile_size, 1)
        )
        
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Read back
        render_colors = metal_buffer_to_tensor(
            render_colors_buf, (I, image_height, image_width, channels), torch.float32
        )
        render_alphas = metal_buffer_to_tensor(
            render_alphas_buf, (I, image_height, image_width), torch.float32
        )
        
        # Optionally read back last_ids (index of last contributing Gaussian per pixel)
        last_ids = None
        if return_last_ids:
            last_ids = metal_buffer_to_tensor(
                last_ids_buf, (I, image_height, image_width), torch.int32
            )
        
    except Exception as e:
        raise RuntimeError(f"Metal rasterization failed: {e}")
    
    return render_colors, render_alphas, last_ids


def spherical_harmonics_metal(
    degrees_to_use: int,
    dirs: Tensor,
    coeffs: Tensor,
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Evaluate spherical harmonics using Metal.
    
    Args:
        degrees_to_use: SH degree (0-4)
        dirs: View directions [..., 3]
        coeffs: SH coefficients [..., K, 3]
        masks: Optional mask [...,]
        
    Returns:
        colors: Computed colors [..., 3]
    """
    if not is_metal_available():
        raise RuntimeError("Metal is not available")
    
    N = dirs.numel() // 3
    K = coeffs.shape[-2]
    
    device = MetalDevice()
    device.compile_shaders()
    
    dirs = ensure_contiguous_float32(dirs)
    coeffs = ensure_contiguous_float32(coeffs)
    
    colors = torch.zeros(N, 3, dtype=torch.float32, device=dirs.device)
    
    try:
        import Metal
        
        pipeline = device.get_pipeline("spherical_harmonics_fwd")
        
        command_buffer = device.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        
        dirs_buf, _ = tensor_to_metal_buffer(dirs.reshape(-1, 3), device)
        coeffs_buf, _ = tensor_to_metal_buffer(coeffs.reshape(-1, K, 3), device)
        masks_buf = None
        if masks is not None:
            masks_buf, _ = tensor_to_metal_buffer(masks.flatten().bool(), device)
        colors_buf, _ = tensor_to_metal_buffer(colors, device)
        
        encoder.setBuffer_offset_atIndex_(dirs_buf, 0, 0)
        encoder.setBuffer_offset_atIndex_(coeffs_buf, 0, 1)
        encoder.setBuffer_offset_atIndex_(masks_buf, 0, 2)
        encoder.setBuffer_offset_atIndex_(colors_buf, 0, 3)
        
        import struct
        params = struct.pack('III', N, K, degrees_to_use)
        params_buf = device.create_buffer_from_bytes(params)
        encoder.setBuffer_offset_atIndex_(params_buf, 0, 4)
        
        threads_per_group = min(256, pipeline.maxTotalThreadsPerThreadgroup())
        n_groups = (N + threads_per_group - 1) // threads_per_group
        
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(n_groups, 1, 1),
            Metal.MTLSize(threads_per_group, 1, 1)
        )
        
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        colors = metal_buffer_to_tensor(colors_buf, (N, 3), torch.float32)
        
    except Exception as e:
        raise RuntimeError(f"Metal SH evaluation failed: {e}")
    
    return colors.reshape(dirs.shape[:-1] + (3,))
