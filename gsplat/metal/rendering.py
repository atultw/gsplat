"""High-level rendering API for Metal backend.

Provides the main rasterization_metal() function that matches
the interface of gsplat.rendering.rasterization().
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal

from .backend import MetalDevice, is_metal_available
from .wrapper import (
    fully_fused_projection_metal,
    isect_tiles_metal,
    isect_offset_encode_metal,
    rasterize_to_pixels_metal,
    spherical_harmonics_metal,
)
from .utils import validate_render_inputs, pad_channels
from .buffers import ensure_contiguous_float32


def frustum_cull_gaussians(
    means: Tensor,      # [N, 3]
    viewmats: Tensor,   # [C, 4, 4]
    Ks: Tensor,         # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    margin: float = 0.2,  # Margin as fraction of image size
) -> Tensor:
    """Fast CPU frustum culling to filter Gaussians before Metal processing.
    
    Args:
        means: Gaussian centers [N, 3]
        viewmats: View matrices [C, 4, 4]
        Ks: Camera intrinsics [C, 3, 3]
        width, height: Image dimensions
        near_plane, far_plane: Clipping planes
        margin: Extra margin around image bounds (fraction)
        
    Returns:
        valid_mask: Boolean tensor [N] of Gaussians to keep
    """
    # Use first camera for culling (assume single camera for now)
    viewmat = viewmats[0].numpy() if viewmats.ndim == 3 else viewmats[0, 0].numpy()
    K = Ks[0].numpy() if Ks.ndim == 3 else Ks[0, 0].numpy()
    means_np = means.numpy() if means.ndim == 2 else means[0].numpy()
    
    N = means_np.shape[0]
    
    # Transform to camera space: P_cam = viewmat @ [P_world, 1]
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    
    # means_cam = means_np @ R.T + t
    means_cam = means_np @ R.T + t
    
    # Depth check (z > near_plane and z < far_plane)
    z = means_cam[:, 2]
    valid_depth = (z > near_plane) & (z < far_plane)
    
    # Project to image plane for frustum check
    # Only for points with valid depth
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Avoid division by zero
    z_safe = np.maximum(z, 0.001)
    
    x_img = means_cam[:, 0] * fx / z_safe + cx
    y_img = means_cam[:, 1] * fy / z_safe + cy
    
    # Check if within image bounds (with margin)
    margin_x = width * margin
    margin_y = height * margin
    
    valid_x = (x_img > -margin_x) & (x_img < width + margin_x)
    valid_y = (y_img > -margin_y) & (y_img < height + margin_y)
    
    # Combine all checks
    valid_mask = valid_depth & valid_x & valid_y
    
    return torch.from_numpy(valid_mask)


def rasterization_metal(
    means: Tensor,                      # [N, 3]
    quats: Tensor,                      # [N, 4]
    scales: Tensor,                     # [N, 3]
    opacities: Tensor,                  # [N]
    colors: Tensor,                     # [N, D] or [N, K, 3] for SH
    viewmats: Tensor,                   # [C, 4, 4]
    Ks: Tensor,                         # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED", "D_ALL"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize 3D Gaussians to images using Metal GPU backend.
    
    This function provides the same interface as gsplat.rendering.rasterization()
    but uses Apple Metal for GPU acceleration instead of CUDA.
    
    Args:
        means: Gaussian centers in world space. [N, 3]
        quats: Quaternions (wxyz convention). [N, 4]
        scales: Gaussian scales. [N, 3]
        opacities: Gaussian opacities in [0, 1]. [N]
        colors: Colors [N, D] or SH coefficients [N, K, 3].
        viewmats: World-to-camera matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width in pixels.
        height: Image height in pixels.
        near_plane: Near clipping plane distance.
        far_plane: Far clipping plane distance.
        radius_clip: Minimum 2D radius to render.
        eps2d: Blur epsilon for numerical stability.
        sh_degree: If set, interpret colors as SH coefficients.
        tile_size: Tile size for rasterization (default 16).
        backgrounds: Background colors. [C, D]
        render_mode: Rendering mode. Options:
            - "RGB": Render colors
            - "D": Accumulated weighted depth (sum of w_i * z_i)
            - "ED": Expected depth (D / alpha)
            - "RGB+D", "RGB+ED": Combined color and depth
            - "D_ALL": Returns [D, ED, VIS] where VIS is the original Gaussian
              index of the last contributing Gaussian per pixel (-1 for background)
        rasterize_mode: "classic" or "antialiased".
        
    Returns:
        render_colors: Rendered images. [C, H, W, D] (D=3 for D_ALL: [D, ED, VIS])
        render_alphas: Rendered alpha masks. [C, H, W, 1]
        meta: Dictionary with intermediate results. For D_ALL mode, includes
              'visible_indices' mapping filtered to original Gaussian indices.
        
    Example:
        >>> means = torch.randn(10000, 3, device="cpu")
        >>> quats = torch.randn(10000, 4, device="cpu")
        >>> scales = torch.rand(10000, 3, device="cpu") * 0.1
        >>> opacities = torch.rand(10000, device="cpu")
        >>> colors = torch.rand(10000, 3, device="cpu")
        >>> viewmats = torch.eye(4, device="cpu").unsqueeze(0)
        >>> Ks = torch.tensor([[300, 0, 150], [0, 300, 100], [0, 0, 1]], 
        ...                   dtype=torch.float32, device="cpu").unsqueeze(0)
        >>> renders, alphas, meta = rasterization_metal(
        ...     means, quats, scales, opacities, colors,
        ...     viewmats, Ks, width=300, height=200
        ... )
    """
    if not is_metal_available():
        raise RuntimeError(
            "Metal is not available. This function requires macOS with "
            "Apple Silicon or AMD GPU."
        )
    
    # Validate render mode
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED", "D_ALL"], \
        f"Invalid render_mode: {render_mode}"
    
    meta = {}
    
    # Get dimensions
    N_orig = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    
    # Frustum culling - filter out Gaussians outside camera view
    # This is a major optimization when most Gaussians are off-screen
    visible_mask = frustum_cull_gaussians(
        means, viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane, margin=0.3
    )
    
    # Apply mask to all inputs
    visible_indices = torch.where(visible_mask)[0]
    N = len(visible_indices)
    
    if N == 0:
        # No visible Gaussians - return empty render
        channels = 3 if render_mode == "RGB" else (4 if render_mode in ["RGB+D", "RGB+ED"] else 1)
        return (
            torch.zeros(C, height, width, channels, device=device),
            torch.zeros(C, height, width, 1, device=device),
            {"visible_count": 0, "total_count": N_orig}
        )
    
    means = means[visible_indices]
    quats = quats[visible_indices]
    scales = scales[visible_indices]
    opacities = opacities[visible_indices]
    colors = colors[visible_indices] if colors.ndim == 2 else colors[visible_indices]
    
    meta["visible_count"] = N
    meta["total_count"] = N_orig
    
    # Ensure batch dimension
    if means.ndim == 2:
        means = means.unsqueeze(0)  # [1, N, 3]
    if quats.ndim == 2:
        quats = quats.unsqueeze(0)
    if scales.ndim == 2:
        scales = scales.unsqueeze(0)
    if opacities.ndim == 1:
        opacities = opacities.unsqueeze(0)
    if viewmats.ndim == 3:
        viewmats = viewmats.unsqueeze(0)  # [1, C, 4, 4]
    if Ks.ndim == 3:
        Ks = Ks.unsqueeze(0)
    
    B = means.shape[0]  # Batch size
    
    # Process colors/SH
    if sh_degree is not None:
        # Colors are SH coefficients [N, K, 3] or [B, N, K, 3]
        if colors.ndim == 3:
            colors = colors.unsqueeze(0)
        
        # Compute view directions
        campos = torch.inverse(viewmats)[..., :3, 3]  # [B, C, 3]
        
        # For each camera, compute view-dependent colors
        rendered_colors_list = []
        for c in range(C):
            cam_pos = campos[0, c]  # [3]
            dirs = means[0] - cam_pos  # [N, 3]
            
            # Evaluate SH
            cam_colors = spherical_harmonics_metal(
                sh_degree, dirs, colors[0], masks=None
            )
            # Apply the +0.5 and clamp like CUDA version
            cam_colors = torch.clamp(cam_colors + 0.5, min=0.0)
            rendered_colors_list.append(cam_colors)
        
        # Stack colors per camera [C, N, 3]
        colors = torch.stack(rendered_colors_list, dim=0).unsqueeze(0)  # [1, C, N, 3]
    else:
        # Colors are already post-activation [N, D]
        if colors.ndim == 2:
            colors = colors.unsqueeze(0)  # [1, N, D]
        if colors.ndim == 3:
            # Broadcast to all cameras
            colors = colors.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, N, D]
    
    # Project Gaussians to 2D
    calc_compensations = (rasterize_mode == "antialiased")
    
    radii, means2d, depths, conics, compensations = fully_fused_projection_metal(
        means,
        None,  # covars
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=calc_compensations,
        opacities=opacities,
    )
    
    # Apply compensation to opacities
    if compensations is not None:
        opacities = opacities.unsqueeze(1) * compensations  # [B, C, N]
    else:
        opacities = opacities.unsqueeze(1).expand(-1, C, -1)  # [B, N] -> [B, 1, N] -> [B, C, N]
    
    meta.update({
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
    })
    
    # Handle render modes
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat([colors, depths.unsqueeze(-1)], dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat([
                backgrounds,
                torch.zeros(C, 1, device=device)
            ], dim=-1)
    elif render_mode in ["D", "ED", "D_ALL"]:
        colors = depths.unsqueeze(-1)
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=device)
    
    # Tile intersection
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    
    # Flatten for tile intersection
    I = B * C
    radii_flat = radii.reshape(I, N, 2)
    means2d_flat = means2d.reshape(I, N, 2)
    depths_flat = depths.reshape(I, N)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_metal(
        means2d_flat,
        radii_flat,
        depths_flat,
        tile_size,
        tile_width,
        tile_height,
        n_images=I,
    )
    
    isect_offsets = isect_offset_encode_metal(
        isect_ids, I, tile_width, tile_height
    )
    
    meta.update({
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
    })
    
    # Flatten colors and opacities for rasterization
    colors_flat = colors.reshape(I, N, -1)
    opacities_flat = opacities.reshape(I, N)
    conics_flat = conics.reshape(I, N, 3)
    
    # Handle channel padding
    channels = colors_flat.shape[-1]
    colors_flat, orig_channels = pad_channels(colors_flat)
    if backgrounds is not None:
        backgrounds, _ = pad_channels(backgrounds)
    
    # Rasterize - request last_ids for D_ALL mode
    need_last_ids = (render_mode == "D_ALL")
    render_colors, render_alphas, last_ids = rasterize_to_pixels_metal(
        means2d_flat.reshape(-1, 2),  # Flatten to [I*N, 2]
        conics_flat.reshape(-1, 3),
        colors_flat.reshape(-1, colors_flat.shape[-1]),
        opacities_flat.reshape(-1),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        return_last_ids=need_last_ids,
    )
    
    # Trim padding
    if colors_flat.shape[-1] != orig_channels:
        render_colors = render_colors[..., :orig_channels]
    
    # Reshape outputs
    render_colors = render_colors.reshape(B, C, height, width, -1)
    render_alphas = render_alphas.reshape(B, C, height, width, 1)
    
    # Handle D_ALL mode: return [D, ED, VIS] stacked
    if render_mode == "D_ALL":
        D = render_colors[..., 0]  # Accumulated weighted depth
        alpha_safe = render_alphas[..., 0].clamp(min=1e-10)
        ED = D / alpha_safe  # Expected depth
        
        # Remap last_ids to original Gaussian indices
        # last_ids contains indices into the sorted isect array
        # We need: isect_idx -> flatten_ids[isect_idx] -> filtered_idx -> visible_indices[filtered_idx] -> original_idx
        last_ids = last_ids.reshape(B, C, height, width)
        
        # Create output tensor for original indices (-1 for background/invalid)
        VIS = torch.full_like(last_ids, -1, dtype=torch.int64)
        
        # Valid pixels have last_ids >= 0 and < n_isects
        n_isects = flatten_ids.shape[0]
        valid_mask = (last_ids >= 0) & (last_ids < n_isects)
        
        if valid_mask.any():
            # Get filtered Gaussian index from flatten_ids
            valid_isect_ids = last_ids[valid_mask].long()
            filtered_gaussian_ids = flatten_ids[valid_isect_ids].long()
            
            # Map filtered indices to original indices via visible_indices
            original_gaussian_ids = visible_indices[filtered_gaussian_ids]
            VIS[valid_mask] = original_gaussian_ids.long()
        
        # Stack [D, ED, VIS] as the output
        render_colors = torch.stack([D, ED, VIS.float()], dim=-1)
        
        # Store visible_indices in meta for reference
        meta["visible_indices"] = visible_indices
    
    # Handle expected depth normalization for ED and RGB+ED modes
    elif render_mode in ["ED", "RGB+ED"]:
        # ED = sum(w_i * d_i) / sum(w_i)
        # render_colors already has weighted depth, divide by alpha
        depth_idx = -1 if render_mode == "ED" else colors.shape[-1] - 1
        alpha_safe = render_alphas.clamp(min=1e-10)
        if render_mode == "ED":
            render_colors = render_colors / alpha_safe
        else:
            render_colors[..., depth_idx:] = render_colors[..., depth_idx:] / alpha_safe
    
    # Remove batch dim if input didn't have it
    if B == 1:
        render_colors = render_colors.squeeze(0)
        render_alphas = render_alphas.squeeze(0)
    
    return render_colors, render_alphas, meta

