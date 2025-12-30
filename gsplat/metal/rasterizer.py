"""Main rasterization interface for Metal backend."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .backend import get_backend, is_available


def render_depth(
    means: Tensor,  # [N, 3] - 3D positions
    quats: Tensor,  # [N, 4] - Rotation quaternions (wxyz)
    scales: Tensor,  # [N, 3] - Scale factors (log scale)
    opacities: Tensor,  # [N] - Opacities (logit scale)
    viewmat: Tensor,  # [4, 4] - World-to-camera transformation
    K: Tensor,  # [3, 3] - Camera intrinsics
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    background: Optional[Tensor] = None,
    return_alpha: bool = True,
) -> Tuple[Tensor, Optional[Tensor], Dict]:
    """Render depth map from Gaussian splats using Metal backend.
    
    Args:
        means: Gaussian centers in world coordinates. Shape [N, 3]
        quats: Rotation quaternions in wxyz format (not required to be normalized). Shape [N, 4]
        scales: Log-space scale factors. Shape [N, 3]
        opacities: Logit-space opacities. Shape [N]
        viewmat: World-to-camera transformation matrix. Shape [4, 4]
        K: Camera intrinsics matrix. Shape [3, 3]
        width: Output image width in pixels
        height: Output image height in pixels
        near_plane: Near clipping plane. Default 0.01
        far_plane: Far clipping plane. Default 1e10
        background: Background depth value. If None, uses 0.0. Shape [] or [1]
        return_alpha: Whether to return alpha channel. Default True
    
    Returns:
        depth: Rendered depth map. Shape [height, width]
        alpha: Rendered alpha channel (if return_alpha=True). Shape [height, width]
        meta: Dictionary with metadata (empty for now)
    
    Example:
        >>> # Assume we have Gaussian parameters in a dict
        >>> params = {
        ...     "means": torch.randn(1000, 3),
        ...     "quats": torch.randn(1000, 4),
        ...     "scales": torch.randn(1000, 3),
        ...     "opacities": torch.randn(1000),
        ... }
        >>> # Camera parameters
        >>> viewmat = torch.eye(4)
        >>> K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=torch.float32)
        >>> # Render depth
        >>> depth, alpha, meta = render_depth(
        ...     params["means"], params["quats"], params["scales"], params["opacities"],
        ...     viewmat, K, 640, 480
        ... )
    """
    if not is_available():
        raise RuntimeError(
            "Metal backend is not available. "
            "This requires macOS with PyTorch MPS support."
        )
    
    # Input validation
    N = means.shape[0]
    assert means.shape == (N, 3), f"means must be shape [N, 3], got {means.shape}"
    assert quats.shape == (N, 4), f"quats must be shape [N, 4], got {quats.shape}"
    assert scales.shape == (N, 3), f"scales must be shape [N, 3], got {scales.shape}"
    assert opacities.shape == (N,), f"opacities must be shape [N], got {opacities.shape}"
    assert viewmat.shape == (4, 4), f"viewmat must be shape [4, 4], got {viewmat.shape}"
    assert K.shape == (3, 3), f"K must be shape [3, 3], got {K.shape}"
    
    # Convert to MPS device if not already
    device = torch.device("mps")
    means = means.to(device)
    quats = quats.to(device)
    scales = scales.to(device)
    opacities = opacities.to(device)
    viewmat = viewmat.to(device)
    K = K.to(device)
    
    # Convert opacities from logit to probability space
    opacities_prob = torch.sigmoid(opacities)
    
    # For now, fall back to PyTorch implementation
    # This is a placeholder - the actual Metal kernels would be called here
    depth, alpha = _render_depth_pytorch(
        means, quats, scales, opacities_prob,
        viewmat, K, width, height,
        near_plane, far_plane, background
    )
    
    meta = {}
    
    if return_alpha:
        return depth, alpha, meta
    else:
        return depth, None, meta


def _render_depth_pytorch(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    background: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """PyTorch fallback implementation for depth rendering.
    
    This is a simplified implementation that doesn't match the full performance
    of the Metal shaders, but provides correct results for testing.
    """
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
    )
    
    # Project Gaussians to 2D
    # Convert scales from log to linear space
    scales_exp = torch.exp(scales)
    
    # Compute covariances
    covars, _ = _quat_scale_to_covar_preci(
        quats, scales_exp, compute_covar=True, compute_preci=False, triu=False
    )
    
    # Add batch dimensions for compatibility
    means_b = means.unsqueeze(0)  # [1, N, 3]
    covars_b = covars.unsqueeze(0)  # [1, N, 3, 3]
    viewmats_b = viewmat.unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
    Ks_b = K.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
    
    # Project
    radii, means2d, depths, conics, _ = _fully_fused_projection(
        means_b, covars_b, viewmats_b, Ks_b,
        width, height,
        eps2d=0.3,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=False
    )
    
    # Remove batch dimensions
    radii = radii.squeeze(0).squeeze(0)  # [N, 2]
    means2d = means2d.squeeze(0).squeeze(0)  # [N, 2]
    depths = depths.squeeze(0).squeeze(0)  # [N]
    conics = conics.squeeze(0).squeeze(0)  # [N, 3]
    
    # Simple rasterization (naive, not tiled)
    depth_image = torch.zeros((height, width), device=means.device)
    alpha_image = torch.zeros((height, width), device=means.device)
    
    # Create pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=means.device),
        torch.arange(width, device=means.device),
        indexing='ij'
    )
    pixels = torch.stack([x_grid + 0.5, y_grid + 0.5], dim=-1)  # [H, W, 2]
    
    # Sort by depth (front to back)
    valid_mask = (radii > 0).any(dim=-1)
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return depth_image, alpha_image
    
    valid_depths = depths[valid_indices]
    sorted_indices = torch.argsort(valid_depths)
    sorted_valid_indices = valid_indices[sorted_indices]
    
    # Accumulate depth for each pixel
    T = torch.ones((height, width), device=means.device)  # Transmittance
    
    for idx in sorted_valid_indices:
        if T.max() < 0.001:
            break
        
        mean2d = means2d[idx]
        conic = conics[idx]
        depth = depths[idx]
        opacity = opacities[idx]
        
        # Compute Gaussian response at all pixels
        delta = pixels - mean2d  # [H, W, 2]
        sigma = -0.5 * (
            conic[0] * delta[..., 0] * delta[..., 0] +
            2 * conic[1] * delta[..., 0] * delta[..., 1] +
            conic[2] * delta[..., 1] * delta[..., 1]
        )  # [H, W]
        
        # Compute alpha
        alpha = torch.clamp(opacity * torch.exp(sigma), max=0.99)
        alpha = torch.where(sigma > 0, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha < 1.0/255.0, torch.zeros_like(alpha), alpha)
        
        # Accumulate
        weight = alpha * T
        depth_image = depth_image + depth * weight
        alpha_image = alpha_image + weight
        
        # Update transmittance
        T = T * (1.0 - alpha)
    
    return depth_image, alpha_image


def load_ply(ply_path: str) -> Dict[str, Tensor]:
    """Load Gaussian splat parameters from a PLY file.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Dictionary with keys: "means", "quats", "scales", "opacities", "sh0", "shN"
        All tensors are returned on CPU.
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    # Extract positions
    means = np.stack([
        vertices['x'],
        vertices['y'],
        vertices['z']
    ], axis=-1).astype(np.float32)
    
    # Extract opacities
    opacities = vertices['opacity'].astype(np.float32)
    
    # Extract scales (usually stored as log scale)
    scales = np.stack([
        vertices['scale_0'],
        vertices['scale_1'],
        vertices['scale_2']
    ], axis=-1).astype(np.float32)
    
    # Extract quaternions (usually stored as rot_0, rot_1, rot_2, rot_3)
    quats = np.stack([
        vertices['rot_0'],  # w
        vertices['rot_1'],  # x
        vertices['rot_2'],  # y
        vertices['rot_3']   # z
    ], axis=-1).astype(np.float32)
    
    # Extract SH coefficients (optional, for color rendering)
    sh0 = None
    shN = None
    try:
        # DC component (first SH band)
        sh0 = np.stack([
            vertices['f_dc_0'],
            vertices['f_dc_1'],
            vertices['f_dc_2']
        ], axis=-1).astype(np.float32)
        sh0 = sh0[:, None, :]  # Add K dimension
        
        # Higher order SH coefficients
        sh_keys = [k for k in vertices.data.dtype.names if k.startswith('f_rest_')]
        if sh_keys:
            sh_rest = np.stack([vertices[k] for k in sh_keys], axis=-1).astype(np.float32)
            # Reshape to [N, K-1, 3]
            num_sh_rest = len(sh_keys) // 3
            shN = sh_rest.reshape(-1, num_sh_rest, 3)
    except (KeyError, ValueError):
        pass
    
    # Convert to PyTorch tensors
    result = {
        "means": torch.from_numpy(means),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(scales),
        "opacities": torch.from_numpy(opacities),
    }
    
    if sh0 is not None:
        result["sh0"] = torch.from_numpy(sh0)
    if shN is not None:
        result["shN"] = torch.from_numpy(shN)
    
    return result
