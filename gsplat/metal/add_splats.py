"""
Post-hoc splat addition from camera images.

This module provides functionality to add new Gaussian splats to an existing scene
based on a camera image and optional depth map.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def depth_to_points(
    depths: Tensor,
    camtoworld: Tensor,
    K: Tensor,
    z_depth: bool = True
) -> Tensor:
    """Convert depth map to 3D points in world coordinates.
    
    Args:
        depths: Depth map [H, W, 1] or [H, W]
        camtoworld: Camera-to-world transformation matrix [4, 4]
        K: Camera intrinsics matrix [3, 3]
        z_depth: Whether depth is z-depth (True) or ray depth (False)
        
    Returns:
        points: 3D points in world coordinates [H, W, 3]
    """
    if depths.ndim == 2:
        depths = depths.unsqueeze(-1)  # [H, W, 1]
    
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    assert camtoworld.shape == (4, 4), f"Invalid camtoworld shape: {camtoworld.shape}"
    assert K.shape == (3, 3), f"Invalid K shape: {K.shape}"
    
    device = depths.device
    height, width = depths.shape[:2]
    
    # Create pixel grid
    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    # Camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / fx,
                (y - cy + 0.5) / fy,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H, W, 3]
    
    # Handle z-depth vs ray depth
    if z_depth:
        # Scale directions by depth / z-component
        directions = camera_dirs * (depths / camera_dirs[..., 2:3])
    else:
        # Ray depth
        directions = camera_dirs * depths
    
    # Transform to world coordinates
    # directions is in camera space, transform to world
    rotation = camtoworld[:3, :3]  # [3, 3]
    translation = camtoworld[:3, 3]  # [3]
    
    # points_cam = directions
    # points_world = R @ points_cam + t
    directions_flat = directions.reshape(-1, 3)  # [H*W, 3]
    points_world_flat = (rotation @ directions_flat.T).T + translation  # [H*W, 3]
    points_world = points_world_flat.reshape(height, width, 3)  # [H, W, 3]
    
    return points_world


def add_splats_from_image(
    params: Dict[str, Tensor],
    image: Tensor,
    depth: Optional[Tensor],
    camera_quat: Tensor,
    camera_position: Tensor,
    fov_degrees: float,
    width: int,
    height: int,
    downsample_factor: int = 4,
    depth_scale: float = 1.0,
    initial_opacity: float = 0.1,
    initial_scale: float = 0.01,
) -> Dict[str, Tensor]:
    """Add new Gaussian splats to a scene from a camera image.
    
    This function creates new Gaussians based on an observed image and optionally
    a depth map. The new Gaussians are added to the existing scene parameters.
    
    Args:
        params: Existing scene parameters dict with keys:
            - "means": [N, 3] existing Gaussian positions
            - "quats": [N, 4] existing quaternions (wxyz)
            - "scales": [N, 3] existing log scales
            - "opacities": [N] existing logit opacities
            Optional:
            - "sh0": [N, 1, 3] existing SH coefficients
            - "shN": [N, K, 3] existing higher-order SH
            - "colors": [N, 3] existing colors (if not using SH)
        image: RGB image [H, W, 3] in range [0, 1]
        depth: Optional depth map [H, W] or [H, W, 1]. If None, uses fixed depth.
        camera_quat: Camera rotation quaternion [4] in wxyz format
        camera_position: Camera position [3] in world coordinates
        fov_degrees: Vertical field of view in degrees
        width: Image width
        height: Image height
        downsample_factor: Factor to downsample image for splat placement (default: 4)
        depth_scale: Scale factor for depth values (default: 1.0)
        initial_opacity: Initial opacity for new Gaussians in [0, 1] (default: 0.1)
        initial_scale: Initial scale for new Gaussians (default: 0.01)
        
    Returns:
        Updated params dict with new Gaussians appended
        
    Example:
        >>> # Existing scene
        >>> params = {
        ...     "means": torch.randn(100, 3),
        ...     "quats": torch.randn(100, 4),
        ...     "scales": torch.ones(100, 3) * -2,
        ...     "opacities": torch.ones(100) * 2,
        ... }
        >>> # New observation
        >>> image = torch.rand(480, 640, 3)  # RGB image
        >>> depth = torch.rand(480, 640) * 10  # Depth map
        >>> camera_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        >>> camera_pos = torch.tensor([0.0, 0.0, -5.0])
        >>> # Add new splats
        >>> params_updated = add_splats_from_image(
        ...     params, image, depth, camera_quat, camera_pos, fov_degrees=60,
        ...     width=640, height=480
        ... )
        >>> print(params_updated["means"].shape)  # More Gaussians than before
    """
    import numpy as np
    
    device = image.device
    
    # Validate inputs
    assert image.shape == (height, width, 3), f"Image shape mismatch: {image.shape}"
    if depth is not None:
        if depth.ndim == 2:
            depth = depth.unsqueeze(-1)
        assert depth.shape[:2] == (height, width), f"Depth shape mismatch: {depth.shape}"
    assert camera_quat.shape == (4,), f"Camera quat shape: {camera_quat.shape}"
    assert camera_position.shape == (3,), f"Camera position shape: {camera_position.shape}"
    
    # Build camera-to-world matrix
    camtoworld = quat_pos_to_matrix(camera_quat, camera_position)
    
    # Build camera intrinsics from FOV
    fov_rad = np.deg2rad(fov_degrees)
    focal_length = height / (2.0 * np.tan(fov_rad / 2.0))
    K = torch.tensor([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Downsample image and depth for efficiency
    if downsample_factor > 1:
        h_down = height // downsample_factor
        w_down = width // downsample_factor
        
        # Downsample image
        image_down = F.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),  # [1, 3, H, W]
            size=(h_down, w_down),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # [H', W', 3]
        
        # Downsample depth
        if depth is not None:
            depth_down = F.interpolate(
                depth.permute(2, 0, 1).unsqueeze(0),  # [1, 1, H, W]
                size=(h_down, w_down),
                mode='nearest'  # Use nearest for depth
            ).squeeze(0).squeeze(0)  # [H', W']
        else:
            depth_down = None
        
        # Adjust intrinsics
        K_down = K.clone()
        K_down[0, 0] /= downsample_factor  # fx
        K_down[1, 1] /= downsample_factor  # fy
        K_down[0, 2] /= downsample_factor  # cx
        K_down[1, 2] /= downsample_factor  # cy
    else:
        image_down = image
        depth_down = depth
        K_down = K
        h_down, w_down = height, width
    
    # Use default depth if not provided
    if depth_down is None:
        depth_down = torch.ones(h_down, w_down, device=device) * 5.0  # Default 5 units
    else:
        depth_down = depth_down * depth_scale
    
    # Convert depth to 3D points
    points_3d = depth_to_points(depth_down, camtoworld, K_down, z_depth=True)  # [H', W', 3]
    
    # Flatten to get new Gaussian positions
    new_means = points_3d.reshape(-1, 3)  # [H'*W', 3]
    n_new = new_means.shape[0]
    
    # Extract colors from image
    colors_new = image_down.reshape(-1, 3)  # [H'*W', 3]
    
    # Initialize new Gaussian parameters
    # Quaternions: all identity (aligned with camera)
    new_quats = camera_quat.unsqueeze(0).repeat(n_new, 1)  # [H'*W', 4]
    
    # Scales: isotropic, small initial size
    new_scales = torch.ones(n_new, 3, device=device) * np.log(initial_scale)
    
    # Opacities: low initial opacity in logit space
    new_opacities = torch.ones(n_new, device=device) * torch.logit(
        torch.tensor(initial_opacity, device=device)
    )
    
    # Handle SH coefficients or direct colors
    if "sh0" in params:
        # Use SH representation
        from gsplat.utils import rgb_to_sh
        
        # Convert RGB to SH
        sh0_new = rgb_to_sh(colors_new).unsqueeze(1)  # [H'*W', 1, 3]
        
        # Initialize higher-order SH to zero
        if "shN" in params:
            K_sh = params["shN"].shape[1]
            shN_new = torch.zeros(n_new, K_sh, 3, device=device)
        else:
            shN_new = None
    else:
        # Use direct RGB colors in logit space
        sh0_new = None
        shN_new = None
        colors_logit_new = torch.logit(colors_new.clamp(0.01, 0.99))
    
    # Concatenate with existing parameters
    params_updated = {}
    params_updated["means"] = torch.cat([params["means"], new_means], dim=0)
    params_updated["quats"] = torch.cat([params["quats"], new_quats], dim=0)
    params_updated["scales"] = torch.cat([params["scales"], new_scales], dim=0)
    params_updated["opacities"] = torch.cat([params["opacities"], new_opacities], dim=0)
    
    if sh0_new is not None:
        params_updated["sh0"] = torch.cat([params["sh0"], sh0_new], dim=0)
        if shN_new is not None and "shN" in params:
            params_updated["shN"] = torch.cat([params["shN"], shN_new], dim=0)
    
    if "colors" in params:
        params_updated["colors"] = torch.cat([params["colors"], colors_logit_new], dim=0)
    
    # Copy any other parameters
    for key in params:
        if key not in params_updated:
            params_updated[key] = params[key]
    
    return params_updated


def quat_pos_to_matrix(quat: Tensor, position: Tensor) -> Tensor:
    """Convert quaternion and position to 4x4 transformation matrix.
    
    Args:
        quat: Quaternion [4] in wxyz format
        position: Position [3]
        
    Returns:
        matrix: 4x4 transformation matrix
    """
    # Normalize quaternion
    quat = quat / quat.norm()
    w, x, y, z = quat
    
    # Build rotation matrix from quaternion
    R = torch.zeros(3, 3, dtype=quat.dtype, device=quat.device)
    R[0, 0] = 1 - 2 * (y*y + z*z)
    R[0, 1] = 2 * (x*y - w*z)
    R[0, 2] = 2 * (x*z + w*y)
    R[1, 0] = 2 * (x*y + w*z)
    R[1, 1] = 1 - 2 * (x*x + z*z)
    R[1, 2] = 2 * (y*z - w*x)
    R[2, 0] = 2 * (x*z - w*y)
    R[2, 1] = 2 * (y*z + w*x)
    R[2, 2] = 1 - 2 * (x*x + y*y)
    
    # Build 4x4 matrix
    matrix = torch.eye(4, dtype=quat.dtype, device=quat.device)
    matrix[:3, :3] = R
    matrix[:3, 3] = position
    
    return matrix


def estimate_depth_from_rendered(
    params: Dict[str, Tensor],
    camera_quat: Tensor,
    camera_position: Tensor,
    fov_degrees: float,
    width: int,
    height: int,
) -> Tensor:
    """Estimate depth map by rendering existing Gaussians.
    
    This is useful when you have an RGB image but no depth map. This function
    renders the depth from the existing scene to use as initialization for new splats.
    
    Args:
        params: Existing scene parameters
        camera_quat: Camera rotation quaternion [4] (wxyz)
        camera_position: Camera position [3]
        fov_degrees: Vertical field of view
        width: Image width
        height: Image height
        
    Returns:
        depth: Rendered depth map [H, W]
    """
    from .rasterizer import render_depth
    import numpy as np
    
    # Build camera matrix
    camtoworld = quat_pos_to_matrix(camera_quat, camera_position)
    viewmat = torch.inverse(camtoworld)
    
    # Build intrinsics
    fov_rad = np.deg2rad(fov_degrees)
    focal_length = height / (2.0 * np.tan(fov_rad / 2.0))
    K = torch.tensor([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=torch.float32, device=params["means"].device)
    
    # Render depth
    depth, _, _ = render_depth(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        viewmat,
        K,
        width,
        height,
    )
    
    return depth
