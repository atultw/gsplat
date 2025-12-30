"""
Simple example demonstrating Metal-based depth rendering from Gaussian splats.

This example shows how to:
1. Load or create Gaussian splat parameters
2. Define a camera viewpoint
3. Render a depth map using the Metal backend

Requirements:
- macOS with Metal support
- PyTorch with MPS backend
"""

import torch
import numpy as np
from gsplat.metal import render_depth, is_available, load_ply


def create_simple_scene():
    """Create a simple scene with a few Gaussian splats."""
    # Create 10 Gaussians in a simple pattern
    N = 10
    
    # Positions: arranged in a line along Z axis
    means = torch.zeros(N, 3)
    means[:, 2] = torch.linspace(-5, 5, N)  # Z coordinates from -5 to 5
    means[:, 0] = torch.randn(N) * 0.5  # Small random X offset
    means[:, 1] = torch.randn(N) * 0.5  # Small random Y offset
    
    # Rotations: random quaternions
    quats = torch.randn(N, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    # Scales: log-space, small splats
    scales = torch.ones(N, 3) * -2.0  # Small scale (log scale)
    
    # Opacities: logit-space, moderately opaque
    opacities = torch.ones(N) * 2.0  # Logit of ~0.88
    
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
    }


def create_camera(fov_degrees=60, width=640, height=480):
    """Create a simple camera looking at the origin."""
    # Camera position: looking from (0, 0, -10) towards origin
    camera_pos = torch.tensor([0.0, 0.0, -10.0])
    
    # View matrix: simple translation (camera looks down +Z in its local frame)
    viewmat = torch.eye(4)
    viewmat[:3, 3] = -camera_pos
    
    # Camera intrinsics
    fov_rad = np.deg2rad(fov_degrees)
    focal_length = width / (2.0 * np.tan(fov_rad / 2.0))
    K = torch.tensor([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return viewmat, K


def main():
    """Main example function."""
    # Check if Metal is available
    if not is_available():
        print("Metal backend is not available.")
        print("This example requires macOS with PyTorch MPS support.")
        return
    
    print("Metal backend is available!")
    
    # Create a simple scene
    print("\nCreating simple scene with 10 Gaussians...")
    params = create_simple_scene()
    print(f"Scene created with {params['means'].shape[0]} Gaussians")
    
    # Create camera
    width, height = 640, 480
    print(f"\nSetting up camera ({width}x{height})...")
    viewmat, K = create_camera(width=width, height=height)
    
    # Render depth map
    print("\nRendering depth map...")
    depth, alpha, meta = render_depth(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"],
        opacities=params["opacities"],
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
    )
    
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    print(f"Alpha map shape: {alpha.shape}")
    print(f"Alpha range: [{alpha.min():.2f}, {alpha.max():.2f}]")
    
    # Optionally save as image
    try:
        from PIL import Image
        
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_img = (depth_normalized.cpu().numpy() * 255).astype(np.uint8)
        
        Image.fromarray(depth_img, mode='L').save("depth_map.png")
        print("\nDepth map saved to 'depth_map.png'")
        
        alpha_img = (alpha.cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(alpha_img, mode='L').save("alpha_map.png")
        print("Alpha map saved to 'alpha_map.png'")
        
    except ImportError:
        print("\nPIL not available, skipping image save")


def example_with_ply():
    """Example loading from PLY file."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_example.py <path_to_ply_file>")
        return
    
    ply_path = sys.argv[1]
    
    print(f"Loading Gaussians from {ply_path}...")
    params = load_ply(ply_path)
    print(f"Loaded {params['means'].shape[0]} Gaussians")
    
    # Create camera
    width, height = 640, 480
    viewmat, K = create_camera(width=width, height=height)
    
    # Render depth
    print("Rendering depth map...")
    depth, alpha, meta = render_depth(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"],
        opacities=params["opacities"],
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
    )
    
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If PLY file provided, load from it
        example_with_ply()
    else:
        # Otherwise run simple example
        main()
