"""
Example demonstrating post-hoc addition of Gaussian splats from camera images.

This shows how to:
1. Start with an initial scene
2. Capture a new view (image + depth)
3. Add new splats to the scene from that view
4. Render the updated scene
"""

import torch
import numpy as np
from gsplat.metal import (
    render_depth,
    add_splats_from_image,
    is_available,
)


def create_initial_scene(n=50):
    """Create a simple initial scene with a few Gaussians."""
    # Create Gaussians in a cube
    means = torch.randn(n, 3) * 2.0
    quats = torch.randn(n, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.ones(n, 3) * -2.0  # Small scale
    opacities = torch.ones(n) * 2.0  # High opacity
    
    # Add colors (using direct RGB, not SH for simplicity)
    colors = torch.rand(n, 3)  # Random colors
    colors_logit = torch.logit(colors.clamp(0.01, 0.99))
    
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors_logit,
    }


def create_test_camera(distance=10.0):
    """Create a camera looking at the origin."""
    # Camera position
    camera_pos = torch.tensor([0.0, 0.0, -distance])
    
    # Camera rotation (identity - looking down +Z)
    camera_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # wxyz
    
    return camera_quat, camera_pos


def create_synthetic_image_and_depth(width=128, height=96):
    """Create a synthetic RGB image and depth map for testing."""
    # Create a simple test image (gradient)
    y_grid, x_grid = torch.meshgrid(
        torch.linspace(0, 1, height),
        torch.linspace(0, 1, width),
        indexing='ij'
    )
    
    # RGB gradient image
    image = torch.stack([
        x_grid,  # Red increases left to right
        y_grid,  # Green increases top to bottom
        1 - x_grid,  # Blue decreases left to right
    ], dim=-1)  # [H, W, 3]
    
    # Create a depth map with a gradient (closer in center)
    center_y, center_x = height // 2, width // 2
    y_offset = (torch.arange(height, dtype=torch.float32) - center_y) / center_y
    x_offset = (torch.arange(width, dtype=torch.float32) - center_x) / center_x
    y_grid, x_grid = torch.meshgrid(y_offset, x_offset, indexing='ij')
    
    # Paraboloid depth: closer in center, farther at edges
    depth = 3.0 + 2.0 * (x_grid**2 + y_grid**2)  # [H, W]
    
    return image, depth


def main():
    """Main example."""
    if not is_available():
        print("Metal backend not available. Requires macOS with MPS.")
        return
    
    print("=== Post-hoc Splat Addition Example ===\n")
    
    # Step 1: Create initial scene
    print("Step 1: Creating initial scene with 50 Gaussians...")
    params = create_initial_scene(n=50)
    print(f"  Initial scene has {params['means'].shape[0]} Gaussians")
    
    # Step 2: Define a new camera viewpoint
    print("\nStep 2: Setting up new camera viewpoint...")
    camera_quat, camera_pos = create_test_camera(distance=10.0)
    print(f"  Camera position: {camera_pos.tolist()}")
    print(f"  Camera rotation (wxyz): {camera_quat.tolist()}")
    
    # Step 3: Create synthetic observation (in practice, this would be real data)
    print("\nStep 3: Creating synthetic image and depth map...")
    width, height = 128, 96
    image, depth = create_synthetic_image_and_depth(width, height)
    print(f"  Image shape: {image.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # Step 4: Add new splats from this observation
    print("\nStep 4: Adding splats from image...")
    params_updated = add_splats_from_image(
        params=params,
        image=image,
        depth=depth,
        camera_quat=camera_quat,
        camera_position=camera_pos,
        fov_degrees=60,
        width=width,
        height=height,
        downsample_factor=4,  # Use every 4th pixel
        initial_opacity=0.3,
        initial_scale=0.02,
    )
    
    n_added = params_updated["means"].shape[0] - params["means"].shape[0]
    print(f"  Added {n_added} new Gaussians")
    print(f"  Total Gaussians: {params_updated['means'].shape[0]}")
    
    # Step 5: Render from the same viewpoint to verify
    print("\nStep 5: Rendering depth from updated scene...")
    import numpy as np
    
    # Build camera matrix for rendering
    from gsplat.metal.add_splats import quat_pos_to_matrix
    camtoworld = quat_pos_to_matrix(camera_quat, camera_pos)
    viewmat = torch.inverse(camtoworld)
    
    fov_rad = np.deg2rad(60)
    focal_length = height / (2.0 * np.tan(fov_rad / 2.0))
    K = torch.tensor([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    depth_rendered, alpha_rendered, _ = render_depth(
        params_updated["means"],
        params_updated["quats"],
        params_updated["scales"],
        params_updated["opacities"],
        viewmat,
        K,
        width,
        height,
    )
    
    print(f"  Rendered depth shape: {depth_rendered.shape}")
    print(f"  Rendered depth range: [{depth_rendered.min():.2f}, {depth_rendered.max():.2f}]")
    print(f"  Rendered alpha range: [{alpha_rendered.min():.2f}, {alpha_rendered.max():.2f}]")
    
    # Step 6: Optionally save visualization
    try:
        from PIL import Image as PILImage
        
        # Save original depth
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_img = (depth_vis.numpy() * 255).astype(np.uint8)
        PILImage.fromarray(depth_img, mode='L').save("original_depth.png")
        
        # Save rendered depth
        depth_rend_vis = (depth_rendered - depth_rendered.min()) / (depth_rendered.max() - depth_rendered.min() + 1e-8)
        depth_rend_img = (depth_rend_vis.cpu().numpy() * 255).astype(np.uint8)
        PILImage.fromarray(depth_rend_img, mode='L').save("rendered_depth_updated.png")
        
        # Save alpha
        alpha_img = (alpha_rendered.cpu().numpy() * 255).astype(np.uint8)
        PILImage.fromarray(alpha_img, mode='L').save("alpha_updated.png")
        
        # Save image
        image_img = (image.numpy() * 255).astype(np.uint8)
        PILImage.fromarray(image_img, mode='RGB').save("input_image.png")
        
        print("\nSaved visualizations:")
        print("  - original_depth.png: Input depth map")
        print("  - rendered_depth_updated.png: Rendered depth after adding splats")
        print("  - alpha_updated.png: Alpha channel")
        print("  - input_image.png: Input RGB image")
        
    except ImportError:
        print("\nPIL not available, skipping visualization save")
    
    print("\n=== Example Complete ===")
    print(f"\nSummary:")
    print(f"  - Started with {params['means'].shape[0]} Gaussians")
    print(f"  - Added {n_added} Gaussians from image")
    print(f"  - Final scene has {params_updated['means'].shape[0]} Gaussians")


def example_with_estimated_depth():
    """Example using estimated depth from rendering instead of ground truth."""
    from gsplat.metal import estimate_depth_from_rendered
    
    if not is_available():
        print("Metal backend not available.")
        return
    
    print("\n=== Example: Adding Splats with Estimated Depth ===\n")
    
    # Create initial scene
    params = create_initial_scene(n=100)
    print(f"Initial scene: {params['means'].shape[0]} Gaussians")
    
    # Define camera
    camera_quat, camera_pos = create_test_camera(distance=8.0)
    
    # Create image (but no depth)
    width, height = 128, 96
    image, _ = create_synthetic_image_and_depth(width, height)
    
    # Estimate depth by rendering existing scene
    print("Estimating depth from existing scene...")
    depth_estimated = estimate_depth_from_rendered(
        params, camera_quat, camera_pos, 60, width, height
    )
    print(f"  Estimated depth range: [{depth_estimated.min():.2f}, {depth_estimated.max():.2f}]")
    
    # Add splats using estimated depth
    params_updated = add_splats_from_image(
        params, image, depth_estimated, camera_quat, camera_pos,
        60, width, height, downsample_factor=8
    )
    
    n_added = params_updated["means"].shape[0] - params["means"].shape[0]
    print(f"Added {n_added} Gaussians using estimated depth")
    print(f"Final scene: {params_updated['means'].shape[0]} Gaussians")


if __name__ == "__main__":
    import sys
    
    if "--estimated-depth" in sys.argv:
        example_with_estimated_depth()
    else:
        main()
