"""
Tests for Metal backend depth rendering.

These tests verify the Metal backend functionality for depth map rendering.
Tests will be skipped on non-macOS systems or when Metal is not available.
"""

import pytest
import torch
import numpy as np

from gsplat.metal import (
    render_depth,
    is_available,
    load_ply,
    add_splats_from_image,
    depth_to_points,
)


# Skip all tests if Metal is not available
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="Metal backend not available (requires macOS with PyTorch MPS)"
)


def create_test_gaussians(n=10):
    """Create simple test Gaussian parameters."""
    means = torch.randn(n, 3) * 2.0
    quats = torch.randn(n, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.ones(n, 3) * -2.0  # Small scale
    opacities = torch.ones(n) * 2.0  # High opacity
    return means, quats, scales, opacities


def create_test_camera(width=64, height=48):
    """Create simple test camera."""
    viewmat = torch.eye(4)
    viewmat[2, 3] = -10.0  # Move camera back
    
    focal = 50.0
    K = torch.tensor([
        [focal, 0, width / 2.0],
        [0, focal, height / 2.0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return viewmat, K


def test_metal_available():
    """Test that Metal backend reports availability correctly."""
    available = is_available()
    assert isinstance(available, bool)
    
    # On macOS with MPS, should be True
    if torch.backends.mps.is_available():
        assert available, "Metal should be available on macOS with MPS"


def test_render_depth_basic():
    """Test basic depth rendering."""
    means, quats, scales, opacities = create_test_gaussians(n=10)
    viewmat, K = create_test_camera(width=64, height=48)
    
    depth, alpha, meta = render_depth(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmat=viewmat,
        K=K,
        width=64,
        height=48,
    )
    
    # Check output shapes
    assert depth.shape == (48, 64), f"Expected shape (48, 64), got {depth.shape}"
    assert alpha.shape == (48, 64), f"Expected shape (48, 64), got {alpha.shape}"
    assert isinstance(meta, dict)
    
    # Check output types
    assert depth.dtype == torch.float32
    assert alpha.dtype == torch.float32
    
    # Check output ranges
    assert depth.min() >= 0, "Depth should be non-negative"
    assert alpha.min() >= 0 and alpha.max() <= 1, "Alpha should be in [0, 1]"


def test_render_depth_empty():
    """Test rendering with no valid Gaussians."""
    # Create Gaussians all behind camera
    means = torch.ones(5, 3) * 100.0  # Far behind
    means[:, 2] = 100.0  # Positive Z (behind camera)
    quats, scales, opacities = torch.randn(5, 4), torch.ones(5, 3) * -2, torch.ones(5) * 2
    
    viewmat, K = create_test_camera()
    
    depth, alpha, _ = render_depth(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmat=viewmat,
        K=K,
        width=64,
        height=48,
    )
    
    # Should return all zeros
    assert depth.max() == 0, "Depth should be zero with no visible Gaussians"
    assert alpha.max() == 0, "Alpha should be zero with no visible Gaussians"


def test_render_depth_single_gaussian():
    """Test rendering a single Gaussian at origin."""
    means = torch.zeros(1, 3)
    means[0, 2] = 5.0  # 5 units in front of camera
    
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity rotation
    scales = torch.ones(1, 3) * -1.0
    opacities = torch.ones(1) * 5.0  # Very opaque
    
    viewmat, K = create_test_camera(width=64, height=48)
    viewmat[2, 3] = 0.0  # Camera at origin
    
    depth, alpha, _ = render_depth(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmat=viewmat,
        K=K,
        width=64,
        height=48,
    )
    
    # Check that center has some depth
    center_depth = depth[24, 32]  # Approximate center
    assert center_depth > 0, "Center pixel should have positive depth"
    assert center_depth < 10, f"Center depth {center_depth} seems too large"
    
    # Check that center has high alpha
    center_alpha = alpha[24, 32]
    assert center_alpha > 0.5, "Center pixel should have high alpha"


def test_render_depth_different_sizes():
    """Test rendering at different image sizes."""
    means, quats, scales, opacities = create_test_gaussians(n=5)
    
    for width, height in [(32, 24), (64, 48), (128, 96)]:
        viewmat, K = create_test_camera(width=width, height=height)
        
        depth, alpha, _ = render_depth(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
        )
        
        assert depth.shape == (height, width)
        assert alpha.shape == (height, width)


def test_parameter_validation():
    """Test that invalid parameters raise errors."""
    means, quats, scales, opacities = create_test_gaussians(n=5)
    viewmat, K = create_test_camera()
    
    # Wrong shapes should raise AssertionError
    with pytest.raises(AssertionError):
        render_depth(
            means=means[:, :2],  # Wrong shape
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmat=viewmat,
            K=K,
            width=64,
            height=48,
        )
    
    with pytest.raises(AssertionError):
        render_depth(
            means=means,
            quats=quats[:, :3],  # Wrong shape
            scales=scales,
            opacities=opacities,
            viewmat=viewmat,
            K=K,
            width=64,
            height=48,
        )


def test_return_alpha_flag():
    """Test the return_alpha flag."""
    means, quats, scales, opacities = create_test_gaussians(n=5)
    viewmat, K = create_test_camera()
    
    # With alpha
    depth, alpha, _ = render_depth(
        means, quats, scales, opacities, viewmat, K, 64, 48,
        return_alpha=True
    )
    assert alpha is not None
    
    # Without alpha
    depth, alpha, _ = render_depth(
        means, quats, scales, opacities, viewmat, K, 64, 48,
        return_alpha=False
    )
    assert alpha is None


def test_device_handling():
    """Test that tensors are moved to MPS device."""
    means, quats, scales, opacities = create_test_gaussians(n=5)
    viewmat, K = create_test_camera()
    
    # Inputs on CPU
    assert means.device.type == "cpu"
    
    # Render (should move to MPS internally)
    depth, alpha, _ = render_depth(
        means, quats, scales, opacities, viewmat, K, 64, 48
    )
    
    # Output should be on MPS
    assert depth.device.type == "mps"
    assert alpha.device.type == "mps"


def test_load_ply_not_implemented():
    """Test that load_ply raises appropriate error for missing file."""
    with pytest.raises((FileNotFoundError, IOError)):
        load_ply("nonexistent_file.ply")


def test_depth_to_points():
    """Test depth to 3D points conversion."""
    # Simple test case
    depth = torch.ones(10, 10) * 5.0  # 5 units depth
    
    # Identity camera at origin
    camtoworld = torch.eye(4)
    
    # Simple intrinsics
    K = torch.tensor([
        [100, 0, 5],
        [0, 100, 5],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    points = depth_to_points(depth, camtoworld, K, z_depth=True)
    
    assert points.shape == (10, 10, 3)
    
    # Center pixel should be roughly at (0, 0, 5)
    center_point = points[5, 5]
    assert abs(center_point[2].item() - 5.0) < 0.1, "Z coordinate should be ~5"


def test_add_splats_from_image_basic():
    """Test adding splats from an image."""
    # Create initial scene
    params = {
        "means": torch.randn(10, 3),
        "quats": torch.randn(10, 4),
        "scales": torch.ones(10, 3) * -2,
        "opacities": torch.ones(10) * 2,
        "colors": torch.randn(10, 3),
    }
    
    # Create test image and depth
    height, width = 32, 48
    image = torch.rand(height, width, 3)
    depth = torch.ones(height, width) * 5.0
    
    # Camera parameters
    camera_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    camera_pos = torch.tensor([0.0, 0.0, -10.0])
    
    # Add splats
    params_updated = add_splats_from_image(
        params, image, depth, camera_quat, camera_pos,
        fov_degrees=60, width=width, height=height,
        downsample_factor=4
    )
    
    # Check that Gaussians were added
    n_original = params["means"].shape[0]
    n_updated = params_updated["means"].shape[0]
    assert n_updated > n_original, "Should have added new Gaussians"
    
    # Check shapes are consistent
    assert params_updated["means"].shape[0] == n_updated
    assert params_updated["quats"].shape == (n_updated, 4)
    assert params_updated["scales"].shape == (n_updated, 3)
    assert params_updated["opacities"].shape == (n_updated,)


def test_add_splats_without_depth():
    """Test adding splats without providing depth (uses default)."""
    params = {
        "means": torch.randn(5, 3),
        "quats": torch.randn(5, 4),
        "scales": torch.ones(5, 3) * -2,
        "opacities": torch.ones(5) * 2,
        "colors": torch.randn(5, 3),
    }
    
    image = torch.rand(24, 32, 3)
    camera_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    camera_pos = torch.tensor([0.0, 0.0, 0.0])
    
    # Add without depth
    params_updated = add_splats_from_image(
        params, image, None, camera_quat, camera_pos,
        fov_degrees=60, width=32, height=24,
        downsample_factor=4
    )
    
    assert params_updated["means"].shape[0] > params["means"].shape[0]


def test_add_splats_different_downsample():
    """Test different downsample factors."""
    params = {
        "means": torch.randn(5, 3),
        "quats": torch.randn(5, 4),
        "scales": torch.ones(5, 3) * -2,
        "opacities": torch.ones(5) * 2,
        "colors": torch.randn(5, 3),
    }
    
    image = torch.rand(64, 64, 3)
    depth = torch.ones(64, 64) * 3.0
    camera_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    camera_pos = torch.tensor([0.0, 0.0, 0.0])
    
    n_original = params["means"].shape[0]
    
    # Test different downsample factors
    for factor in [2, 4, 8]:
        params_updated = add_splats_from_image(
            params, image, depth, camera_quat, camera_pos,
            fov_degrees=60, width=64, height=64,
            downsample_factor=factor
        )
        
        n_added = params_updated["means"].shape[0] - n_original
        expected_added = (64 // factor) * (64 // factor)
        assert n_added == expected_added, f"With factor {factor}, added {n_added}, expected {expected_added}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
