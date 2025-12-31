"""Tests for the Metal backend rendering functions.

Usage:
    pytest tests/test_metal_rasterization.py -v

These tests verify:
1. Metal backend availability and basic functionality
2. Numerical correctness compared to CUDA reference (when available)
3. All render modes: RGB, D, ED, RGB+D, RGB+ED
4. Edge cases: empty scene, single Gaussian, many Gaussians
"""

import math
from typing import Optional

import pytest
import torch

# Skip all tests if not on macOS
import platform
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on macOS"
)


def _create_test_gaussians(N: int = 1000, device: str = "cpu"):
    """Create random Gaussians for testing."""
    torch.manual_seed(42)
    
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)  # Normalize
    scales = torch.rand(N, 3, device=device) * 0.1
    opacities = torch.rand(N, device=device)
    colors = torch.rand(N, 3, device=device)
    
    return means, quats, scales, opacities, colors


def _create_test_cameras(C: int = 1, width: int = 200, height: int = 150, device: str = "cpu"):
    """Create test cameras."""
    focal = 200.0
    Ks = torch.tensor([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=torch.float32).expand(C, -1, -1)
    
    viewmats = torch.eye(4, device=device).expand(C, -1, -1)
    
    return viewmats, Ks, width, height


class TestMetalAvailability:
    """Test Metal backend availability detection."""
    
    def test_is_metal_available(self):
        """Test that is_metal_available returns a boolean."""
        from gsplat.metal import is_metal_available
        
        result = is_metal_available()
        assert isinstance(result, bool)
    
    def test_get_metal_device_info(self):
        """Test device info retrieval."""
        from gsplat.metal import get_metal_device_info
        
        info = get_metal_device_info()
        assert isinstance(info, dict)
        assert "available" in info


@pytest.mark.skipif(
    not pytest.importorskip("gsplat.metal").is_metal_available(),
    reason="Metal not available"
)
class TestMetalRasterization:
    """Test Metal rasterization functionality."""
    
    def test_basic_rgb_rendering(self):
        """Test basic RGB rendering."""
        from gsplat.metal import rasterization_metal
        
        means, quats, scales, opacities, colors = _create_test_gaussians(100)
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB"
        )
        
        assert renders.shape == (1, height, width, 3)
        assert alphas.shape == (1, height, width, 1)
        assert isinstance(meta, dict)
    
    def test_depth_rendering(self):
        """Test depth-only rendering."""
        from gsplat.metal import rasterization_metal
        
        means, quats, scales, opacities, colors = _create_test_gaussians(100)
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="D"
        )
        
        assert renders.shape == (1, height, width, 1)
        assert alphas.shape == (1, height, width, 1)
    
    def test_rgb_plus_depth_rendering(self):
        """Test RGB+D rendering."""
        from gsplat.metal import rasterization_metal
        
        means, quats, scales, opacities, colors = _create_test_gaussians(100)
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB+D"
        )
        
        assert renders.shape == (1, height, width, 4)
        assert alphas.shape == (1, height, width, 1)
    
    def test_expected_depth_rendering(self):
        """Test ED (expected depth) rendering."""
        from gsplat.metal import rasterization_metal
        
        means, quats, scales, opacities, colors = _create_test_gaussians(100)
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="ED"
        )
        
        assert renders.shape == (1, height, width, 1)
    
    def test_multiple_cameras(self):
        """Test rendering to multiple cameras."""
        from gsplat.metal import rasterization_metal
        
        means, quats, scales, opacities, colors = _create_test_gaussians(100)
        viewmats, Ks, width, height = _create_test_cameras(C=3)
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB"
        )
        
        assert renders.shape == (3, height, width, 3)
        assert alphas.shape == (3, height, width, 1)
    
    def test_spherical_harmonics(self):
        """Test SH color rendering."""
        from gsplat.metal import rasterization_metal
        
        N = 100
        sh_degree = 3
        K = (sh_degree + 1) ** 2
        
        means, quats, scales, opacities, _ = _create_test_gaussians(N)
        sh_coeffs = torch.randn(N, K, 3)
        
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, sh_coeffs,
            viewmats, Ks, width, height,
            sh_degree=sh_degree,
            render_mode="RGB"
        )
        
        assert renders.shape == (1, height, width, 3)
    
    def test_empty_scene(self):
        """Test rendering with no Gaussians."""
        from gsplat.metal import rasterization_metal
        
        means = torch.empty(0, 3)
        quats = torch.empty(0, 4)
        scales = torch.empty(0, 3)
        opacities = torch.empty(0)
        colors = torch.empty(0, 3)
        
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB"
        )
        
        assert renders.shape == (1, height, width, 3)
        assert alphas.sum() == 0  # No Gaussians = no alpha
    
    def test_single_gaussian(self):
        """Test rendering with single Gaussian."""
        from gsplat.metal import rasterization_metal
        
        means = torch.tensor([[0.0, 0.0, 2.0]])
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1]])
        opacities = torch.tensor([0.9])
        colors = torch.tensor([[1.0, 0.0, 0.0]])  # Red
        
        viewmats, Ks, width, height = _create_test_cameras()
        
        renders, alphas, meta = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB"
        )
        
        assert renders.shape == (1, height, width, 3)
        # Center pixel should have some red color
        center_x, center_y = width // 2, height // 2
        assert renders[0, center_y, center_x, 0] > 0


@pytest.mark.skipif(
    not pytest.importorskip("gsplat.metal").is_metal_available(),
    reason="Metal not available"
)
class TestMetalVsCUDAComparison:
    """Compare Metal output against CUDA reference."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available for comparison"
    )
    def test_numerical_comparison_rgb(self):
        """Compare Metal and CUDA RGB output."""
        from gsplat.metal import rasterization_metal
        from gsplat.rendering import rasterization as rasterization_cuda
        
        # Create test data on CPU (works for both)
        means, quats, scales, opacities, colors = _create_test_gaussians(1000, "cpu")
        viewmats, Ks, width, height = _create_test_cameras(device="cpu")
        
        # Render with Metal
        metal_renders, metal_alphas, _ = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="RGB"
        )
        
        # Render with CUDA
        cuda_means = means.cuda()
        cuda_quats = quats.cuda()
        cuda_scales = scales.cuda()
        cuda_opacities = opacities.cuda()
        cuda_colors = colors.cuda()
        cuda_viewmats = viewmats.cuda()
        cuda_Ks = Ks.cuda()
        
        cuda_renders, cuda_alphas, _ = rasterization_cuda(
            cuda_means, cuda_quats, cuda_scales, cuda_opacities, cuda_colors,
            cuda_viewmats, cuda_Ks, width, height,
            render_mode="RGB"
        )
        
        # Compare (allow some tolerance for float differences)
        torch.testing.assert_close(
            metal_renders.cpu(), cuda_renders.cpu(),
            rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            metal_alphas.cpu(), cuda_alphas.cpu(),
            rtol=1e-3, atol=1e-3
        )
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available for comparison"
    )
    def test_numerical_comparison_depth(self):
        """Compare Metal and CUDA depth output."""
        from gsplat.metal import rasterization_metal
        from gsplat.rendering import rasterization as rasterization_cuda
        
        means, quats, scales, opacities, colors = _create_test_gaussians(1000, "cpu")
        viewmats, Ks, width, height = _create_test_cameras(device="cpu")
        
        # Metal
        metal_renders, metal_alphas, _ = rasterization_metal(
            means, quats, scales, opacities, colors,
            viewmats, Ks, width, height,
            render_mode="D"
        )
        
        # CUDA
        cuda_renders, cuda_alphas, _ = rasterization_cuda(
            means.cuda(), quats.cuda(), scales.cuda(), opacities.cuda(), colors.cuda(),
            viewmats.cuda(), Ks.cuda(), width, height,
            render_mode="D"
        )
        
        torch.testing.assert_close(
            metal_renders.cpu(), cuda_renders.cpu(),
            rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
