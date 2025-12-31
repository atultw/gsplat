// Gaussian projection kernel for Metal
// Port of ProjectionEWA3DGSFused.cu (forward pass only)

#include <metal_stdlib>
#include "common.h"

using namespace metal;

// ============================================
// Projection Kernel Parameters
// ============================================

struct ProjectionParams {
    uint B;             // Batch size
    uint C;             // Number of cameras
    uint N;             // Number of Gaussians
    uint image_width;
    uint image_height;
    float eps2d;        // Blur epsilon
    float near_plane;
    float far_plane;
    float radius_clip;
};

// ============================================
// Projection Kernel
// ============================================

/// Project 3D Gaussians to 2D for rendering
/// Each thread processes one (batch, camera, gaussian) tuple
kernel void projection_ewa_3dgs_fused_fwd(
    // Inputs
    device const float* means       [[buffer(0)]],  // [B, N, 3]
    device const float* quats       [[buffer(1)]],  // [B, N, 4]
    device const float* scales      [[buffer(2)]],  // [B, N, 3]
    device const float* opacities   [[buffer(3)]],  // [B, N]
    device const float* viewmats    [[buffer(4)]],  // [B, C, 4, 4]
    device const float* Ks          [[buffer(5)]],  // [B, C, 3, 3]
    // Outputs
    device int* radii               [[buffer(6)]],  // [B, C, N, 2]
    device float* means2d           [[buffer(7)]],  // [B, C, N, 2]
    device float* depths            [[buffer(8)]],  // [B, C, N]
    device float* conics            [[buffer(9)]],  // [B, C, N, 3]
    device float* compensations     [[buffer(10)]], // [B, C, N] optional
    // Parameters
    constant ProjectionParams& params [[buffer(11)]],
    // Thread info
    uint idx [[thread_position_in_grid]]
) {
    uint B = params.B;
    uint C = params.C;
    uint N = params.N;
    
    if (idx >= B * C * N) {
        return;
    }
    
    uint bid = idx / (C * N);           // Batch ID
    uint cid = (idx / N) % C;           // Camera ID
    uint gid = idx % N;                 // Gaussian ID
    
    // Read Gaussian parameters
    uint means_offset = bid * N * 3 + gid * 3;
    float3 mean_w = float3(
        means[means_offset],
        means[means_offset + 1],
        means[means_offset + 2]
    );
    
    uint quats_offset = bid * N * 4 + gid * 4;
    float4 quat = float4(
        quats[quats_offset],
        quats[quats_offset + 1],
        quats[quats_offset + 2],
        quats[quats_offset + 3]
    );
    
    uint scales_offset = bid * N * 3 + gid * 3;
    float3 scale = float3(
        scales[scales_offset],
        scales[scales_offset + 1],
        scales[scales_offset + 2]
    );
    
    // Read view matrix (row-major in storage)
    uint viewmat_offset = bid * C * 16 + cid * 16;
    mat3 R = mat3(
        float3(viewmats[viewmat_offset + 0], viewmats[viewmat_offset + 4], viewmats[viewmat_offset + 8]),
        float3(viewmats[viewmat_offset + 1], viewmats[viewmat_offset + 5], viewmats[viewmat_offset + 9]),
        float3(viewmats[viewmat_offset + 2], viewmats[viewmat_offset + 6], viewmats[viewmat_offset + 10])
    );
    float3 t = float3(
        viewmats[viewmat_offset + 3],
        viewmats[viewmat_offset + 7],
        viewmats[viewmat_offset + 11]
    );
    
    // Read intrinsics
    uint K_offset = bid * C * 9 + cid * 9;
    float fx = Ks[K_offset + 0];
    float fy = Ks[K_offset + 4];
    float cx = Ks[K_offset + 2];
    float cy = Ks[K_offset + 5];
    
    // Transform to camera space
    float3 mean_c = pos_world_to_cam(R, t, mean_w);
    
    // Check near/far planes
    if (mean_c.z < params.near_plane || mean_c.z > params.far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    
    // Compute covariance
    mat3 covar_w = quat_scale_to_covar(quat, scale);
    mat3 covar_c = covar_world_to_cam(R, covar_w);
    
    // Project to 2D
    mat2 cov2d;
    float2 mean2d_val;
    persp_proj(
        mean_c, covar_c,
        fx, fy, cx, cy,
        params.image_width, params.image_height,
        cov2d, mean2d_val
    );
    
    // Add blur and compute compensation
    float compensation;
    float det = add_blur(params.eps2d, cov2d, compensation);
    
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    
    // Compute inverse covariance (conics)
    mat2 cov2d_inv = inverse2x2(cov2d);
    
    // Compute bounding radius with opacity-aware extension
    float extend = 3.33f;
    float opacity = opacities[bid * N + gid];
    float effective_opacity = opacity * compensation;
    
    if (effective_opacity < ALPHA_THRESHOLD) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    
    // Opacity-aware bounding box (from the paper)
    extend = min(extend, sqrt(2.0f * log(effective_opacity / ALPHA_THRESHOLD)));
    
    // Compute tight rectangular bounding box
    float radius_x = ceil(extend * sqrt(cov2d[0][0]));
    float radius_y = ceil(extend * sqrt(cov2d[1][1]));
    
    if (radius_x <= params.radius_clip && radius_y <= params.radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    
    // Check if Gaussian is in image bounds
    if (mean2d_val.x + radius_x <= 0 || 
        mean2d_val.x - radius_x >= float(params.image_width) ||
        mean2d_val.y + radius_y <= 0 || 
        mean2d_val.y - radius_y >= float(params.image_height)) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }
    
    // Write outputs
    radii[idx * 2] = int(radius_x);
    radii[idx * 2 + 1] = int(radius_y);
    means2d[idx * 2] = mean2d_val.x;
    means2d[idx * 2 + 1] = mean2d_val.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = cov2d_inv[0][0];
    conics[idx * 3 + 1] = cov2d_inv[0][1];
    conics[idx * 3 + 2] = cov2d_inv[1][1];
    
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}
