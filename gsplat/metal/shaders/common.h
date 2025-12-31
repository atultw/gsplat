// Common header for gsplat Metal shaders
// Defines types, constants, and utility functions

#pragma once

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// ============================================
// Type Aliases (matching CUDA implementation)
// ============================================

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
typedef float2x2 mat2;
typedef float3x3 mat3;
typedef float4x4 mat4;

// ============================================
// Constants
// ============================================

constant float ALPHA_THRESHOLD = 1.0f / 255.0f;
constant uint TILE_SIZE = 16;
constant float PI = 3.14159265359f;

// ============================================
// Quaternion Operations
// ============================================

/// Convert quaternion (wxyz) to rotation matrix
inline mat3 quat_to_rotmat(float4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    
    // Normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    
    // Note: Metal uses column-major matrices
    return mat3(
        float3(1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy)),
        float3(2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx)),
        float3(2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2))
    );
}

/// Compute covariance matrix from quaternion and scale
inline mat3 quat_scale_to_covar(float4 quat, float3 scale) {
    mat3 R = quat_to_rotmat(quat);
    mat3 S = mat3(
        float3(scale[0], 0.f, 0.f),
        float3(0.f, scale[1], 0.f),
        float3(0.f, 0.f, scale[2])
    );
    mat3 M = R * S;
    return M * transpose(M);
}

// ============================================
// Coordinate Transformations
// ============================================

/// Transform position from world to camera coordinates
inline float3 pos_world_to_cam(mat3 R, float3 t, float3 p_world) {
    return R * p_world + t;
}

/// Transform covariance from world to camera coordinates
inline mat3 covar_world_to_cam(mat3 R, mat3 covar_world) {
    return R * covar_world * transpose(R);
}

// ============================================
// Projection Operations
// ============================================

/// Perspective projection of 3D Gaussian to 2D
/// Returns: (mean2d, cov2d)
inline void persp_proj(
    float3 mean_c,      // Mean in camera space
    mat3 cov_c,         // Covariance in camera space
    float fx, float fy, // Focal lengths
    float cx, float cy, // Principal point
    uint width, uint height,
    thread mat2& cov2d,  // Output 2D covariance
    thread float2& mean2d // Output 2D mean
) {
    float x = mean_c[0], y = mean_c[1], z = mean_c[2];
    
    float tan_fovx = 0.5f * float(width) / fx;
    float tan_fovy = 0.5f * float(height) / fy;
    float lim_x_pos = (float(width) - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (float(height) - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;
    
    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));
    
    // Jacobian of projection (3x2 in row-major, but we use 2x3 transpose for Metal)
    // J = [fx/z,    0,   -fx*tx/z^2]
    //     [0,    fy/z,   -fy*ty/z^2]
    float3 J_row0 = float3(fx * rz, 0.f, -fx * tx * rz2);
    float3 J_row1 = float3(0.f, fy * rz, -fy * ty * rz2);
    
    // cov2d = J * cov_c * J^T
    // Since Metal is column-major, we compute carefully
    float3 Jc0 = cov_c * J_row0;  // cov_c * J^T column 0
    float3 Jc1 = cov_c * J_row1;  // cov_c * J^T column 1
    
    cov2d = mat2(
        float2(dot(J_row0, Jc0), dot(J_row1, Jc0)),
        float2(dot(J_row0, Jc1), dot(J_row1, Jc1))
    );
    
    mean2d = float2(fx * x * rz + cx, fy * y * rz + cy);
}

/// Add blur to 2D covariance and compute compensation factor
inline float add_blur(float eps2d, thread mat2& covar, thread float& compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

/// Compute inverse of 2x2 matrix
inline mat2 inverse2x2(mat2 M) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    float inv_det = 1.f / det;
    return mat2(
        float2(M[1][1] * inv_det, -M[0][1] * inv_det),
        float2(-M[1][0] * inv_det, M[0][0] * inv_det)
    );
}

// ============================================
// Gaussian Evaluation
// ============================================

/// Evaluate 2D Gaussian at a point
/// Returns alpha contribution
inline float eval_gaussian_2d(
    float2 xy,          // Pixel position
    float2 mean2d,      // Gaussian mean
    float3 conic,       // Inverse covariance (upper triangle: a, b, c)
    float opacity       // Gaussian opacity
) {
    float2 delta = xy - mean2d;
    float sigma = 0.5f * (conic.x * delta.x * delta.x +
                          conic.z * delta.y * delta.y) +
                  conic.y * delta.x * delta.y;
    
    if (sigma < 0.f) {
        return 0.f;
    }
    
    float alpha = min(0.999f, opacity * exp(-sigma));
    return alpha;
}

// ============================================
// Utility Functions
// ============================================

/// Compute number of bits needed to represent a value
inline uint bit_width(uint x) {
    return x == 0 ? 1 : (32 - clz(x));
}
