// Spherical harmonics evaluation kernel for Metal
// Port of SphericalHarmonicsCUDA.cu

#include <metal_stdlib>
#include "common.h"

using namespace metal;

// ============================================
// Spherical Harmonics Evaluation
// ============================================

/// Evaluate spherical harmonics up to degree 4
/// Based on "Efficient Spherical Harmonic Evaluation" by Peter-Pike Sloan
inline float3 sh_coeffs_to_color(
    uint degree,
    float3 dir,
    device const float* coeffs,  // [K, 3] - K SH coefficients per color
    uint K
) {
    // Normalize direction
    float inv_norm = rsqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-10f);
    float x = dir.x * inv_norm;
    float y = dir.y * inv_norm;
    float z = dir.z * inv_norm;
    
    float3 result = float3(0.0f);
    
    // L=0
    result += 0.2820947917738781f * float3(coeffs[0], coeffs[1], coeffs[2]);
    
    if (degree < 1 || K < 4) return result;
    
    // L=1
    result += 0.48860251190292f * float3(
        -y * coeffs[3] + z * coeffs[6] - x * coeffs[9],
        -y * coeffs[4] + z * coeffs[7] - x * coeffs[10],
        -y * coeffs[5] + z * coeffs[8] - x * coeffs[11]
    );
    
    if (degree < 2 || K < 9) return result;
    
    // L=2
    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;
    
    result += float3(
        pSH4 * coeffs[12] + pSH5 * coeffs[15] + pSH6 * coeffs[18] + pSH7 * coeffs[21] + pSH8 * coeffs[24],
        pSH4 * coeffs[13] + pSH5 * coeffs[16] + pSH6 * coeffs[19] + pSH7 * coeffs[22] + pSH8 * coeffs[25],
        pSH4 * coeffs[14] + pSH5 * coeffs[17] + pSH6 * coeffs[20] + pSH7 * coeffs[23] + pSH8 * coeffs[26]
    );
    
    if (degree < 3 || K < 16) return result;
    
    // L=3
    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    
    result += float3(
        pSH9 * coeffs[27] + pSH10 * coeffs[30] + pSH11 * coeffs[33] + pSH12 * coeffs[36] +
        pSH13 * coeffs[39] + pSH14 * coeffs[42] + pSH15 * coeffs[45],
        pSH9 * coeffs[28] + pSH10 * coeffs[31] + pSH11 * coeffs[34] + pSH12 * coeffs[37] +
        pSH13 * coeffs[40] + pSH14 * coeffs[43] + pSH15 * coeffs[46],
        pSH9 * coeffs[29] + pSH10 * coeffs[32] + pSH11 * coeffs[35] + pSH12 * coeffs[38] +
        pSH13 * coeffs[41] + pSH14 * coeffs[44] + pSH15 * coeffs[47]
    );
    
    if (degree < 4 || K < 25) return result;
    
    // L=4
    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = 1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    
    result += float3(
        pSH16 * coeffs[48] + pSH17 * coeffs[51] + pSH18 * coeffs[54] + pSH19 * coeffs[57] +
        pSH20 * coeffs[60] + pSH21 * coeffs[63] + pSH22 * coeffs[66] + pSH23 * coeffs[69] + pSH24 * coeffs[72],
        pSH16 * coeffs[49] + pSH17 * coeffs[52] + pSH18 * coeffs[55] + pSH19 * coeffs[58] +
        pSH20 * coeffs[61] + pSH21 * coeffs[64] + pSH22 * coeffs[67] + pSH23 * coeffs[70] + pSH24 * coeffs[73],
        pSH16 * coeffs[50] + pSH17 * coeffs[53] + pSH18 * coeffs[56] + pSH19 * coeffs[59] +
        pSH20 * coeffs[62] + pSH21 * coeffs[65] + pSH22 * coeffs[68] + pSH23 * coeffs[71] + pSH24 * coeffs[74]
    );
    
    return result;
}

// ============================================
// SH Evaluation Kernel
// ============================================

struct SHParams {
    uint N;             // Number of Gaussians
    uint K;             // Number of SH coefficients
    uint degree;        // SH degree to use
};

/// Compute view-dependent colors from SH coefficients
kernel void spherical_harmonics_fwd(
    device const float* dirs       [[buffer(0)]],   // [N, 3] - view directions
    device const float* coeffs     [[buffer(1)]],   // [N, K, 3] - SH coefficients
    device const bool* masks       [[buffer(2)]],   // [N] - optional mask
    device float* colors           [[buffer(3)]],   // [N, 3] - output colors
    constant SHParams& params      [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.N) {
        return;
    }
    
    // Check mask
    if (masks != nullptr && !masks[idx]) {
        colors[idx * 3] = 0.0f;
        colors[idx * 3 + 1] = 0.0f;
        colors[idx * 3 + 2] = 0.0f;
        return;
    }
    
    // Read direction
    float3 dir = float3(dirs[idx * 3], dirs[idx * 3 + 1], dirs[idx * 3 + 2]);
    
    // Evaluate SH
    device const float* sh_coeffs = coeffs + idx * params.K * 3;
    float3 color = sh_coeffs_to_color(params.degree, dir, sh_coeffs, params.K);
    
    // Write output
    colors[idx * 3] = color.x;
    colors[idx * 3 + 1] = color.y;
    colors[idx * 3 + 2] = color.z;
}

/// Compute view directions from Gaussian means and camera positions
kernel void compute_view_dirs(
    device const float* means      [[buffer(0)]],   // [N, 3] - Gaussian positions
    device const float* campos     [[buffer(1)]],   // [C, 3] - Camera positions
    device const int* batch_ids    [[buffer(2)]],   // [nnz] - batch indices
    device const int* camera_ids   [[buffer(3)]],   // [nnz] - camera indices
    device const int* gaussian_ids [[buffer(4)]],   // [nnz] - Gaussian indices
    device float* dirs             [[buffer(5)]],   // [nnz, 3] - output directions
    constant uint& N               [[buffer(6)]],
    constant uint& C               [[buffer(7)]],
    constant uint& nnz             [[buffer(8)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= nnz) {
        return;
    }
    
    int bid = batch_ids[idx];
    int cid = camera_ids[idx];
    int gid = gaussian_ids[idx];
    
    // Compute Gaussian position offset
    uint mean_idx = bid * N * 3 + gid * 3;
    float3 pos = float3(means[mean_idx], means[mean_idx + 1], means[mean_idx + 2]);
    
    // Compute camera position offset
    uint cam_idx = bid * C * 3 + cid * 3;
    float3 cam = float3(campos[cam_idx], campos[cam_idx + 1], campos[cam_idx + 2]);
    
    // View direction = position - camera
    float3 dir = pos - cam;
    
    dirs[idx * 3] = dir.x;
    dirs[idx * 3 + 1] = dir.y;
    dirs[idx * 3 + 2] = dir.z;
}
