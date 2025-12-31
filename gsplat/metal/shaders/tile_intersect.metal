// Tile intersection kernel for Metal
// Port of IntersectTile.cu

#include <metal_stdlib>
#include "common.h"

using namespace metal;

// ============================================
// Tile Intersection Kernels
// ============================================

struct TileIntersectParams {
    uint I;              // Number of images (B * C)
    uint N;              // Number of Gaussians per image
    uint nnz;            // Total Gaussians (for packed mode)
    uint tile_size;
    uint tile_width;
    uint tile_height;
    uint tile_n_bits;
    uint image_n_bits;
    bool packed;
    bool first_pass;     // True for counting, false for writing
};

/// First pass: count tiles per Gaussian
/// Second pass: write intersection IDs
kernel void intersect_tile(
    // Inputs
    device const float* means2d        [[buffer(0)]],  // [I, N, 2] or [nnz, 2]
    device const int* radii            [[buffer(1)]],  // [I, N, 2] or [nnz, 2]
    device const float* depths         [[buffer(2)]],  // [I, N] or [nnz]
    device const long* image_ids       [[buffer(3)]],  // [nnz] (packed only)
    device const long* gaussian_ids    [[buffer(4)]],  // [nnz] (packed only)
    device const long* cum_tiles       [[buffer(5)]],  // [I, N] or [nnz] (second pass)
    // Outputs
    device int* tiles_per_gauss        [[buffer(6)]],  // [I, N] or [nnz] (first pass)
    device long* isect_ids             [[buffer(7)]],  // [n_isects] (second pass)
    device int* flatten_ids            [[buffer(8)]],  // [n_isects] (second pass)
    // Parameters
    constant TileIntersectParams& params [[buffer(9)]],
    // Thread info
    uint idx [[thread_position_in_grid]]
) {
    uint n_elements = params.packed ? params.nnz : params.I * params.N;
    if (idx >= n_elements) {
        return;
    }
    
    float radius_x = float(radii[idx * 2]);
    float radius_y = float(radii[idx * 2 + 1]);
    
    if (radius_x <= 0 || radius_y <= 0) {
        if (params.first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }
    
    float2 mean2d = float2(means2d[idx * 2], means2d[idx * 2 + 1]);
    
    float tile_radius_x = radius_x / float(params.tile_size);
    float tile_radius_y = radius_y / float(params.tile_size);
    float tile_x = mean2d.x / float(params.tile_size);
    float tile_y = mean2d.y / float(params.tile_size);
    
    // Compute tile range (tile_min inclusive, tile_max exclusive)
    uint tile_min_x = min(max(0u, uint(floor(tile_x - tile_radius_x))), params.tile_width);
    uint tile_min_y = min(max(0u, uint(floor(tile_y - tile_radius_y))), params.tile_height);
    uint tile_max_x = min(max(0u, uint(ceil(tile_x + tile_radius_x))), params.tile_width);
    uint tile_max_y = min(max(0u, uint(ceil(tile_y + tile_radius_y))), params.tile_height);
    
    uint tile_count = (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x);
    
    if (params.first_pass) {
        tiles_per_gauss[idx] = int(tile_count);
        return;
    }
    
    // Second pass: write intersection IDs
    long iid; // Image ID
    if (params.packed) {
        iid = image_ids[idx];
    } else {
        iid = long(idx / params.N);
    }
    
    long iid_enc = iid << (32 + params.tile_n_bits);
    
    // Bit-cast depth to int32 for sorting
    float depth_val = depths[idx];
    int depth_i32 = as_type<int>(depth_val);
    long depth_enc = long(uint(depth_i32));
    
    long cur_idx = (idx == 0) ? 0 : cum_tiles[idx - 1];
    
    for (uint i = tile_min_y; i < tile_max_y; ++i) {
        for (uint j = tile_min_x; j < tile_max_x; ++j) {
            long tile_id = long(i * params.tile_width + j);
            // Encode: image_id | tile_id | depth
            isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_enc;
            flatten_ids[cur_idx] = int(idx);
            ++cur_idx;
        }
    }
}

/// Compute tile offsets from sorted intersection IDs
kernel void intersect_offset(
    device const long* isect_ids [[buffer(0)]],  // [n_isects]
    device int* offsets          [[buffer(1)]],  // [I, tile_height, tile_width]
    constant uint& n_isects      [[buffer(2)]],
    constant uint& I             [[buffer(3)]],
    constant uint& n_tiles       [[buffer(4)]],
    constant uint& tile_n_bits   [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= n_isects) {
        return;
    }
    
    long isect_id_curr = isect_ids[idx] >> 32;
    long iid_curr = isect_id_curr >> tile_n_bits;
    long tid_curr = isect_id_curr & ((1L << tile_n_bits) - 1);
    long id_curr = iid_curr * long(n_tiles) + tid_curr;
    
    if (idx == 0) {
        // Write offsets until first valid tile
        for (uint i = 0; i < uint(id_curr) + 1; ++i) {
            offsets[i] = int(idx);
        }
    }
    
    if (idx == n_isects - 1) {
        // Write remaining offsets
        for (uint i = uint(id_curr) + 1; i < I * n_tiles; ++i) {
            offsets[i] = int(n_isects);
        }
    }
    
    if (idx > 0) {
        long isect_id_prev = isect_ids[idx - 1] >> 32;
        if (isect_id_prev == isect_id_curr) {
            return;
        }
        
        long iid_prev = isect_id_prev >> tile_n_bits;
        long tid_prev = isect_id_prev & ((1L << tile_n_bits) - 1);
        long id_prev = iid_prev * long(n_tiles) + tid_prev;
        
        for (uint i = uint(id_prev) + 1; i < uint(id_curr) + 1; ++i) {
            offsets[i] = int(idx);
        }
    }
}
