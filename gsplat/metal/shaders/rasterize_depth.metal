#include <metal_stdlib>
using namespace metal;

struct Projected2D {
    float2 mean2d;      // 2D projected mean
    float depth;        // Z-depth
    float3 conic;       // Inverse covariance (upper triangle: xx, xy, yy)
    int2 radii;         // Bounding box radii in pixels
    bool valid;         // Whether projection is valid
};

struct TileData {
    atomic_uint count;
    uint gaussian_ids[256]; // Max 256 Gaussians per tile
};

constant int TILE_SIZE = 16;

// Rasterize depth for a single tile
kernel void rasterize_depth_tile(
    constant Projected2D* projected [[buffer(0)]],
    constant float* opacities [[buffer(1)]],
    constant uint* sorted_gaussian_ids [[buffer(2)]],
    constant uint* tile_offsets [[buffer(3)]],
    device float* depth_buffer [[buffer(4)]],
    device float* alpha_buffer [[buffer(5)]],
    constant uint& image_width [[buffer(6)]],
    constant uint& image_height [[buffer(7)]],
    constant uint& tile_width [[buffer(8)]],
    uint2 tile_id [[thread_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    if (tile_id.x >= tile_width || tile_id.y >= (image_height + TILE_SIZE - 1) / TILE_SIZE) {
        return;
    }
    
    // Calculate pixel position
    uint2 pixel = uint2(
        tile_id.x * TILE_SIZE + local_id.x,
        tile_id.y * TILE_SIZE + local_id.y
    );
    
    if (pixel.x >= image_width || pixel.y >= image_height) {
        return;
    }
    
    uint pixel_idx = pixel.y * image_width + pixel.x;
    float2 pixf = float2(pixel) + 0.5; // Pixel center
    
    // Get tile range
    uint tile_idx = tile_id.y * tile_width + tile_id.x;
    uint start = tile_offsets[tile_idx];
    uint end = tile_offsets[tile_idx + 1];
    
    // Accumulated values
    float T = 1.0; // Transmittance
    float depth_accum = 0.0;
    
    // Iterate through Gaussians for this tile (sorted front-to-back by depth)
    for (uint i = start; i < end && T > 0.001; i++) {
        uint g_idx = sorted_gaussian_ids[i];
        Projected2D proj = projected[g_idx];
        
        if (!proj.valid) continue;
        
        // Compute Gaussian weight at this pixel
        float2 delta = pixf - proj.mean2d;
        float sigma = -0.5 * (proj.conic.x * delta.x * delta.x + 
                              2.0 * proj.conic.y * delta.x * delta.y +
                              proj.conic.z * delta.y * delta.y);
        
        if (sigma > 0.0) continue; // Outside 3-sigma bound
        
        float alpha = min(0.99, opacities[g_idx] * exp(sigma));
        if (alpha < 1.0/255.0) continue;
        
        // Accumulate depth
        float weight = alpha * T;
        depth_accum += proj.depth * weight;
        
        // Update transmittance
        T *= (1.0 - alpha);
    }
    
    // Write output
    depth_buffer[pixel_idx] = depth_accum;
    alpha_buffer[pixel_idx] = 1.0 - T;
}

// Sort Gaussians by depth for each tile
kernel void sort_gaussians_per_tile(
    constant Projected2D* projected [[buffer(0)]],
    device uint* tile_gaussian_counts [[buffer(1)]],
    device uint* tile_gaussian_ids [[buffer(2)]],
    device uint* tile_offsets [[buffer(3)]],
    constant uint& num_gaussians [[buffer(4)]],
    constant uint& tile_width [[buffer(5)]],
    constant uint& tile_height [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_gaussians) return;
    
    Projected2D proj = projected[gid];
    if (!proj.valid) return;
    
    // Find which tiles this Gaussian overlaps
    int2 rect_min = int2(
        max(0, int(proj.mean2d.x - proj.radii.x) / TILE_SIZE),
        max(0, int(proj.mean2d.y - proj.radii.y) / TILE_SIZE)
    );
    int2 rect_max = int2(
        min(int(tile_width) - 1, int(proj.mean2d.x + proj.radii.x) / TILE_SIZE),
        min(int(tile_height) - 1, int(proj.mean2d.y + proj.radii.y) / TILE_SIZE)
    );
    
    // Add this Gaussian to overlapping tiles
    for (int ty = rect_min.y; ty <= rect_max.y; ty++) {
        for (int tx = rect_min.x; tx <= rect_max.x; tx++) {
            uint tile_idx = ty * tile_width + tx;
            uint count = atomic_fetch_add_explicit(
                (device atomic_uint*)&tile_gaussian_counts[tile_idx],
                1,
                memory_order_relaxed
            );
            
            // Store Gaussian ID in tile's list (with depth for sorting)
            // This is simplified; a real implementation would use a better data structure
            if (count < 256) { // Max Gaussians per tile
                tile_gaussian_ids[tile_idx * 256 + count] = gid;
            }
        }
    }
}

// Compact and create offsets for tile data
kernel void create_tile_offsets(
    constant uint* tile_gaussian_counts [[buffer(0)]],
    device uint* tile_offsets [[buffer(1)]],
    constant uint& num_tiles [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid > num_tiles) return;
    
    if (tid == 0) {
        tile_offsets[0] = 0;
    }
    
    if (tid < num_tiles) {
        // Prefix sum to create offsets
        uint offset = 0;
        for (uint i = 0; i < tid; i++) {
            offset += tile_gaussian_counts[i];
        }
        tile_offsets[tid + 1] = offset;
    }
}
