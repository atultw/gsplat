// Rasterization kernel for Metal
// Port of RasterizeToPixels3DGSFwd.cu (forward pass only)

#include <metal_stdlib>
#include "common.h"

using namespace metal;

// ============================================
// Rasterization Parameters
// ============================================

struct RasterizeParams {
    uint I;             // Number of images
    uint N;             // Number of Gaussians per image (0 if packed)
    uint n_isects;      // Total intersections
    uint image_width;
    uint image_height;
    uint tile_size;
    uint tile_width;
    uint tile_height;
    uint channels;      // Number of color channels
    bool packed;
};

// ============================================
// Rasterization Kernel
// ============================================

/// Rasterize Gaussians to pixels using tile-based rendering
/// Each threadgroup processes one tile, each thread processes one pixel
kernel void rasterize_to_pixels_fwd(
    // Gaussian data
    device const float* means2d         [[buffer(0)]],   // [I, N, 2] or [nnz, 2]
    device const float* conics          [[buffer(1)]],   // [I, N, 3] or [nnz, 3]
    device const float* colors          [[buffer(2)]],   // [I, N, C] or [nnz, C]
    device const float* opacities       [[buffer(3)]],   // [I, N] or [nnz]
    device const float* backgrounds     [[buffer(4)]],   // [I, C] or nullptr
    device const bool* masks            [[buffer(5)]],   // [I, th, tw] or nullptr
    // Tile data
    device const int* tile_offsets      [[buffer(6)]],   // [I, tile_height, tile_width]
    device const int* flatten_ids       [[buffer(7)]],   // [n_isects]
    // Outputs
    device float* render_colors         [[buffer(8)]],   // [I, H, W, C]
    device float* render_alphas         [[buffer(9)]],   // [I, H, W]
    device int* last_ids                [[buffer(10)]],  // [I, H, W]
    // Parameters
    constant RasterizeParams& params    [[buffer(11)]],
    // Thread/group info
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    // Group layout: (image_id, tile_row, tile_col)
    uint image_id = group_id.x;
    uint tile_row = group_id.y;
    uint tile_col = group_id.z;
    uint tile_id = tile_row * params.tile_width + tile_col;
    
    // Thread layout: (x, y) within tile
    uint local_x = local_id.x;
    uint local_y = local_id.y;
    uint block_size = group_size.x * group_size.y;
    uint thread_rank = local_y * group_size.x + local_x;
    
    // Pixel coordinates
    uint px = tile_col * params.tile_size + local_x;
    uint py = tile_row * params.tile_size + local_y;
    
    // Pointer offsets for this image
    uint tile_offset_base = image_id * params.tile_height * params.tile_width;
    uint pixel_offset_base = image_id * params.image_height * params.image_width;
    
    float pixel_x = float(px) + 0.5f;
    float pixel_y = float(py) + 0.5f;
    uint pix_id = py * params.image_width + px;
    
    // Check if pixel is inside image
    bool inside = (py < params.image_height && px < params.image_width);
    bool done = !inside;
    
    // Check tile mask if provided
    if (masks != nullptr && inside && !masks[tile_offset_base + tile_id]) {
        // Write background color and return
        uint out_idx = (pixel_offset_base + pix_id) * params.channels;
        for (uint c = 0; c < params.channels; ++c) {
            render_colors[out_idx + c] = (backgrounds != nullptr) 
                ? backgrounds[image_id * params.channels + c] 
                : 0.0f;
        }
        render_alphas[pixel_offset_base + pix_id] = 0.0f;
        last_ids[pixel_offset_base + pix_id] = 0;
        return;
    }
    
    // Get Gaussian range for this tile
    int range_start = tile_offsets[tile_offset_base + tile_id];
    int range_end;
    if (image_id == params.I - 1 && tile_id == params.tile_width * params.tile_height - 1) {
        range_end = int(params.n_isects);
    } else {
        range_end = tile_offsets[tile_offset_base + tile_id + 1];
    }
    
    uint num_batches = ((range_end - range_start) + block_size - 1) / block_size;
    
    // Shared memory for batching Gaussians
    threadgroup int id_batch[256];
    threadgroup float3 xy_opacity_batch[256];
    threadgroup float3 conic_batch[256];
    
    // Per-thread state
    float T = 1.0f;  // Transmittance
    uint cur_idx = 0;
    float max_contrib = 0.0f;
    int best_cur_idx = -1;
    
    // Fixed channel array (Metal doesn't support VLAs)
    float pix_out[64];
    for (uint c = 0; c < 64 && c < params.channels; ++c) {
        pix_out[c] = 0.0f;
    }
    
    // Process Gaussians in batches
    for (uint b = 0; b < num_batches; ++b) {
        // Early exit if all threads are done
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load Gaussian data into shared memory
        uint batch_start = uint(range_start) + block_size * b;
        uint load_idx = batch_start + thread_rank;
        
        if (load_idx < uint(range_end)) {
            int g = flatten_ids[load_idx];
            id_batch[thread_rank] = g;
            
            float2 xy = float2(means2d[g * 2], means2d[g * 2 + 1]);
            float opac = opacities[g];
            xy_opacity_batch[thread_rank] = float3(xy.x, xy.y, opac);
            conic_batch[thread_rank] = float3(conics[g * 3], conics[g * 3 + 1], conics[g * 3 + 2]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Process Gaussians in this batch
        uint batch_size = min(block_size, uint(range_end) - batch_start);
        
        for (uint t = 0; t < batch_size && !done; ++t) {
            float3 conic = conic_batch[t];
            float3 xy_opac = xy_opacity_batch[t];
            float opac = xy_opac.z;
            float2 delta = float2(xy_opac.x - pixel_x, xy_opac.y - pixel_y);
            
            float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                   conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
            
            float alpha = min(0.999f, opac * exp(-sigma));
            
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }
            
            float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) {
                done = true;
                break;
            }
            
            int g = id_batch[t];
            float vis = alpha * T;
            
            // Accumulate color
            for (uint c = 0; c < params.channels && c < 64; ++c) {
                pix_out[c] += colors[g * params.channels + c] * vis;
            }
            
            cur_idx = batch_start + t;
            
            if (vis > max_contrib) {
                max_contrib = vis;
                best_cur_idx = int(cur_idx);
            }
            
            T = next_T;
        }
    }
    
    // Write output
    if (inside) {
        uint out_idx = (pixel_offset_base + pix_id) * params.channels;
        
        for (uint c = 0; c < params.channels && c < 64; ++c) {
            float bg = (backgrounds != nullptr) 
                ? backgrounds[image_id * params.channels + c] 
                : 0.0f;
            render_colors[out_idx + c] = pix_out[c] + T * bg;
        }
        
        render_alphas[pixel_offset_base + pix_id] = 1.0f - T;
        last_ids[pixel_offset_base + pix_id] = best_cur_idx;
    }
}

// ============================================
// Specialized kernels for common channel counts
// ============================================

/// Optimized kernel for 3 color channels (RGB)
kernel void rasterize_to_pixels_fwd_rgb(
    device const float* means2d         [[buffer(0)]],
    device const float* conics          [[buffer(1)]],
    device const float* colors          [[buffer(2)]],
    device const float* opacities       [[buffer(3)]],
    device const float* backgrounds     [[buffer(4)]],
    device const bool* masks            [[buffer(5)]],
    device const int* tile_offsets      [[buffer(6)]],
    device const int* flatten_ids       [[buffer(7)]],
    device float* render_colors         [[buffer(8)]],
    device float* render_alphas         [[buffer(9)]],
    device int* last_ids                [[buffer(10)]],
    constant RasterizeParams& params    [[buffer(11)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint image_id = group_id.x;
    uint tile_row = group_id.y;
    uint tile_col = group_id.z;
    uint tile_id = tile_row * params.tile_width + tile_col;
    
    uint local_x = local_id.x;
    uint local_y = local_id.y;
    uint block_size = group_size.x * group_size.y;
    uint thread_rank = local_y * group_size.x + local_x;
    
    uint px = tile_col * params.tile_size + local_x;
    uint py = tile_row * params.tile_size + local_y;
    
    uint tile_offset_base = image_id * params.tile_height * params.tile_width;
    uint pixel_offset_base = image_id * params.image_height * params.image_width;
    
    float pixel_x = float(px) + 0.5f;
    float pixel_y = float(py) + 0.5f;
    uint pix_id = py * params.image_width + px;
    
    bool inside = (py < params.image_height && px < params.image_width);
    bool done = !inside;
    
    if (masks != nullptr && inside && !masks[tile_offset_base + tile_id]) {
        uint out_idx = (pixel_offset_base + pix_id) * 3;
        float3 bg = (backgrounds != nullptr) 
            ? float3(backgrounds[image_id * 3], backgrounds[image_id * 3 + 1], backgrounds[image_id * 3 + 2])
            : float3(0.0f);
        render_colors[out_idx] = bg.x;
        render_colors[out_idx + 1] = bg.y;
        render_colors[out_idx + 2] = bg.z;
        render_alphas[pixel_offset_base + pix_id] = 0.0f;
        last_ids[pixel_offset_base + pix_id] = 0;
        return;
    }
    
    int range_start = tile_offsets[tile_offset_base + tile_id];
    int range_end = (image_id == params.I - 1 && tile_id == params.tile_width * params.tile_height - 1)
        ? int(params.n_isects)
        : tile_offsets[tile_offset_base + tile_id + 1];
    
    uint num_batches = ((range_end - range_start) + block_size - 1) / block_size;
    
    threadgroup int id_batch[256];
    threadgroup float3 xy_opacity_batch[256];
    threadgroup float3 conic_batch[256];
    
    float T = 1.0f;
    uint cur_idx = 0;
    float max_contrib = 0.0f;
    int best_cur_idx = -1;
    float3 pix_out = float3(0.0f);
    
    for (uint b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint batch_start = uint(range_start) + block_size * b;
        uint load_idx = batch_start + thread_rank;
        
        if (load_idx < uint(range_end)) {
            int g = flatten_ids[load_idx];
            id_batch[thread_rank] = g;
            xy_opacity_batch[thread_rank] = float3(means2d[g * 2], means2d[g * 2 + 1], opacities[g]);
            conic_batch[thread_rank] = float3(conics[g * 3], conics[g * 3 + 1], conics[g * 3 + 2]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint batch_size = min(block_size, uint(range_end) - batch_start);
        
        for (uint t = 0; t < batch_size && !done; ++t) {
            float3 conic = conic_batch[t];
            float3 xy_opac = xy_opacity_batch[t];
            float2 delta = float2(xy_opac.x - pixel_x, xy_opac.y - pixel_y);
            
            float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) 
                        + conic.y * delta.x * delta.y;
            float alpha = min(0.999f, xy_opac.z * exp(-sigma));
            
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;
            
            float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { done = true; break; }
            
            int g = id_batch[t];
            float vis = alpha * T;
            pix_out += float3(colors[g * 3], colors[g * 3 + 1], colors[g * 3 + 2]) * vis;
            cur_idx = batch_start + t;
            
            if (vis > max_contrib) {
                max_contrib = vis;
                best_cur_idx = int(cur_idx);
            }
            
            T = next_T;
        }
    }
    
    if (inside) {
        float3 bg = (backgrounds != nullptr)
            ? float3(backgrounds[image_id * 3], backgrounds[image_id * 3 + 1], backgrounds[image_id * 3 + 2])
            : float3(0.0f);
        uint out_idx = (pixel_offset_base + pix_id) * 3;
        render_colors[out_idx] = pix_out.x + T * bg.x;
        render_colors[out_idx + 1] = pix_out.y + T * bg.y;
        render_colors[out_idx + 2] = pix_out.z + T * bg.z;
        render_alphas[pixel_offset_base + pix_id] = 1.0f - T;
        last_ids[pixel_offset_base + pix_id] = best_cur_idx;
    }
}

/// Optimized kernel for 1 channel (depth)
kernel void rasterize_to_pixels_fwd_depth(
    device const float* means2d         [[buffer(0)]],
    device const float* conics          [[buffer(1)]],
    device const float* colors          [[buffer(2)]],  // depths as "colors"
    device const float* opacities       [[buffer(3)]],
    device const float* backgrounds     [[buffer(4)]],
    device const bool* masks            [[buffer(5)]],
    device const int* tile_offsets      [[buffer(6)]],
    device const int* flatten_ids       [[buffer(7)]],
    device float* render_colors         [[buffer(8)]],
    device float* render_alphas         [[buffer(9)]],
    device int* last_ids                [[buffer(10)]],
    constant RasterizeParams& params    [[buffer(11)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    uint image_id = group_id.x;
    uint tile_row = group_id.y;
    uint tile_col = group_id.z;
    uint tile_id = tile_row * params.tile_width + tile_col;
    
    uint local_x = local_id.x;
    uint local_y = local_id.y;
    uint block_size = group_size.x * group_size.y;
    uint thread_rank = local_y * group_size.x + local_x;
    
    uint px = tile_col * params.tile_size + local_x;
    uint py = tile_row * params.tile_size + local_y;
    
    uint tile_offset_base = image_id * params.tile_height * params.tile_width;
    uint pixel_offset_base = image_id * params.image_height * params.image_width;
    
    float pixel_x = float(px) + 0.5f;
    float pixel_y = float(py) + 0.5f;
    uint pix_id = py * params.image_width + px;
    
    bool inside = (py < params.image_height && px < params.image_width);
    bool done = !inside;
    
    if (masks != nullptr && inside && !masks[tile_offset_base + tile_id]) {
        render_colors[pixel_offset_base + pix_id] = (backgrounds != nullptr) ? backgrounds[image_id] : 0.0f;
        render_alphas[pixel_offset_base + pix_id] = 0.0f;
        last_ids[pixel_offset_base + pix_id] = 0;
        return;
    }
    
    int range_start = tile_offsets[tile_offset_base + tile_id];
    int range_end = (image_id == params.I - 1 && tile_id == params.tile_width * params.tile_height - 1)
        ? int(params.n_isects) : tile_offsets[tile_offset_base + tile_id + 1];
    
    uint num_batches = ((range_end - range_start) + block_size - 1) / block_size;
    
    threadgroup int id_batch[256];
    threadgroup float3 xy_opacity_batch[256];
    threadgroup float3 conic_batch[256];
    
    float T = 1.0f;
    uint cur_idx = 0;
    float max_contrib = 0.0f;
    int best_cur_idx = -1;
    float pix_out = 0.0f;
    
    for (uint b = 0; b < num_batches; ++b) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint batch_start = uint(range_start) + block_size * b;
        uint load_idx = batch_start + thread_rank;
        
        if (load_idx < uint(range_end)) {
            int g = flatten_ids[load_idx];
            id_batch[thread_rank] = g;
            xy_opacity_batch[thread_rank] = float3(means2d[g * 2], means2d[g * 2 + 1], opacities[g]);
            conic_batch[thread_rank] = float3(conics[g * 3], conics[g * 3 + 1], conics[g * 3 + 2]);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint batch_size = min(block_size, uint(range_end) - batch_start);
        
        for (uint t = 0; t < batch_size && !done; ++t) {
            float3 conic = conic_batch[t];
            float3 xy_opac = xy_opacity_batch[t];
            float2 delta = float2(xy_opac.x - pixel_x, xy_opac.y - pixel_y);
            
            float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) 
                        + conic.y * delta.x * delta.y;
            float alpha = min(0.999f, xy_opac.z * exp(-sigma));
            
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;
            
            float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { done = true; break; }
            
            int g = id_batch[t];
            float contrib = alpha * T;
            pix_out += colors[g] * contrib;
            cur_idx = batch_start + t;
            
            if (contrib > max_contrib) {
                max_contrib = contrib;
                best_cur_idx = int(cur_idx);
            }
            
            T = next_T;
        }
    }
    
    if (inside) {
        float bg = (backgrounds != nullptr) ? backgrounds[image_id] : 0.0f;
        render_colors[pixel_offset_base + pix_id] = pix_out + T * bg;
        render_alphas[pixel_offset_base + pix_id] = 1.0f - T;
        last_ids[pixel_offset_base + pix_id] = best_cur_idx;
    }
}
