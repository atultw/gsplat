// Radix sort for intersection IDs
// Simplified bitonic sort for Metal

#include <metal_stdlib>
using namespace metal;

// ============================================
// Bitonic Sort for Tile-Based Rendering
// ============================================

/// Bitonic compare and swap
inline void bitonic_compare_swap(
    device long* keys,
    device int* values,
    uint i, uint j,
    bool ascending
) {
    if ((keys[i] > keys[j]) == ascending) {
        // Swap keys
        long temp_key = keys[i];
        keys[i] = keys[j];
        keys[j] = temp_key;
        
        // Swap values
        int temp_val = values[i];
        values[i] = values[j];
        values[j] = temp_val;
    }
}

/// Bitonic sort step kernel
/// Each threadgroup sorts a portion of the data
kernel void bitonic_sort_step(
    device long* keys           [[buffer(0)]],
    device int* values          [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    constant uint& stage        [[buffer(3)]],  // log2 of current block size
    constant uint& step         [[buffer(4)]],  // log2 of comparison distance
    uint idx [[thread_position_in_grid]]
) {
    uint block_size = 1u << (stage + 1);
    uint half_block = 1u << step;
    
    // Determine which pair to compare
    uint block_id = idx / half_block;
    uint local_id = idx % half_block;
    
    uint i = block_id * half_block * 2 + local_id;
    uint j = i + half_block;
    
    if (j >= n) {
        return;
    }
    
    // Determine sort direction (alternates per block)
    bool ascending = ((i / block_size) % 2) == 0;
    
    bitonic_compare_swap(keys, values, i, j, ascending);
}

/// Parallel radix sort for 64-bit keys
/// Uses counting sort per radix digit
struct RadixSortParams {
    uint n;             // Number of elements
    uint bit_offset;    // Starting bit for this pass
    uint bits_per_pass; // Usually 4 or 8
};

/// Count occurrences of each radix digit
kernel void radix_count(
    device const long* keys     [[buffer(0)]],
    device atomic_uint* counts  [[buffer(1)]],  // [num_buckets]
    constant RadixSortParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.n) {
        return;
    }
    
    uint num_buckets = 1u << params.bits_per_pass;
    uint mask = num_buckets - 1;
    
    long key = keys[idx];
    uint digit = uint((key >> params.bit_offset) & mask);
    
    atomic_fetch_add_explicit(&counts[digit], 1, memory_order_relaxed);
}

/// Scatter elements to sorted positions using prefix sums
kernel void radix_scatter(
    device const long* keys_in      [[buffer(0)]],
    device const int* values_in     [[buffer(1)]],
    device long* keys_out           [[buffer(2)]],
    device int* values_out          [[buffer(3)]],
    device const uint* offsets      [[buffer(4)]],  // Prefix sums of counts
    device atomic_uint* counters    [[buffer(5)]],  // Per-bucket insertion counters
    constant RadixSortParams& params [[buffer(6)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.n) {
        return;
    }
    
    uint num_buckets = 1u << params.bits_per_pass;
    uint mask = num_buckets - 1;
    
    long key = keys_in[idx];
    int value = values_in[idx];
    uint digit = uint((key >> params.bit_offset) & mask);
    
    // Get output position
    uint base = offsets[digit];
    uint offset = atomic_fetch_add_explicit(&counters[digit], 1, memory_order_relaxed);
    
    keys_out[base + offset] = key;
    values_out[base + offset] = value;
}

/// Simple single-threadgroup sort for small arrays
/// Used for final sorting within tiles
kernel void insertion_sort_local(
    device long* keys       [[buffer(0)]],
    device int* values      [[buffer(1)]],
    constant uint& start    [[buffer(2)]],
    constant uint& end      [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Only thread 0 does the work for simplicity
    if (tid != 0) {
        return;
    }
    
    uint n = end - start;
    if (n <= 1) {
        return;
    }
    
    // Simple insertion sort (fine for small arrays)
    for (uint i = start + 1; i < end; ++i) {
        long key = keys[i];
        int value = values[i];
        uint j = i;
        
        while (j > start && keys[j - 1] > key) {
            keys[j] = keys[j - 1];
            values[j] = values[j - 1];
            --j;
        }
        
        keys[j] = key;
        values[j] = value;
    }
}
