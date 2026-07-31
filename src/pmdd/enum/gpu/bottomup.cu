// ----------------------------------------------------------
// CUDA layer expansion for MDD top-down/bottom-up passes
// ----------------------------------------------------------

#include "enum_types.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "../multiobj_enum.hpp"
#include "dominance_utils.cuh"
#include "enum_types.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize = 32;

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct SquareToDoubleInt {
    __host__ __device__ double operator()(const int x) const {
        const double value = static_cast<double>(x);
        return value * value;
    }
};

inline double population_std_from_sums(const double sum, const double sum_sq,
                                       const long long count) {
    if (count <= 0) {
        return 0.0;
    }
    const double mean = sum / static_cast<double>(count);
    double variance = sum_sq / static_cast<double>(count) - mean * mean;
    if (variance < 0.0) {
        variance = 0.0;
    }
    return std::sqrt(variance);
}

inline double population_std_from_device_counts(const thrust::device_vector<int> &counts) {
    const long long count = static_cast<long long>(counts.size());
    if (count <= 0) {
        return 0.0;
    }
    const long long sum_ll =
        thrust::reduce(counts.begin(), counts.end(), 0LL, thrust::plus<long long>());
    const double sum_sq = thrust::transform_reduce(
        counts.begin(), counts.end(), SquareToDoubleInt(), 0.0, thrust::plus<double>());
    return population_std_from_sums(static_cast<double>(sum_ll), sum_sq, count);
}

// ---------------------------------------------------------------
// Kernels for MDD layer expansion.
// ---------------------------------------------------------------

__global__ void count_edge_candidates_kernel(const int *edge_src, const int *prev_offsets,
                                             int *edge_counts, int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges)
        return;
    const int src = edge_src[e];
    edge_counts[e] = prev_offsets[src + 1] - prev_offsets[src];
}

__global__ void count_destination_candidates_kernel(const int *in_edge_offsets,
                                                    const int *edge_offsets, int next_nodes,
                                                    int *dst_counts, int *dst_blocks) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= next_nodes)
        return;
    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    const int count = edge_offsets[edge_end] - edge_offsets[edge_begin];
    dst_counts[dst] = count;
    if (dst_blocks) {
        dst_blocks[dst] = ceil_div(count, kThreadsPerBlock);
    }
}

__global__ void materialize_edge_candidates_kernel(const int *edge_src, const ObjType *edge_weights,
                                                   const int *edge_offsets, const int *edge_counts,
                                                   const int *prev_offsets,
                                                   const ObjType *prev_points, int num_edges,
                                                   ObjType *cand_points) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp = global_thread / kWarpSize;
    const int lane = global_thread & (kWarpSize - 1);
    const int total_warps = (gridDim.x * blockDim.x) / kWarpSize;
    if (total_warps <= 0)
        return;

    for (int e = global_warp; e < num_edges; e += total_warps) {
        const int src = edge_src[e];
        const int src_begin = prev_offsets[src];
        const int out_begin = edge_offsets[e];
        const int count = edge_counts[e];
        const ObjType *w = edge_weights + e * NOBJS;

        for (int k = lane; k < count; k += kWarpSize) {
            const int src_idx = src_begin + k;
            const int out_idx = out_begin + k;
#pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                cand_points[out_idx * NOBJS + o] = prev_points[src_idx * NOBJS + o] + w[o];
        }
    }
}

// Binary search helper device function
__device__ int find_dst_node(int block_idx, const int *block_offsets, int next_nodes) {
    int low = 0, high = next_nodes;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (block_idx < block_offsets[mid + 1]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

// mark_local_dominated_kernel uses a strictly load balanced 1D grid.
__global__ void mark_local_dominated_kernel(const ObjType *points, const int *in_edge_offsets,
                                            const int *edge_offsets, const int *block_offsets,
                                            int next_nodes, int *alive, int *next_sizes) {
    const int block_idx = blockIdx.x;
    const int dst = find_dst_node(block_idx, block_offsets, next_nodes);
    if (dst >= next_nodes)
        return;

    const int tile_i = block_idx - block_offsets[dst];

    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    const int begin = edge_offsets[edge_begin];
    const int end = edge_offsets[edge_end];
    const int len = end - begin;

    const int local_i = tile_i * blockDim.x + threadIdx.x;
    const bool valid_i = (local_i < len);
    const int i = begin + local_i;

    ObjType point_i[NOBJS];
    if (valid_i) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            point_i[o] = points[i * NOBJS + o];
    }

    bool dominated = false;
    __shared__ ObjType sh_points[kThreadsPerBlock * NOBJS];
    for (int j_base = 0; j_base < len; j_base += blockDim.x) {
        const int j_local = j_base + threadIdx.x;
        if (j_local < len) {
            const int j = begin + j_local;
#pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh_points[threadIdx.x * NOBJS + o] = points[j * NOBJS + o];
        }
        __syncthreads();
        if (valid_i && !dominated) {
            const int tile_count = min(len - j_base, (int)blockDim.x);
            for (int jj = 0; jj < tile_count; ++jj) {
                const int local_j = j_base + jj;

                if (local_j == local_i)
                    continue;
                if (dominates_or_tie_before(&sh_points[jj * NOBJS], point_i, local_j < local_i)) {
                    dominated = true;
                    break;
                }
            }
        }
        __syncthreads();
    }

    const int keep = (valid_i && !dominated) ? 1 : 0;
    if (valid_i)
        alive[i] = keep;

    __shared__ int live_sh[kThreadsPerBlock];
    live_sh[threadIdx.x] = keep;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            live_sh[threadIdx.x] += live_sh[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(&next_sizes[dst], live_sh[0]);
}

__global__ void compact_alive_points_kernel(const int *alive, const int *alive_prefix,
                                            const ObjType *in_points, ObjType *out_points,
                                            int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points || !alive[i])
        return;
    const int out_idx = alive_prefix[i];
    for (int o = 0; o < NOBJS; ++o)
        out_points[out_idx * NOBJS + o] = in_points[i * NOBJS + o];
}

// Layer-value heuristic: sum_i( sizes[i] * arc_counts[i] )
__global__ void compute_layer_score_kernel(const int *offsets, const int *arc_counts, int *out,
                                           int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = (offsets[i + 1] - offsets[i]) * arc_counts[i];
}

// Removed functors for global sorting since they break edge_offsets mapping

} // anonymous namespace

// ---------------------------------------------------------------
// expand_layer_frontiers: runs expansion kernels for one MDD layer.
// Works identically for top-down and bottom-up.
// ---------------------------------------------------------------
bool expand_layer_frontiers(
    const thrust::device_vector<int> &in_edge_offsets, const thrust::device_vector<int> &edge_src,
    const thrust::device_vector<ObjType> &edge_weights, int num_edges, int next_nodes,
    const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch, long long *total_candidates_out,
    long long *total_next_out, double *std_candidates_out, double *std_survivors_out,
    long long *gpu_mem_baseline_used_bytes, long long *gpu_mem_peak_used_bytes,
    long long *gpu_mem_peak_reserved_bytes) {
    if (total_candidates_out != NULL) {
        *total_candidates_out = 0;
    }
    if (total_next_out != NULL) {
        *total_next_out = 0;
    }
    if (std_candidates_out != NULL) {
        *std_candidates_out = 0.0;
    }
    if (std_survivors_out != NULL) {
        *std_survivors_out = 0.0;
    }
    d_next_sizes.assign(next_nodes, 0);
    d_next_offsets.assign(next_nodes + 1, 0);

    if (num_edges == 0) {
        d_next_points.clear();
        return true;
    }

    // Per-edge candidate counts
    thrust::device_vector<int> d_edge_counts(num_edges, 0);
    thrust::device_vector<int> d_edge_offsets(num_edges + 1, 0);

    count_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(edge_src.data()), thrust::raw_pointer_cast(d_prev_offsets.data()),
        thrust::raw_pointer_cast(d_edge_counts.data()), num_edges);
    if (!sync_kernel("edge_counts", reason))
        return false;

    thrust::exclusive_scan(d_edge_counts.begin(), d_edge_counts.end(), d_edge_offsets.begin());
    const int last_offset = d_edge_offsets[num_edges - 1];
    const int last_count = d_edge_counts[num_edges - 1];
    const int total_candidates = last_offset + last_count;
    if (total_candidates_out != NULL) {
        *total_candidates_out = total_candidates;
    }
    d_edge_offsets[num_edges] = total_candidates;

    if (total_candidates == 0) {
        d_next_points.clear();
        return true;
    }

    thrust::device_vector<int> d_cand_counts(next_nodes, 0);
    thrust::device_vector<int> d_dst_blocks(next_nodes, 0);

    count_destination_candidates_kernel<<<ceil_div(next_nodes, kThreadsPerBlock),
                                          kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(in_edge_offsets.data()),
        thrust::raw_pointer_cast(d_edge_offsets.data()), next_nodes,
        thrust::raw_pointer_cast(d_cand_counts.data()),
        thrust::raw_pointer_cast(d_dst_blocks.data()));
    if (!sync_kernel("dst_counts", reason))
        return false;
    if (std_candidates_out != NULL) {
        *std_candidates_out = population_std_from_device_counts(d_cand_counts);
    }

    if (total_candidates > max_candidate_points_per_batch) {
        thrust::host_vector<int> h_in_edge_offsets = in_edge_offsets;
        thrust::host_vector<int> h_edge_offsets = d_edge_offsets;

        d_next_points.clear();
        d_next_sizes.assign(next_nodes, 0);

        int dst_begin = 0;
        while (dst_begin < next_nodes) {
            int dst_end = dst_begin + 1;
            long long batch_candidates =
                static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_end]]) -
                static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_begin]]);

            while (dst_end < next_nodes && batch_candidates < max_candidate_points_per_batch) {
                const long long next_candidates =
                    static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_end + 1]]) -
                    static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_begin]]);
                if (next_candidates > max_candidate_points_per_batch && batch_candidates > 0) {
                    break;
                }
                ++dst_end;
                batch_candidates = next_candidates;
            }

            if (batch_candidates <= 0) {
                ++dst_begin;
                continue;
            }

            const int edge_begin = h_in_edge_offsets[dst_begin];
            const int edge_end = h_in_edge_offsets[dst_end];
            const int batch_edges = edge_end - edge_begin;
            const int batch_nodes = dst_end - dst_begin;

            thrust::host_vector<int> h_batch_in_offsets(batch_nodes + 1, 0);
            for (int i = 0; i <= batch_nodes; ++i) {
                h_batch_in_offsets[i] = h_in_edge_offsets[dst_begin + i] - edge_begin;
            }
            thrust::device_vector<int> d_batch_in_offsets = h_batch_in_offsets;
            thrust::device_vector<int> d_batch_edge_offsets(batch_edges + 1, 0);

            thrust::exclusive_scan(d_edge_counts.begin() + edge_begin,
                                   d_edge_counts.begin() + edge_end, d_batch_edge_offsets.begin());
            const int batch_last_offset = d_batch_edge_offsets[batch_edges - 1];
            const int batch_last_count = d_edge_counts[edge_end - 1];
            const int batch_total_candidates = batch_last_offset + batch_last_count;
            d_batch_edge_offsets[batch_edges] = batch_total_candidates;
            if (batch_total_candidates <= 0) {
                dst_begin = dst_end;
                continue;
            }

            thrust::device_vector<ObjType> d_batch_cand_points(batch_total_candidates * NOBJS, 0);
            materialize_edge_candidates_kernel<<<ceil_div(batch_edges, kThreadsPerBlock),
                                                 kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(edge_src.data()) + edge_begin,
                thrust::raw_pointer_cast(edge_weights.data()) + edge_begin * NOBJS,
                thrust::raw_pointer_cast(d_batch_edge_offsets.data()),
                thrust::raw_pointer_cast(d_edge_counts.data()) + edge_begin,
                thrust::raw_pointer_cast(d_prev_offsets.data()),
                thrust::raw_pointer_cast(d_prev_points.data()), batch_edges,
                thrust::raw_pointer_cast(d_batch_cand_points.data()));
            if (!sync_kernel("batch_expand_cand", reason))
                return false;

            if (gpu_mem_baseline_used_bytes != NULL && gpu_mem_peak_used_bytes != NULL &&
                gpu_mem_peak_reserved_bytes != NULL) {
                if (!sample_gpu_memory_peak(reason, *gpu_mem_baseline_used_bytes,
                                            gpu_mem_peak_used_bytes, gpu_mem_peak_reserved_bytes)) {
                    return false;
                }
            }

            thrust::device_vector<int> d_batch_sizes(batch_nodes, 0);
            thrust::device_vector<int> d_batch_blocks(batch_nodes, 0);
            count_destination_candidates_kernel<<<ceil_div(batch_nodes, kThreadsPerBlock),
                                                  kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                thrust::raw_pointer_cast(d_batch_edge_offsets.data()), batch_nodes,
                thrust::raw_pointer_cast(d_batch_sizes.data()),
                thrust::raw_pointer_cast(d_batch_blocks.data()));
            if (!sync_kernel("batch_dst_counts", reason))
                return false;

            thrust::device_vector<int> d_batch_alive(batch_total_candidates, 0);
            thrust::device_vector<int> d_batch_next_sizes(batch_nodes, 0);
            const int max_seg_size = thrust::reduce(d_batch_sizes.begin(), d_batch_sizes.end(), 0,
                                                    thrust::maximum<int>());
            if (max_seg_size > 0) {
                thrust::device_vector<int> d_batch_block_offsets(batch_nodes + 1, 0);
                thrust::exclusive_scan(d_batch_blocks.begin(), d_batch_blocks.end(),
                                       d_batch_block_offsets.begin());
                d_batch_block_offsets[batch_nodes] =
                    thrust::reduce(d_batch_blocks.begin(), d_batch_blocks.end(), 0);

                const int total_blocks = d_batch_block_offsets[batch_nodes];
                if (total_blocks > 0) {
                    mark_local_dominated_kernel<<<total_blocks, kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(d_batch_cand_points.data()),
                        thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                        thrust::raw_pointer_cast(d_batch_edge_offsets.data()),
                        thrust::raw_pointer_cast(d_batch_block_offsets.data()), batch_nodes,
                        thrust::raw_pointer_cast(d_batch_alive.data()),
                        thrust::raw_pointer_cast(d_batch_next_sizes.data()));
                    if (!sync_kernel("batch_dom", reason))
                        return false;
                }
            }

            thrust::device_vector<int> d_batch_alive_prefix(batch_total_candidates, 0);
            thrust::exclusive_scan(d_batch_alive.begin(), d_batch_alive.end(),
                                   d_batch_alive_prefix.begin());
            const int batch_total_next =
                thrust::reduce(d_batch_alive.begin(), d_batch_alive.end(), 0);

            thrust::device_vector<ObjType> d_batch_next_points(batch_total_next * NOBJS, 0);
            if (batch_total_next > 0) {
                compact_alive_points_kernel<<<ceil_div(batch_total_candidates, kThreadsPerBlock),
                                              kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(d_batch_alive.data()),
                    thrust::raw_pointer_cast(d_batch_alive_prefix.data()),
                    thrust::raw_pointer_cast(d_batch_cand_points.data()),
                    thrust::raw_pointer_cast(d_batch_next_points.data()), batch_total_candidates);
                if (!sync_kernel("batch_scatter", reason))
                    return false;

                const size_t old_size = d_next_points.size();
                d_next_points.resize(old_size + d_batch_next_points.size());
                thrust::copy(d_batch_next_points.begin(), d_batch_next_points.end(),
                             d_next_points.begin() + old_size);
            }

            thrust::copy(d_batch_next_sizes.begin(), d_batch_next_sizes.end(),
                         d_next_sizes.begin() + dst_begin);

            dst_begin = dst_end;
        }

        const int total_next = d_next_points.size() / NOBJS;
        if (total_next_out != NULL) {
            *total_next_out = total_next;
        }
        if (std_survivors_out != NULL) {
            *std_survivors_out = population_std_from_device_counts(d_next_sizes);
        }

        thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
        d_next_offsets[next_nodes] = total_next;
        return true;
    }

    // Expand candidate points
    thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);
    materialize_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(edge_src.data()), thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(d_edge_offsets.data()),
        thrust::raw_pointer_cast(d_edge_counts.data()),
        thrust::raw_pointer_cast(d_prev_offsets.data()),
        thrust::raw_pointer_cast(d_prev_points.data()), num_edges,
        thrust::raw_pointer_cast(d_cand_points.data()));
    if (!sync_kernel("expand_cand", reason))
        return false;
    // Primary peak checkpoint: candidate points are fully materialized and not yet compacted by
    // dominance.
    if (gpu_mem_baseline_used_bytes != NULL && gpu_mem_peak_used_bytes != NULL &&
        gpu_mem_peak_reserved_bytes != NULL) {
        if (!sample_gpu_memory_peak(reason, *gpu_mem_baseline_used_bytes, gpu_mem_peak_used_bytes,
                                    gpu_mem_peak_reserved_bytes)) {
            return false;
        }
    }

    thrust::device_vector<int> d_alive(total_candidates, 0);

    const int max_seg_size =
        thrust::reduce(d_cand_counts.begin(), d_cand_counts.end(), 0, thrust::maximum<int>());
    if (max_seg_size > 0) {
        thrust::device_vector<int> d_block_offsets(next_nodes + 1, 0);
        thrust::exclusive_scan(d_dst_blocks.begin(), d_dst_blocks.end(), d_block_offsets.begin());
        d_block_offsets[next_nodes] = thrust::reduce(d_dst_blocks.begin(), d_dst_blocks.end(), 0);

        const int total_blocks = d_block_offsets[next_nodes];
        if (total_blocks > 0) {
            mark_local_dominated_kernel<<<total_blocks, kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_cand_points.data()),
                thrust::raw_pointer_cast(in_edge_offsets.data()),
                thrust::raw_pointer_cast(d_edge_offsets.data()),
                thrust::raw_pointer_cast(d_block_offsets.data()), next_nodes,
                thrust::raw_pointer_cast(d_alive.data()),
                thrust::raw_pointer_cast(d_next_sizes.data()));
            if (!sync_kernel("dom", reason))
                return false;
        }
    }

    // Compact surviving points
    thrust::device_vector<int> d_alive_prefix(total_candidates, 0);
    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
    const int total_next = thrust::reduce(d_next_sizes.begin(), d_next_sizes.end(), 0);
    if (total_next_out != NULL) {
        *total_next_out = total_next;
    }
    if (std_survivors_out != NULL) {
        *std_survivors_out = population_std_from_device_counts(d_next_sizes);
    }

    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(), d_next_offsets.begin());
    d_next_offsets[next_nodes] = total_next;

    d_next_points.resize(total_next * NOBJS);
    if (total_next > 0) {
        compact_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock),
                                      kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_alive.data()),
            thrust::raw_pointer_cast(d_alive_prefix.data()),
            thrust::raw_pointer_cast(d_cand_points.data()),
            thrust::raw_pointer_cast(d_next_points.data()), total_candidates);
        if (!sync_kernel("scatter", reason))
            return false;
    }
    return true;
}

// Compute layer value heuristic on GPU, return scalar to host
int compute_expansion_score(const thrust::device_vector<int> &offsets,
                            const thrust::device_vector<int> &arc_counts, int num_nodes) {
    if (num_nodes <= 0)
        return 0;
    thrust::device_vector<int> tmp(num_nodes, 0);
    compute_layer_score_kernel<<<ceil_div(num_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(offsets.data()), thrust::raw_pointer_cast(arc_counts.data()),
        thrust::raw_pointer_cast(tmp.data()), num_nodes);
    cudaDeviceSynchronize();
    return thrust::reduce(tmp.begin(), tmp.end(), 0);
}

// ---------------------------------------------------------------
// Layer expansion API
// ---------------------------------------------------------------

bool bottomup_expand_mdd_layer(
    const PackedMDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch, long long *total_candidates_out,
    long long *total_next_out, double *std_candidates_out, double *std_survivors_out,
    long long *gpu_mem_baseline_used_bytes, long long *gpu_mem_peak_used_bytes,
    long long *gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.bu_in_edge_offsets, packed_layer.bu_edge_src, packed_layer.bu_edge_weights,
        packed_layer.bu_num_edges, packed_layer.num_nodes, d_prev_offsets, d_prev_points,
        d_next_sizes, d_next_offsets, d_next_points, reason, max_candidate_points_per_batch,
        total_candidates_out, total_next_out, std_candidates_out, std_survivors_out,
        gpu_mem_baseline_used_bytes, gpu_mem_peak_used_bytes, gpu_mem_peak_reserved_bytes);
}

bool bottomup_expand_bdd_layer(
    const PackedBDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch, long long *total_candidates_out,
    long long *total_next_out, double *std_candidates_out, double *std_survivors_out,
    long long *gpu_mem_baseline_used_bytes, long long *gpu_mem_peak_used_bytes,
    long long *gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.bu_in_edge_offsets, packed_layer.bu_edge_src, packed_layer.bu_edge_weights,
        packed_layer.bu_num_edges, packed_layer.num_nodes, d_prev_offsets, d_prev_points,
        d_next_sizes, d_next_offsets, d_next_points, reason, max_candidate_points_per_batch,
        total_candidates_out, total_next_out, std_candidates_out, std_survivors_out,
        gpu_mem_baseline_used_bytes, gpu_mem_peak_used_bytes, gpu_mem_peak_reserved_bytes);
}
