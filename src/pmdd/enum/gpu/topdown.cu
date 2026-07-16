// ----------------------------------------------------------
// CUDA Top-Down Enumeration - Implementation
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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize = 32;

struct LayerDominanceContext {
    thrust::device_vector<int> *next_sizes;
    thrust::device_vector<int> *next_offsets;
    thrust::device_vector<ObjType> *next_points;
    thrust::device_vector<int> *layer_min_weight;
    thrust::device_vector<int> *layer_single_parent_id;
    thrust::device_vector<int> *layer_single_parent_arc;
    std::string *reason;
    bool *warned_non_knapsack;
    bool failed;
};

thread_local LayerDominanceContext *g_layer_dom_ctx = NULL;

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

inline bool fail_layer_filter(LayerDominanceContext *ctx, const std::string &message) {
    if (ctx != NULL) {
        ctx->failed = true;
    }
    return set_reason(ctx != NULL ? ctx->reason : NULL, message);
}

__global__ void count_edge_candidates_kernel(const int *edge_src, const int *prev_offsets,
                                             int *edge_counts, int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }
    const int src = edge_src[e];
    edge_counts[e] = prev_offsets[src + 1] - prev_offsets[src];
}

__global__ void count_destination_candidates_kernel(const int *in_edge_offsets,
                                                    const int *edge_offsets, int next_nodes,
                                                    int *dst_counts, int *dst_blocks) {
    const int dst = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst >= next_nodes) {
        return;
    }
    const int edge_begin = in_edge_offsets[dst];
    const int edge_end = in_edge_offsets[dst + 1];
    const int count = edge_offsets[edge_end] - edge_offsets[edge_begin];
    dst_counts[dst] = count;
    if (dst_blocks != NULL) {
        dst_blocks[dst] = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    }
}

__global__ void materialize_edge_candidates_kernel(const int *edge_src, const ObjType *edge_weights,
                                                   const int *edge_offsets, const int *edge_counts,
                                                   const int *prev_offsets,
                                                   const ObjType *prev_points, int num_edges,
                                                   ObjType *cand_points) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp = global_thread / kWarpSize;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int total_warps = (gridDim.x * blockDim.x) / kWarpSize;
    if (total_warps <= 0) {
        return;
    }

    for (int e = global_warp; e < num_edges; e += total_warps) {
        const int src = edge_src[e];
        const int src_begin = prev_offsets[src];
        const int out_begin = edge_offsets[e];
        const int count = edge_counts[e];
        const ObjType *w = edge_weights + (e * NOBJS);

        for (int k = lane; k < count; k += kWarpSize) {
            const int src_idx = src_begin + k;
            const int out_idx = out_begin + k;

#pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                cand_points[out_idx * NOBJS + o] = prev_points[src_idx * NOBJS + o] + w[o];
            }
        }
    }
}

__device__ int find_dst_node(int block_idx, const int *block_offsets, int next_nodes) {
    int low = 0;
    int high = next_nodes;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        if (block_idx < block_offsets[mid + 1]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return low;
}

__global__ void mark_dominated_by_dst_dynamic_1d_kernel(const ObjType *points,
                                                        const int *in_edge_offsets,
                                                        const int *edge_offsets,
                                                        const int *block_offsets, int next_nodes,
                                                        int *alive, int *next_sizes) {
    const int block_idx = blockIdx.x;
    const int dst = find_dst_node(block_idx, block_offsets, next_nodes);
    if (dst >= next_nodes) {
        return;
    }

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
        for (int o = 0; o < NOBJS; ++o) {
            point_i[o] = points[i * NOBJS + o];
        }
    }

    bool dominated = false;
    __shared__ ObjType sh_points[kThreadsPerBlock * NOBJS];
    for (int j_base = 0; j_base < len; j_base += blockDim.x) {
        const int j_local = j_base + threadIdx.x;
        if (j_local < len) {
            const int j = begin + j_local;
#pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                sh_points[threadIdx.x * NOBJS + o] = points[j * NOBJS + o];
            }
        }
        __syncthreads();

        if (valid_i && !dominated) {
            const int tile_count = min(len - j_base, static_cast<int>(blockDim.x));
            for (int jj = 0; jj < tile_count; ++jj) {
                const int local_j = j_base + jj;
                if (local_j == local_i) {
                    continue;
                }

                if (dominates_or_tie_before(&sh_points[jj * NOBJS], point_i, local_j < local_i)) {
                    dominated = true;
                    break;
                }
            }
        }
        __syncthreads();
    }

    const int keep = (valid_i && !dominated) ? 1 : 0;
    if (valid_i) {
        alive[i] = keep;
    }

    __shared__ int live_sh[kThreadsPerBlock];
    live_sh[threadIdx.x] = keep;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            live_sh[threadIdx.x] += live_sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&next_sizes[dst], live_sh[0]);
    }
}

__global__ void compact_alive_points_kernel(const int *alive, const int *alive_prefix,
                                            const ObjType *in_points, ObjType *out_points,
                                            int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points || !alive[i]) {
        return;
    }

    const int out_idx = alive_prefix[i];
    for (int o = 0; o < NOBJS; ++o) {
        out_points[out_idx * NOBJS + o] = in_points[i * NOBJS + o];
    }
}

__global__ void mark_dominated_knapsack_pairs_kernel(const int *target_nodes, const int *cmp_node_a,
                                                     const int *cmp_node_b, const int *offsets,
                                                     const ObjType *points, int num_targets,
                                                     int *keep) {
    const int t = blockIdx.x;
    if (t >= num_targets) {
        return;
    }

    const int target_node = target_nodes[t];
    const int cmpA = cmp_node_a[t];
    const int cmpB = cmp_node_b[t];

    const int target_begin = offsets[target_node];
    const int target_end = offsets[target_node + 1];

    for (int local_i = threadIdx.x; local_i < target_end - target_begin; local_i += blockDim.x) {
        const int target_idx = target_begin + local_i;
        bool dominated = false;

        const int cmp_nodes[2] = {cmpA, cmpB};
        for (int c = 0; c < 2 && !dominated; ++c) {
            const int cmp_node = cmp_nodes[c];
            if (cmp_node < 0) {
                continue;
            }

            const int cmp_begin = offsets[cmp_node];
            const int cmp_end = offsets[cmp_node + 1];
            for (int cmp_idx = cmp_begin; cmp_idx < cmp_end && !dominated; ++cmp_idx) {
                dominated = true;
                for (int o = 0; o < NOBJS && dominated; ++o) {
                    const ObjType a = points[cmp_idx * NOBJS + o];
                    const ObjType b = points[target_idx * NOBJS + o];
                    dominated = (a >= b);
                }
            }
        }

        keep[target_idx] = dominated ? 0 : 1;
    }
}

__global__ void recompute_sizes_from_keep_kernel(const int *offsets, const int *keep, int num_nodes,
                                                 int *out_sizes) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) {
        return;
    }

    const int begin = offsets[node];
    const int end = offsets[node + 1];
    int live = 0;
    for (int i = begin; i < end; ++i) {
        live += keep[i];
    }
    out_sizes[node] = live;
}

__global__ void build_knapsack_dominance_pairs_kernel(const int *active_nodes, int active_size,
                                                      const int *single_parent_id,
                                                      const int *single_parent_arc,
                                                      int *target_nodes, int *cmp_node_a,
                                                      int *cmp_node_b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active_size - 1) {
        return;
    }

    const int node1 = active_nodes[i];
    const int p1 = single_parent_id[node1];
    const int a1 = single_parent_arc[node1];

    int cmpA = -1;
    int cmpB = -1;
    int cmp_count = 0;

    for (int delta = 1; delta <= 2 && i + delta < active_size; ++delta) {
        const int node2 = active_nodes[i + delta];
        const int p2 = single_parent_id[node2];
        const int a2 = single_parent_arc[node2];
        const bool skip = (p1 >= 0 && p2 >= 0 && p1 == p2 && a1 != a2);
        if (skip) {
            continue;
        }
        if (cmp_count == 0) {
            cmpA = node2;
        } else if (cmp_count == 1) {
            cmpB = node2;
        }
        ++cmp_count;
        if (cmp_count >= 2) {
            break;
        }
    }

    target_nodes[i] = node1;
    cmp_node_a[i] = cmpA;
    cmp_node_b[i] = cmpB;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct IsPositiveInt {
    __host__ __device__ bool operator()(int x) const { return x > 0; }
};

inline bool apply_knapsack_state_dominance(BDD *bdd, const int layer, EnumerationStats *stats,
                                           LayerDominanceContext *ctx) {
    if (ctx == NULL || ctx->next_sizes == NULL || ctx->next_offsets == NULL ||
        ctx->next_points == NULL) {
        return fail_layer_filter(ctx, "Invalid CUDA dominance layer context");
    }
    if (ctx->layer_min_weight == NULL || ctx->layer_single_parent_id == NULL ||
        ctx->layer_single_parent_arc == NULL) {
        return fail_layer_filter(
            ctx, "CUDA knapsack dominance filtering requires packed per-layer metadata");
    }

    const int next_nodes = ctx->next_sizes->size();
    if (next_nodes <= 1) {
        return true;
    }
    if (ctx->next_sizes->size() != static_cast<size_t>(next_nodes) ||
        ctx->next_offsets->size() != static_cast<size_t>(next_nodes + 1)) {
        return fail_layer_filter(ctx, "CUDA dominance vectors do not match current layer size");
    }
    if (ctx->layer_min_weight->size() != static_cast<size_t>(next_nodes) ||
        ctx->layer_single_parent_id->size() != static_cast<size_t>(next_nodes) ||
        ctx->layer_single_parent_arc->size() != static_cast<size_t>(next_nodes)) {
        return fail_layer_filter(ctx,
                                 "CUDA dominance metadata vectors do not match current layer size");
    }

    thrust::device_vector<int> d_active_nodes(next_nodes, 0);
    auto active_end = thrust::copy_if(
        thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(next_nodes),
        ctx->next_sizes->begin(), d_active_nodes.begin(), IsPositiveInt());
    const int active_size = active_end - d_active_nodes.begin();
    if (active_size <= 1) {
        return true;
    }
    d_active_nodes.resize(active_size);

    thrust::device_vector<int> d_keys(active_size, 0);
    thrust::gather(d_active_nodes.begin(), d_active_nodes.end(), ctx->layer_min_weight->begin(),
                   d_keys.begin());
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_active_nodes.begin(),
                        thrust::greater<int>());

    const int num_targets = active_size - 1;
    thrust::device_vector<int> d_target_nodes(num_targets, 0);
    thrust::device_vector<int> d_cmp_node_a(num_targets, -1);
    thrust::device_vector<int> d_cmp_node_b(num_targets, -1);

    ScopedCudaEventTimer build_pairs_timer("cudaEventCreate/build_knapsack_dominance_pairs_kernel",
                                           ctx->reason);
    if (!build_pairs_timer.ok()) {
        return false;
    }
    build_knapsack_dominance_pairs_kernel<<<ceil_div(num_targets, kThreadsPerBlock),
                                            kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_active_nodes.data()), active_size,
        thrust::raw_pointer_cast(ctx->layer_single_parent_id->data()),
        thrust::raw_pointer_cast(ctx->layer_single_parent_arc->data()),
        thrust::raw_pointer_cast(d_target_nodes.data()),
        thrust::raw_pointer_cast(d_cmp_node_a.data()),
        thrust::raw_pointer_cast(d_cmp_node_b.data()));
    if (!sync_kernel("build_knapsack_dominance_pairs_kernel", ctx->reason)) {
        return false;
    }
    if (!build_pairs_timer.finish_and_add(
            stats != NULL ? &stats->kernel_dominance_s : NULL,
            "cudaEventElapsedTime/build_knapsack_dominance_pairs_kernel")) {
        return false;
    }

    const int old_total = ctx->next_points->size() / NOBJS;
    if (old_total <= 0) {
        return true;
    }
    const int last_offset = (*(ctx->next_offsets))[next_nodes];
    if (last_offset != old_total) {
        return fail_layer_filter(ctx, "CUDA dominance offsets do not match points count");
    }

    thrust::device_vector<int> d_keep(old_total, 1);
    ScopedCudaEventTimer mark_knapsack_timer("cudaEventCreate/mark_dominated_knapsack_pairs_kernel",
                                             ctx->reason);
    if (!mark_knapsack_timer.ok()) {
        return false;
    }
    mark_dominated_knapsack_pairs_kernel<<<num_targets, kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_target_nodes.data()),
        thrust::raw_pointer_cast(d_cmp_node_a.data()),
        thrust::raw_pointer_cast(d_cmp_node_b.data()),
        thrust::raw_pointer_cast(ctx->next_offsets->data()),
        thrust::raw_pointer_cast(ctx->next_points->data()), num_targets,
        thrust::raw_pointer_cast(d_keep.data()));
    if (!sync_kernel("mark_dominated_knapsack_pairs_kernel", ctx->reason)) {
        return false;
    }
    if (!mark_knapsack_timer.finish_and_add(
            stats != NULL ? &stats->kernel_dominance_s : NULL,
            "cudaEventElapsedTime/mark_dominated_knapsack_pairs_kernel")) {
        return false;
    }

    thrust::device_vector<int> d_filtered_sizes(next_nodes, 0);
    ScopedCudaEventTimer recompute_sizes_timer("cudaEventCreate/recompute_sizes_from_keep_kernel",
                                               ctx->reason);
    if (!recompute_sizes_timer.ok()) {
        return false;
    }
    recompute_sizes_from_keep_kernel<<<ceil_div(next_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(ctx->next_offsets->data()),
        thrust::raw_pointer_cast(d_keep.data()), next_nodes,
        thrust::raw_pointer_cast(d_filtered_sizes.data()));
    if (!sync_kernel("recompute_sizes_from_keep_kernel", ctx->reason)) {
        return false;
    }
    if (!recompute_sizes_timer.finish_and_add(
            stats != NULL ? &stats->kernel_dominance_s : NULL,
            "cudaEventElapsedTime/recompute_sizes_from_keep_kernel")) {
        return false;
    }

    thrust::device_vector<int> d_filtered_offsets(next_nodes + 1, 0);
    thrust::exclusive_scan(d_filtered_sizes.begin(), d_filtered_sizes.end(),
                           d_filtered_offsets.begin());
    const int new_total = thrust::reduce(d_keep.begin(), d_keep.end(), 0);
    d_filtered_offsets[next_nodes] = new_total;

    thrust::device_vector<ObjType> d_filtered_points(new_total * NOBJS, 0);
    if (new_total > 0) {
        thrust::device_vector<int> d_keep_prefix(old_total, 0);
        thrust::exclusive_scan(d_keep.begin(), d_keep.end(), d_keep_prefix.begin());

        ScopedCudaEventTimer scatter_dom_timer(
            "cudaEventCreate/compact_alive_points_kernel_dominance", ctx->reason);
        if (!scatter_dom_timer.ok()) {
            return false;
        }
        compact_alive_points_kernel<<<ceil_div(old_total, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_keep.data()), thrust::raw_pointer_cast(d_keep_prefix.data()),
            thrust::raw_pointer_cast(ctx->next_points->data()),
            thrust::raw_pointer_cast(d_filtered_points.data()), old_total);
        if (!sync_kernel("compact_alive_points_kernel_dominance", ctx->reason)) {
            return false;
        }
        if (!scatter_dom_timer.finish_and_add(
                stats != NULL ? &stats->kernel_dominance_s : NULL,
                "cudaEventElapsedTime/compact_alive_points_kernel_dominance")) {
            return false;
        }
    }

    ctx->next_sizes->swap(d_filtered_sizes);
    ctx->next_offsets->swap(d_filtered_offsets);
    ctx->next_points->swap(d_filtered_points);

    if (stats != NULL && old_total > new_total) {
        stats->dominance_filtered_total += old_total - new_total;
    }
    return true;
}

} // namespace

void MultiobjEnum::filter_dominance_cuda(BDD *bdd, const int layer, const int problem_type,
                                         const int state_dominance, EnumerationStats *stats) {
    if (state_dominance <= 0) {
        return;
    }
    if (problem_type == 1) {
        filter_dominance_knapsack_cuda(bdd, layer, stats);
        return;
    }

    LayerDominanceContext *ctx = g_layer_dom_ctx;
    if (ctx != NULL && ctx->warned_non_knapsack != NULL && !*(ctx->warned_non_knapsack)) {
        *(ctx->warned_non_knapsack) = true;
    }
}

void MultiobjEnum::filter_dominance_knapsack_cuda(BDD *bdd, const int layer,
                                                  EnumerationStats *stats) {
    LayerDominanceContext *ctx = g_layer_dom_ctx;
    if (ctx == NULL) {
        return;
    }
    if (!apply_knapsack_state_dominance(bdd, layer, stats, ctx)) {
        ctx->failed = true;
    }
}

ParetoFrontier *enumerate_bdd_topdown(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      std::string *reason,
                                      long long max_candidate_points_per_batch) {
    if (bdd == NULL) {
        set_reason(reason, "BDD pointer is NULL");
        return NULL;
    }
    if (bdd->num_layers <= 0) {
        set_reason(reason, "BDD has zero layers");
        return NULL;
    }
    long long gpu_mem_baseline_used_bytes = 0;
    long long gpu_mem_peak_used_bytes = 0;
    long long gpu_mem_peak_reserved_bytes = 0;
    if (stats != NULL) {
        if (!capture_gpu_memory_used(reason, &gpu_mem_baseline_used_bytes)) {
            return NULL;
        }
        gpu_mem_peak_reserved_bytes = gpu_mem_baseline_used_bytes;
        stats->std_candidates_per_layer.assign(bdd->num_layers, 0.0);
        stats->std_frontier_survivors_per_layer.assign(bdd->num_layers, 0.0);
    }

    const int root_idx = bdd->get_root()->index;
    const int root_nodes = bdd->layers[0].size();
    if (root_idx < 0 || root_idx >= root_nodes) {
        set_reason(reason, "Invalid root index for layer 0");
        return NULL;
    }

    std::vector<PackedBDDLayer> packed_layers;
    pack_bdd_layers(bdd, packed_layers, false, false, problem_type, state_dominance, maximization,
                    stats);
    const bool pack_knapsack_meta = (problem_type == 1 && state_dominance > 0);

    thrust::host_vector<int> h_prev_sizes(root_nodes, 0);
    h_prev_sizes[root_idx] = 1;

    thrust::device_vector<int> d_prev_sizes = h_prev_sizes;
    thrust::device_vector<int> d_prev_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_prev_sizes.begin(), d_prev_sizes.end(), d_prev_offsets.begin());
    d_prev_offsets[root_nodes] = 1;

    thrust::device_vector<ObjType> d_prev_points(NOBJS, 0);
    bool warned_non_knapsack_dominance = false;

    for (int l = 1; l < bdd->num_layers; ++l) {
        const auto layer_begin = std::chrono::steady_clock::now();
        const int prev_nodes = bdd->layers[l - 1].size();
        const int next_nodes = bdd->layers[l].size();
        long long layer_candidates = 0;
        double layer_candidates_std = 0.0;
        if (d_prev_offsets.size() != static_cast<size_t>(prev_nodes + 1)) {
            set_reason(reason, "Previous offsets size does not match layer size");
            return NULL;
        }

        thrust::device_vector<int> d_next_sizes(next_nodes, 0);
        thrust::device_vector<int> d_next_offsets(next_nodes + 1, 0);
        thrust::device_vector<ObjType> d_next_points;
        thrust::device_vector<int> d_cand_counts(next_nodes, 0);

        PackedBDDLayer &packed = packed_layers[l];
        const int num_edges = packed.td_edge_src.size();
        if (num_edges > 0) {
            thrust::device_vector<int> d_edge_counts(num_edges, 0);
            thrust::device_vector<int> d_edge_offsets(num_edges + 1, 0);

            ScopedCudaEventTimer edge_counts_timer("cudaEventCreate/count_edge_candidates_kernel",
                                                   reason);
            if (!edge_counts_timer.ok()) {
                return NULL;
            }
            count_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock),
                                           kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(packed.td_edge_src.data()),
                thrust::raw_pointer_cast(d_prev_offsets.data()),
                thrust::raw_pointer_cast(d_edge_counts.data()), num_edges);
            if (!sync_kernel("count_edge_candidates_kernel", reason)) {
                return NULL;
            }
            if (!edge_counts_timer.finish_and_add(
                    stats != NULL ? &stats->kernel_expand_td_s : NULL,
                    "cudaEventElapsedTime/count_edge_candidates_kernel")) {
                return NULL;
            }

            thrust::exclusive_scan(d_edge_counts.begin(), d_edge_counts.end(),
                                   d_edge_offsets.begin());
            const int last_offset = d_edge_offsets[num_edges - 1];
            const int last_count = d_edge_counts[num_edges - 1];
            const int total_candidates = last_offset + last_count;
            layer_candidates = total_candidates;
            d_edge_offsets[num_edges] = total_candidates;

            if (total_candidates > 0) {
                thrust::device_vector<int> d_dst_blocks(next_nodes, 0);
                ScopedCudaEventTimer dst_counts_timer(
                    "cudaEventCreate/count_destination_candidates_kernel", reason);
                if (!dst_counts_timer.ok()) {
                    return NULL;
                }
                count_destination_candidates_kernel<<<ceil_div(next_nodes, kThreadsPerBlock),
                                                      kThreadsPerBlock>>>(
                    thrust::raw_pointer_cast(packed.td_in_edge_offsets.data()),
                    thrust::raw_pointer_cast(d_edge_offsets.data()), next_nodes,
                    thrust::raw_pointer_cast(d_cand_counts.data()),
                    thrust::raw_pointer_cast(d_dst_blocks.data()));
                if (!sync_kernel("count_destination_candidates_kernel", reason)) {
                    return NULL;
                }
                if (!dst_counts_timer.finish_and_add(
                        stats != NULL ? &stats->kernel_expand_td_s : NULL,
                        "cudaEventElapsedTime/count_destination_candidates_kernel")) {
                    return NULL;
                }
                layer_candidates_std = population_std_from_device_counts(d_cand_counts);

                if (total_candidates > max_candidate_points_per_batch) {
                    thrust::host_vector<int> h_in_edge_offsets = packed.td_in_edge_offsets;
                    thrust::host_vector<int> h_edge_offsets = d_edge_offsets;

                    d_next_points.clear();
                    d_next_sizes.assign(next_nodes, 0);

                    int dst_begin = 0;
                    while (dst_begin < next_nodes) {
                        int dst_end = dst_begin + 1;
                        long long batch_candidates =
                            static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_end]]) -
                            static_cast<long long>(h_edge_offsets[h_in_edge_offsets[dst_begin]]);

                        while (dst_end < next_nodes &&
                               batch_candidates < max_candidate_points_per_batch) {
                            const long long next_candidates =
                                static_cast<long long>(
                                    h_edge_offsets[h_in_edge_offsets[dst_end + 1]]) -
                                static_cast<long long>(
                                    h_edge_offsets[h_in_edge_offsets[dst_begin]]);
                            if (next_candidates > max_candidate_points_per_batch &&
                                batch_candidates > 0) {
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
                                               d_edge_counts.begin() + edge_end,
                                               d_batch_edge_offsets.begin());
                        const int batch_last_offset = d_batch_edge_offsets[batch_edges - 1];
                        const int batch_last_count = d_edge_counts[edge_end - 1];
                        const int batch_total_candidates = batch_last_offset + batch_last_count;
                        d_batch_edge_offsets[batch_edges] = batch_total_candidates;
                        if (batch_total_candidates <= 0) {
                            dst_begin = dst_end;
                            continue;
                        }

                        thrust::device_vector<ObjType> d_batch_cand_points(
                            batch_total_candidates * NOBJS, 0);
                        ScopedCudaEventTimer batch_expand_timer(
                            "cudaEventCreate/batch_materialize_edge_candidates_kernel", reason);
                        if (!batch_expand_timer.ok()) {
                            return NULL;
                        }
                        materialize_edge_candidates_kernel<<<
                            ceil_div(batch_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(packed.td_edge_src.data()) + edge_begin,
                            thrust::raw_pointer_cast(packed.td_edge_weights.data()) +
                                edge_begin * NOBJS,
                            thrust::raw_pointer_cast(d_batch_edge_offsets.data()),
                            thrust::raw_pointer_cast(d_edge_counts.data()) + edge_begin,
                            thrust::raw_pointer_cast(d_prev_offsets.data()),
                            thrust::raw_pointer_cast(d_prev_points.data()), batch_edges,
                            thrust::raw_pointer_cast(d_batch_cand_points.data()));
                        if (!sync_kernel("batch_materialize_edge_candidates_kernel", reason)) {
                            return NULL;
                        }
                        if (!batch_expand_timer.finish_and_add(
                                stats != NULL ? &stats->kernel_expand_td_s : NULL,
                                "cudaEventElapsedTime/batch_materialize_edge_candidates_kernel")) {
                            return NULL;
                        }
                        if (stats != NULL &&
                            !sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes,
                                                    &gpu_mem_peak_used_bytes,
                                                    &gpu_mem_peak_reserved_bytes)) {
                            return NULL;
                        }

                        thrust::device_vector<int> d_batch_sizes(batch_nodes, 0);
                        thrust::device_vector<int> d_batch_blocks(batch_nodes, 0);
                        count_destination_candidates_kernel<<<
                            ceil_div(batch_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                            thrust::raw_pointer_cast(d_batch_edge_offsets.data()), batch_nodes,
                            thrust::raw_pointer_cast(d_batch_sizes.data()),
                            thrust::raw_pointer_cast(d_batch_blocks.data()));
                        if (!sync_kernel("batch_count_destination_candidates_kernel", reason)) {
                            return NULL;
                        }

                        thrust::device_vector<int> d_batch_alive(batch_total_candidates, 0);
                        thrust::device_vector<int> d_batch_next_sizes(batch_nodes, 0);
                        const int batch_max_seg_size = thrust::reduce(
                            d_batch_sizes.begin(), d_batch_sizes.end(), 0, thrust::maximum<int>());
                        if (batch_max_seg_size > 0) {
                            thrust::device_vector<int> d_batch_block_offsets(batch_nodes + 1, 0);
                            thrust::exclusive_scan(d_batch_blocks.begin(), d_batch_blocks.end(),
                                                   d_batch_block_offsets.begin());
                            d_batch_block_offsets[batch_nodes] =
                                thrust::reduce(d_batch_blocks.begin(), d_batch_blocks.end(), 0);

                            const int batch_total_blocks = d_batch_block_offsets[batch_nodes];
                            if (batch_total_blocks > 0) {
                                ScopedCudaEventTimer batch_dom_timer(
                                    "cudaEventCreate/batch_mark_dominated_by_dst_dynamic_1d_kernel",
                                    reason);
                                if (!batch_dom_timer.ok()) {
                                    return NULL;
                                }
                                mark_dominated_by_dst_dynamic_1d_kernel<<<batch_total_blocks,
                                                                          kThreadsPerBlock>>>(
                                    thrust::raw_pointer_cast(d_batch_cand_points.data()),
                                    thrust::raw_pointer_cast(d_batch_in_offsets.data()),
                                    thrust::raw_pointer_cast(d_batch_edge_offsets.data()),
                                    thrust::raw_pointer_cast(d_batch_block_offsets.data()),
                                    batch_nodes, thrust::raw_pointer_cast(d_batch_alive.data()),
                                    thrust::raw_pointer_cast(d_batch_next_sizes.data()));
                                if (!sync_kernel("batch_mark_dominated_by_dst_dynamic_1d_kernel",
                                                 reason)) {
                                    return NULL;
                                }
                                if (!batch_dom_timer.finish_and_add(
                                        stats != NULL ? &stats->kernel_dominance_s : NULL,
                                        "cudaEventElapsedTime/"
                                        "batch_mark_dominated_by_dst_dynamic_1d_kernel")) {
                                    return NULL;
                                }
                            }
                        }

                        thrust::device_vector<int> d_batch_alive_prefix(batch_total_candidates, 0);
                        thrust::exclusive_scan(d_batch_alive.begin(), d_batch_alive.end(),
                                               d_batch_alive_prefix.begin());
                        const int batch_total_next =
                            thrust::reduce(d_batch_next_sizes.begin(), d_batch_next_sizes.end(), 0);

                        thrust::device_vector<ObjType> d_batch_next_points(batch_total_next * NOBJS,
                                                                           0);
                        if (batch_total_next > 0) {
                            ScopedCudaEventTimer batch_scatter_timer(
                                "cudaEventCreate/batch_compact_alive_points_kernel", reason);
                            if (!batch_scatter_timer.ok()) {
                                return NULL;
                            }
                            compact_alive_points_kernel<<<ceil_div(batch_total_candidates,
                                                                   kThreadsPerBlock),
                                                          kThreadsPerBlock>>>(
                                thrust::raw_pointer_cast(d_batch_alive.data()),
                                thrust::raw_pointer_cast(d_batch_alive_prefix.data()),
                                thrust::raw_pointer_cast(d_batch_cand_points.data()),
                                thrust::raw_pointer_cast(d_batch_next_points.data()),
                                batch_total_candidates);
                            if (!sync_kernel("batch_compact_alive_points_kernel", reason)) {
                                return NULL;
                            }
                            if (!batch_scatter_timer.finish_and_add(
                                    stats != NULL ? &stats->kernel_expand_td_s : NULL,
                                    "cudaEventElapsedTime/batch_compact_alive_points_kernel")) {
                                return NULL;
                            }

                            const size_t old_size = d_next_points.size();
                            d_next_points.resize(old_size + d_batch_next_points.size());
                            thrust::copy(d_batch_next_points.begin(), d_batch_next_points.end(),
                                         d_next_points.begin() + old_size);
                        }

                        thrust::copy(d_batch_next_sizes.begin(), d_batch_next_sizes.end(),
                                     d_next_sizes.begin() + dst_begin);

                        dst_begin = dst_end;
                    }

                    const int total_next =
                        thrust::reduce(d_next_sizes.begin(), d_next_sizes.end(), 0);
                    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(),
                                           d_next_offsets.begin());
                    d_next_offsets[next_nodes] = total_next;
                } else {
                    thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);

                    ScopedCudaEventTimer expand_timer(
                        "cudaEventCreate/materialize_edge_candidates_kernel", reason);
                    if (!expand_timer.ok()) {
                        return NULL;
                    }
                    materialize_edge_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock),
                                                         kThreadsPerBlock>>>(
                        thrust::raw_pointer_cast(packed.td_edge_src.data()),
                        thrust::raw_pointer_cast(packed.td_edge_weights.data()),
                        thrust::raw_pointer_cast(d_edge_offsets.data()),
                        thrust::raw_pointer_cast(d_edge_counts.data()),
                        thrust::raw_pointer_cast(d_prev_offsets.data()),
                        thrust::raw_pointer_cast(d_prev_points.data()), num_edges,
                        thrust::raw_pointer_cast(d_cand_points.data()));
                    if (!sync_kernel("materialize_edge_candidates_kernel", reason)) {
                        return NULL;
                    }
                    if (!expand_timer.finish_and_add(
                            stats != NULL ? &stats->kernel_expand_td_s : NULL,
                            "cudaEventElapsedTime/materialize_edge_candidates_kernel")) {
                        return NULL;
                    }
                    // Primary peak checkpoint: candidate points are fully materialized before
                    // dominance compaction.
                    if (stats != NULL &&
                        !sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes,
                                                &gpu_mem_peak_used_bytes,
                                                &gpu_mem_peak_reserved_bytes)) {
                        return NULL;
                    }

                    thrust::device_vector<int> d_alive(total_candidates, 0);
                    const int max_seg_size = thrust::reduce(
                        d_cand_counts.begin(), d_cand_counts.end(), 0, thrust::maximum<int>());
                    if (max_seg_size > 0) {
                        thrust::device_vector<int> d_block_offsets(next_nodes + 1, 0);
                        thrust::exclusive_scan(d_dst_blocks.begin(), d_dst_blocks.end(),
                                               d_block_offsets.begin());
                        d_block_offsets[next_nodes] =
                            thrust::reduce(d_dst_blocks.begin(), d_dst_blocks.end(), 0);
                        const int total_blocks = d_block_offsets[next_nodes];
                        if (total_blocks > 0) {
                            ScopedCudaEventTimer dom_timer(
                                "cudaEventCreate/mark_dominated_by_dst_dynamic_1d_kernel", reason);
                            if (!dom_timer.ok()) {
                                return NULL;
                            }
                            mark_dominated_by_dst_dynamic_1d_kernel<<<total_blocks,
                                                                      kThreadsPerBlock>>>(
                                thrust::raw_pointer_cast(d_cand_points.data()),
                                thrust::raw_pointer_cast(packed.td_in_edge_offsets.data()),
                                thrust::raw_pointer_cast(d_edge_offsets.data()),
                                thrust::raw_pointer_cast(d_block_offsets.data()), next_nodes,
                                thrust::raw_pointer_cast(d_alive.data()),
                                thrust::raw_pointer_cast(d_next_sizes.data()));
                            if (!sync_kernel("mark_dominated_by_dst_dynamic_1d_kernel", reason)) {
                                return NULL;
                            }
                            if (!dom_timer.finish_and_add(
                                    stats != NULL ? &stats->kernel_dominance_s : NULL,
                                    "cudaEventElapsedTime/"
                                    "mark_dominated_by_dst_dynamic_1d_kernel")) {
                                return NULL;
                            }
                        }
                    }

                    thrust::device_vector<int> d_alive_prefix(total_candidates, 0);
                    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
                    const int total_next =
                        thrust::reduce(d_next_sizes.begin(), d_next_sizes.end(), 0);

                    thrust::exclusive_scan(d_next_sizes.begin(), d_next_sizes.end(),
                                           d_next_offsets.begin());
                    d_next_offsets[next_nodes] = total_next;

                    d_next_points.resize(total_next * NOBJS);
                    if (total_next > 0) {
                        ScopedCudaEventTimer scatter_timer(
                            "cudaEventCreate/compact_alive_points_kernel", reason);
                        if (!scatter_timer.ok()) {
                            return NULL;
                        }
                        compact_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock),
                                                      kThreadsPerBlock>>>(
                            thrust::raw_pointer_cast(d_alive.data()),
                            thrust::raw_pointer_cast(d_alive_prefix.data()),
                            thrust::raw_pointer_cast(d_cand_points.data()),
                            thrust::raw_pointer_cast(d_next_points.data()), total_candidates);
                        if (!sync_kernel("compact_alive_points_kernel", reason)) {
                            return NULL;
                        }
                        if (!scatter_timer.finish_and_add(
                                stats != NULL ? &stats->kernel_expand_td_s : NULL,
                                "cudaEventElapsedTime/compact_alive_points_kernel")) {
                            return NULL;
                        }
                    }
                }
            }
        }

        if (state_dominance > 0) {
            LayerDominanceContext dom_ctx;
            dom_ctx.next_sizes = &d_next_sizes;
            dom_ctx.next_offsets = &d_next_offsets;
            dom_ctx.next_points = &d_next_points;
            dom_ctx.layer_min_weight = pack_knapsack_meta ? &packed.min_weight : NULL;
            dom_ctx.layer_single_parent_id = pack_knapsack_meta ? &packed.single_parent_id : NULL;
            dom_ctx.layer_single_parent_arc = pack_knapsack_meta ? &packed.single_parent_arc : NULL;
            dom_ctx.reason = reason;
            dom_ctx.warned_non_knapsack = &warned_non_knapsack_dominance;
            dom_ctx.failed = false;

            g_layer_dom_ctx = &dom_ctx;
            clock_t init = clock();
            MultiobjEnum::filter_dominance_cuda(bdd, l, problem_type, state_dominance, stats);
            if (stats != NULL) {
                stats->cpu_state_dominance_s +=
                    static_cast<double>(clock() - init) / CLOCKS_PER_SEC;
            }
            g_layer_dom_ctx = NULL;

            if (dom_ctx.failed) {
                return NULL;
            }
        }
        if (stats != NULL) {
            const long long layer_survivors = d_next_points.size() / NOBJS;
            const double layer_survivors_std = population_std_from_device_counts(d_next_sizes);
            stats->work_candidates_total += layer_candidates;
            stats->work_candidates_peak = std::max(stats->work_candidates_peak, layer_candidates);
            stats->work_frontier_survivors_total += layer_survivors;
            stats->work_frontier_peak_points =
                std::max(stats->work_frontier_peak_points, layer_survivors);
            stats->std_candidates_per_layer[l] = layer_candidates_std;
            stats->std_frontier_survivors_per_layer[l] = layer_survivors_std;
            if (!sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes,
                                        &gpu_mem_peak_used_bytes, &gpu_mem_peak_reserved_bytes)) {
                return NULL;
            }
        }

        d_prev_offsets.swap(d_next_offsets);
        d_prev_points.swap(d_next_points);
    }

    const int term_idx = bdd->get_terminal()->index;
    thrust::host_vector<int> h_offsets = d_prev_offsets;
    if (term_idx < 0 || term_idx + 1 >= h_offsets.size()) {
        set_reason(reason, "Invalid terminal index after CUDA enumeration");
        return NULL;
    }

    const int begin = h_offsets[term_idx];
    const int end = h_offsets[term_idx + 1];
    const int num_points = std::max(0, end - begin);

    ParetoFrontier *frontier = new ParetoFrontier;
    frontier->sols.resize(num_points * NOBJS, 0);
    if (num_points > 0) {
        thrust::host_vector<ObjType> h_points = d_prev_points;
        std::copy(h_points.begin() + begin * NOBJS, h_points.begin() + end * NOBJS,
                  frontier->sols.begin());
    }

    if (reason != NULL) {
        reason->clear();
    }
    if (stats != NULL) {
        if (!sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes, &gpu_mem_peak_used_bytes,
                                    &gpu_mem_peak_reserved_bytes)) {
            return NULL;
        }
        stats->gpu_mem_peak_used_bytes = gpu_mem_peak_used_bytes;
        stats->gpu_mem_peak_reserved_bytes = gpu_mem_peak_reserved_bytes;
        stats->kernel_total_s = stats->kernel_expand_td_s + stats->kernel_dominance_s;
    }
    return frontier;
}

// ---------------------------------------------------------------
// MDD GPU Top-Down Layer Expansion Wrapper
// ---------------------------------------------------------------

bool topdown_expand_mdd_layer(
    const PackedMDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch, long long *total_candidates_out,
    long long *total_next_out, double *std_candidates_out, double *std_survivors_out,
    long long *gpu_mem_baseline_used_bytes, long long *gpu_mem_peak_used_bytes,
    long long *gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.td_in_edge_offsets, packed_layer.td_edge_src, packed_layer.td_edge_weights,
        packed_layer.td_num_edges, packed_layer.num_nodes, d_prev_offsets, d_prev_points,
        d_next_sizes, d_next_offsets, d_next_points, reason, max_candidate_points_per_batch,
        total_candidates_out, total_next_out, std_candidates_out, std_survivors_out,
        gpu_mem_baseline_used_bytes, gpu_mem_peak_used_bytes, gpu_mem_peak_reserved_bytes);
}

// ---------------------------------------------------------------
// BDD GPU Top-Down Layer Expansion Wrapper
// ---------------------------------------------------------------

bool topdown_expand_bdd_layer(
    const PackedBDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch, long long *total_candidates_out,
    long long *total_next_out, double *std_candidates_out, double *std_survivors_out,
    long long *gpu_mem_baseline_used_bytes, long long *gpu_mem_peak_used_bytes,
    long long *gpu_mem_peak_reserved_bytes) {

    return expand_layer_frontiers(
        packed_layer.td_in_edge_offsets, packed_layer.td_edge_src, packed_layer.td_edge_weights,
        packed_layer.td_num_edges, packed_layer.num_nodes, d_prev_offsets, d_prev_points,
        d_next_sizes, d_next_offsets, d_next_points, reason, max_candidate_points_per_batch,
        total_candidates_out, total_next_out, std_candidates_out, std_survivors_out,
        gpu_mem_baseline_used_bytes, gpu_mem_peak_used_bytes, gpu_mem_peak_reserved_bytes);
}
