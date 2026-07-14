// ----------------------------------------------------------
// CUDA Coupled (Dynamic Layer Cutset) Enumeration for MDD
// ----------------------------------------------------------

#include "enum_types.cuh"

#include <algorithm>
#include <chrono>
#include <climits>
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

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct HasBothFrontiers {
    const int *td_off;
    const int *bu_off;

    __host__ __device__ bool operator()(int node) const {
        return (td_off[node + 1] - td_off[node]) > 0 && (bu_off[node + 1] - bu_off[node]) > 0;
    }
};

struct FrontierSumScore {
    const int *td_off;
    const int *bu_off;

    __host__ __device__ long long operator()(int node) const {
        return static_cast<long long>(td_off[node + 1] - td_off[node]) +
               static_cast<long long>(bu_off[node + 1] - bu_off[node]);
    }
};

// ---------------------------------------------------------------
// Cutset coupling kernels for MDD
// ---------------------------------------------------------------

__global__ void compact_alive_points_kernel(const int *alive, const int *prefix,
                                            const ObjType *in_pts, ObjType *out_pts, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !alive[i])
        return;
    const int oi = prefix[i];
    for (int o = 0; o < NOBJS; ++o)
        out_pts[oi * NOBJS + o] = in_pts[i * NOBJS + o];
}

// Cartesian product: for each node, td[i] + bu[j] for all pairs
__global__ void materialize_cutset_products_kernel(const ObjType *td_pts, const int *td_off,
                                                   const ObjType *bu_pts, const int *bu_off,
                                                   const int *td_sz, const int *bu_sz,
                                                   const int *prod_off, int num_nodes,
                                                   ObjType *out) {
    const int node = blockIdx.x;
    if (node >= num_nodes)
        return;
    const int ts = td_sz[node], bs = bu_sz[node];
    const int total = ts * bs;
    if (total <= 0)
        return;
    const int tbase = td_off[node], bbase = bu_off[node];
    const int obase = prod_off[node];

    for (int p = threadIdx.x; p < total; p += blockDim.x) {
        const int ti = p / bs, bi = p % bs;
        const int oi = obase + p;
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            out[oi * NOBJS + o] =
                td_pts[(tbase + ti) * NOBJS + o] + bu_pts[(bbase + bi) * NOBJS + o];
    }
}

// For each node (indexed via nz_nodes), compute its ideal/utopia product point:
// ub[o] = max_td[o] + max_bu[o]. Every actual product td[i]+bu[j] of that node is
// componentwise <= this point, so if a frontier point dominates-or-ties ub, it
// dominates-or-ties every product the node could ever produce.
__global__ void compute_node_ub_kernel(const ObjType *td_pts, const int *td_off,
                                       const ObjType *bu_pts, const int *bu_off,
                                       const int *nz_nodes, int num_nz_nodes, ObjType *ub_out) {
    const int j = blockIdx.x;
    if (j >= num_nz_nodes)
        return;
    const int node = nz_nodes[j];
    const int tbeg = td_off[node], tend = td_off[node + 1];
    const int bbeg = bu_off[node], bend = bu_off[node + 1];

    ObjType local_max_td[NOBJS];
    ObjType local_max_bu[NOBJS];
#pragma unroll
    for (int o = 0; o < NOBJS; ++o) {
        local_max_td[o] = INT_MIN;
        local_max_bu[o] = INT_MIN;
    }
    for (int i = tbeg + threadIdx.x; i < tend; i += blockDim.x) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            local_max_td[o] = max(local_max_td[o], td_pts[i * NOBJS + o]);
    }
    for (int i = bbeg + threadIdx.x; i < bend; i += blockDim.x) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            local_max_bu[o] = max(local_max_bu[o], bu_pts[i * NOBJS + o]);
    }

    __shared__ ObjType sh_td[kThreadsPerBlock * NOBJS];
    __shared__ ObjType sh_bu[kThreadsPerBlock * NOBJS];
#pragma unroll
    for (int o = 0; o < NOBJS; ++o) {
        sh_td[threadIdx.x * NOBJS + o] = local_max_td[o];
        sh_bu[threadIdx.x * NOBJS + o] = local_max_bu[o];
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                sh_td[threadIdx.x * NOBJS + o] =
                    max(sh_td[threadIdx.x * NOBJS + o], sh_td[(threadIdx.x + stride) * NOBJS + o]);
                sh_bu[threadIdx.x * NOBJS + o] =
                    max(sh_bu[threadIdx.x * NOBJS + o], sh_bu[(threadIdx.x + stride) * NOBJS + o]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            ub_out[j * NOBJS + o] = sh_td[o] + sh_bu[o];
    }
}

// Global dominance filter across all points (single flat array)
__global__ void mark_globally_dominated_kernel(const ObjType *points, int num_points, int *alive) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_points);

    ObjType pi[NOBJS];
    if (valid) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            pi[o] = points[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_points; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_points) {
#pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = points[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_points - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                const int gj = jb + jj;
                if (gj == i)
                    continue;
                bool ge = true, strict = false;
#pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= pi[o]);
                    strict = strict || (sh[jj * NOBJS + o] > pi[o]);
                }
                if (ge && (strict || gj < i))
                    dom = true;
            }
        }
        __syncthreads();
    }
    if (valid)
        alive[i] = dom ? 0 : 1;
}

// Like mark_dominated_by_frontier_kernel, but equality also removes candidates.
// This keeps duplicates from being appended when two already-clean frontiers are
// merged and an identical point exists in the old frontier.
__global__ void mark_dominated_or_equal_by_frontier_kernel(const ObjType *candidates, int num_cand,
                                                           const ObjType *frontier,
                                                           int num_frontier, int *alive) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_cand);

    ObjType ci[NOBJS];
    if (valid) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            ci[o] = candidates[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_frontier; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_frontier) {
#pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = frontier[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_frontier - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                bool ge = true;
#pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= ci[o]);
                }
                if (ge)
                    dom = true;
            }
        }
        __syncthreads();
    }
    if (valid)
        alive[i] = dom ? 0 : 1;
}

// Check which frontier points are dominated by any of the new candidates
// alive[i] = 0 if some candidate strictly dominates frontier[i], else 1.
__global__ void mark_frontier_dominated_by_batch_kernel(const ObjType *frontier, int num_frontier,
                                                        const ObjType *candidates, int num_cand,
                                                        int *alive) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = (i < num_frontier);

    ObjType fi[NOBJS];
    if (valid) {
#pragma unroll
        for (int o = 0; o < NOBJS; ++o)
            fi[o] = frontier[i * NOBJS + o];
    }

    bool dom = false;
    __shared__ ObjType sh[kThreadsPerBlock * NOBJS];
    for (int jb = 0; jb < num_cand; jb += blockDim.x) {
        const int jl = jb + threadIdx.x;
        if (jl < num_cand) {
#pragma unroll
            for (int o = 0; o < NOBJS; ++o)
                sh[threadIdx.x * NOBJS + o] = candidates[jl * NOBJS + o];
        }
        __syncthreads();
        if (valid && !dom) {
            const int tc = min(num_cand - jb, (int)blockDim.x);
            for (int jj = 0; jj < tc && !dom; ++jj) {
                bool ge = true, strict = false;
#pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    ge = ge && (sh[jj * NOBJS + o] >= fi[o]);
                    strict = strict || (sh[jj * NOBJS + o] > fi[o]);
                }
                if (ge && strict)
                    dom = true;
            }
        }
        __syncthreads();
    }
    if (valid)
        alive[i] = dom ? 0 : 1;
}

// Removed functors for global sorting since they break edge_offsets mapping

bool compact_points_by_alive(thrust::device_vector<ObjType> &points, int &num_points,
                             const thrust::device_vector<int> &alive, int live_points,
                             const char *kernel_name, std::string *reason) {
    if (live_points == num_points) {
        return true;
    }

    thrust::device_vector<int> prefix(num_points, 0);
    thrust::exclusive_scan(alive.begin(), alive.end(), prefix.begin());

    thrust::device_vector<ObjType> compacted(live_points * NOBJS);
    if (live_points > 0) {
        compact_alive_points_kernel<<<ceil_div(num_points, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(alive.data()), thrust::raw_pointer_cast(prefix.data()),
            thrust::raw_pointer_cast(points.data()), thrust::raw_pointer_cast(compacted.data()),
            num_points);
        if (!sync_kernel(kernel_name, reason))
            return false;
    }

    points.swap(compacted);
    num_points = live_points;
    return true;
}

bool self_prune_points(thrust::device_vector<ObjType> &points, int &num_points,
                       std::string *reason) {
    if (num_points <= 1) {
        return true;
    }

    thrust::device_vector<int> alive(num_points, 0);
    mark_globally_dominated_kernel<<<ceil_div(num_points, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(points.data()), num_points,
        thrust::raw_pointer_cast(alive.data()));
    if (!sync_kernel("local_self_prune", reason))
        return false;

    const int live_points = thrust::reduce(alive.begin(), alive.end(), 0);
    return compact_points_by_alive(points, num_points, alive, live_points,
                                   "local_self_prune_compact", reason);
}

bool merge_clean_points_into_frontier(thrust::device_vector<ObjType> &frontier, int &frontier_size,
                                      thrust::device_vector<ObjType> &incoming, int &incoming_size,
                                      std::string *reason) {
    if (incoming_size <= 0) {
        return true;
    }

    if (frontier_size <= 0) {
        frontier.swap(incoming);
        frontier_size = incoming_size;
        incoming_size = 0;
        return true;
    }

    thrust::device_vector<int> alive_new(incoming_size, 0);
    mark_dominated_or_equal_by_frontier_kernel<<<ceil_div(incoming_size, kThreadsPerBlock),
                                                 kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(incoming.data()), incoming_size,
        thrust::raw_pointer_cast(frontier.data()), frontier_size,
        thrust::raw_pointer_cast(alive_new.data()));
    if (!sync_kernel("merge_filter_new", reason))
        return false;

    const int kept_new = thrust::reduce(alive_new.begin(), alive_new.end(), 0);
    if (!compact_points_by_alive(incoming, incoming_size, alive_new, kept_new, "merge_compact_new",
                                 reason)) {
        return false;
    }
    if (incoming_size <= 0) {
        return true;
    }

    int kept_frontier = frontier_size;
    thrust::device_vector<int> alive_old(frontier_size, 0);
    mark_frontier_dominated_by_batch_kernel<<<ceil_div(frontier_size, kThreadsPerBlock),
                                              kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(frontier.data()), frontier_size,
        thrust::raw_pointer_cast(incoming.data()), incoming_size,
        thrust::raw_pointer_cast(alive_old.data()));
    if (!sync_kernel("merge_filter_old", reason))
        return false;

    kept_frontier = thrust::reduce(alive_old.begin(), alive_old.end(), 0);
    if (!compact_points_by_alive(frontier, frontier_size, alive_old, kept_frontier,
                                 "merge_compact_old", reason)) {
        return false;
    }

    const int new_total = frontier_size + incoming_size;
    frontier.resize(new_total * NOBJS);
    thrust::copy_n(incoming.begin(), incoming_size * NOBJS,
                   frontier.begin() + frontier_size * NOBJS);
    frontier_size = new_total;
    incoming.clear();
    incoming_size = 0;
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------
// expand_layer_frontiers: runs expansion kernels for one MDD layer.
// Works identically for top-down and bottom-up.
// ---------------------------------------------------------------

ParetoFrontier *couple_cutsets_cuda(int cutset_nodes,
                                    const thrust::device_vector<int> &d_td_offsets,
                                    const thrust::device_vector<ObjType> &d_td_points,
                                    const thrust::device_vector<int> &d_bu_offsets,
                                    const thrust::device_vector<ObjType> &d_bu_points,
                                    EnumerationStats *stats, std::string *reason,
                                    long long gpu_max_prod) {

    clock_t t0 = clock();

    thrust::host_vector<int> h_td_off = d_td_offsets;
    thrust::host_vector<int> h_bu_off = d_bu_offsets;

    long long total_products = 0;
    for (int i = 0; i < cutset_nodes; ++i) {
        int ts = h_td_off[i + 1] - h_td_off[i];
        int bs = h_bu_off[i + 1] - h_bu_off[i];
        long long p = (long long)ts * bs;
        total_products += p;
    }
    if (stats != NULL) {
        stats->work_join_products_total += total_products;
    }

    // Collect nonzero nodes and sort on device by CPU-like cutset priority:
    // td frontier size + bu frontier size (largest first).
    thrust::device_vector<int> d_all_nodes(cutset_nodes);
    thrust::copy(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(cutset_nodes),
                 d_all_nodes.begin());

    thrust::device_vector<int> d_nz_nodes(cutset_nodes);
    HasBothFrontiers has_both{thrust::raw_pointer_cast(d_td_offsets.data()),
                              thrust::raw_pointer_cast(d_bu_offsets.data())};
    auto nz_end =
        thrust::copy_if(d_all_nodes.begin(), d_all_nodes.end(), d_nz_nodes.begin(), has_both);
    d_nz_nodes.resize(static_cast<int>(nz_end - d_nz_nodes.begin()));

    thrust::device_vector<long long> d_nz_scores(d_nz_nodes.size());
    FrontierSumScore score_fn{thrust::raw_pointer_cast(d_td_offsets.data()),
                              thrust::raw_pointer_cast(d_bu_offsets.data())};
    thrust::transform(d_nz_nodes.begin(), d_nz_nodes.end(), d_nz_scores.begin(), score_fn);
    thrust::sort_by_key(d_nz_scores.begin(), d_nz_scores.end(), d_nz_nodes.begin(),
                        thrust::greater<long long>());

    // Ideal/utopia product point per node, in the same sorted order as d_nz_nodes:
    // ub[j] = max_td[node_j] + max_bu[node_j]. Used to skip whole nodes below once
    // the running frontier already dominates-or-ties everything that node could
    // ever produce (improvements.md 3.1).
    thrust::device_vector<ObjType> d_ub_points(d_nz_nodes.size() * NOBJS, 0);
    if (!d_nz_nodes.empty()) {
        compute_node_ub_kernel<<<static_cast<int>(d_nz_nodes.size()), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_td_points.data()), thrust::raw_pointer_cast(d_td_offsets.data()),
            thrust::raw_pointer_cast(d_bu_points.data()), thrust::raw_pointer_cast(d_bu_offsets.data()),
            thrust::raw_pointer_cast(d_nz_nodes.data()), static_cast<int>(d_nz_nodes.size()),
            thrust::raw_pointer_cast(d_ub_points.data()));
        if (!sync_kernel("compute_node_ub", reason))
            return NULL;
    }

    thrust::host_vector<int> h_nz_nodes = d_nz_nodes;
    std::vector<int> nz_nodes(h_nz_nodes.begin(), h_nz_nodes.end());

    // Running frontier stored on device
    thrust::device_vector<ObjType> d_frontier_pts;
    int frontier_size = 0;

    const long long MAX_BATCH_PRODUCTS = gpu_max_prod;

    int batch_begin = 0;
    while (batch_begin < static_cast<int>(nz_nodes.size())) {
        long long bprod_total = 0;
        int batch_end = batch_begin;
        while (batch_end < static_cast<int>(nz_nodes.size())) {
            const int ni = nz_nodes[batch_end];
            const long long p = static_cast<long long>(h_td_off[ni + 1] - h_td_off[ni]) *
                                static_cast<long long>(h_bu_off[ni + 1] - h_bu_off[ni]);
            if (bprod_total + p > MAX_BATCH_PRODUCTS && bprod_total > 0)
                break;
            bprod_total += p;
            ++batch_end;
        }
        if (bprod_total == 0) {
            batch_begin = batch_end;
            continue;
        }

        const int batch_node_count = batch_end - batch_begin;
        std::vector<int> h_btd_off(batch_node_count), h_bbu_off(batch_node_count);
        std::vector<int> h_btd_sz(batch_node_count), h_bbu_sz(batch_node_count);
        std::vector<int> h_bprod_off(batch_node_count + 1, 0);
        for (int j = 0; j < batch_node_count; ++j) {
            const int ni = nz_nodes[batch_begin + j];
            h_btd_off[j] = h_td_off[ni];
            h_bbu_off[j] = h_bu_off[ni];
            h_btd_sz[j] = h_td_off[ni + 1] - h_td_off[ni];
            h_bbu_sz[j] = h_bu_off[ni + 1] - h_bu_off[ni];
            h_bprod_off[j + 1] = h_bprod_off[j] + h_btd_sz[j] * h_bbu_sz[j];
        }

        const int batch_product_count = h_bprod_off[batch_node_count];
        if (stats != NULL) {
            stats->work_candidates_total += batch_product_count;
        }

        thrust::device_vector<int> d_btd_off(h_btd_off.begin(), h_btd_off.end());
        thrust::device_vector<int> d_bbu_off(h_bbu_off.begin(), h_bbu_off.end());
        thrust::device_vector<int> d_btd_sz(h_btd_sz.begin(), h_btd_sz.end());
        thrust::device_vector<int> d_bbu_sz(h_bbu_sz.begin(), h_bbu_sz.end());
        thrust::device_vector<int> d_bprod_off(h_bprod_off.begin(), h_bprod_off.end());

        thrust::device_vector<ObjType> d_batch_points(batch_product_count * NOBJS, 0);
        materialize_cutset_products_kernel<<<batch_node_count, kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_td_points.data()),
            thrust::raw_pointer_cast(d_btd_off.data()),
            thrust::raw_pointer_cast(d_bu_points.data()),
            thrust::raw_pointer_cast(d_bbu_off.data()), thrust::raw_pointer_cast(d_btd_sz.data()),
            thrust::raw_pointer_cast(d_bbu_sz.data()), thrust::raw_pointer_cast(d_bprod_off.data()),
            batch_node_count, thrust::raw_pointer_cast(d_batch_points.data()));
        if (!sync_kernel("cart_batch", reason))
            return NULL;

        int batch_frontier_size = batch_product_count;
        if (frontier_size > 0) {
            thrust::device_vector<int> alive_batch(batch_frontier_size, 0);
            mark_dominated_or_equal_by_frontier_kernel<<<
                ceil_div(batch_frontier_size, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_batch_points.data()), batch_frontier_size,
                thrust::raw_pointer_cast(d_frontier_pts.data()), frontier_size,
                thrust::raw_pointer_cast(alive_batch.data()));
            if (!sync_kernel("batch_filter_global", reason))
                return NULL;

            const int kept_batch = thrust::reduce(alive_batch.begin(), alive_batch.end(), 0);
            if (!compact_points_by_alive(d_batch_points, batch_frontier_size, alive_batch,
                                         kept_batch, "batch_filter_global_compact", reason)) {
                return NULL;
            }
        }

        if (!self_prune_points(d_batch_points, batch_frontier_size, reason)) {
            return NULL;
        }
        if (stats != NULL) {
            stats->work_frontier_survivors_total += batch_frontier_size;
        }

        // Incremental ideal-point check (improvements.md 3.1 fix): a still-remaining
        // node's ub point only needs testing against points *appended* to the running
        // frontier since it was last checked, not the whole frontier. d_batch_points at
        // this point is exactly this batch's clean, frontier-filtered, self-pruned
        // survivor set -- precisely what's about to be appended (merge_clean_points_
        // into_frontier's own internal vs-frontier filter below is then a no-op, since
        // nothing has touched d_frontier_pts since the filter already applied above).
        //
        // This is exact, not a heuristic: if some now-superseded frontier point f_old
        // dominated-or-tied a node's ub, f_old can only have been removed by a
        // strictly-dominating replacement f_new (see mark_frontier_dominated_by_batch_
        // kernel), and f_new >= f_old >= ub componentwise, so f_new also dominates-or-
        // ties ub. Every still-remaining node was already checked against every earlier
        // batch's survivors (by induction from batch 0), so checking against just this
        // batch's survivors here extends that same guarantee. Cost per batch drops from
        // O(remaining x frontier_size) to O(remaining x batch_survivors) -- the former
        // grows unboundedly over the run, the latter does not.
        if (batch_frontier_size > 0 && batch_end < static_cast<int>(nz_nodes.size())) {
            const int remaining = static_cast<int>(nz_nodes.size()) - batch_end;
            thrust::device_vector<int> ub_alive(remaining, 0);
            mark_dominated_or_equal_by_frontier_kernel<<<ceil_div(remaining, kThreadsPerBlock),
                                                          kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_ub_points.data()) + batch_end * NOBJS, remaining,
                thrust::raw_pointer_cast(d_batch_points.data()), batch_frontier_size,
                thrust::raw_pointer_cast(ub_alive.data()));
            if (!sync_kernel("ub_prune_check", reason))
                return NULL;

            const int kept = thrust::reduce(ub_alive.begin(), ub_alive.end(), 0);
            if (kept < remaining) {
                thrust::device_vector<ObjType> ub_suffix(d_ub_points.begin() + batch_end * NOBJS,
                                                         d_ub_points.end());
                int suffix_n = remaining;
                if (!compact_points_by_alive(ub_suffix, suffix_n, ub_alive, kept,
                                             "ub_prune_compact", reason)) {
                    return NULL;
                }
                d_ub_points.resize(batch_end * NOBJS + kept * NOBJS);
                thrust::copy(ub_suffix.begin(), ub_suffix.end(),
                            d_ub_points.begin() + batch_end * NOBJS);

                thrust::host_vector<int> h_alive = ub_alive;
                int w = batch_end;
                for (int r = batch_end; r < static_cast<int>(nz_nodes.size()); ++r) {
                    if (h_alive[r - batch_end]) {
                        nz_nodes[w++] = nz_nodes[r];
                    }
                }
                nz_nodes.resize(w);
            }
        }

        if (!merge_clean_points_into_frontier(d_frontier_pts, frontier_size, d_batch_points,
                                              batch_frontier_size, reason)) {
            return NULL;
        }
        if (stats != NULL) {
            stats->work_frontier_peak_points =
                std::max(stats->work_frontier_peak_points, static_cast<long long>(frontier_size));
        }

        batch_begin = batch_end;
    }
    clock_t t1 = clock();
    if (stats != NULL) {
        stats->wall_join_s += (double)(t1 - t0) / CLOCKS_PER_SEC;
    }

    // Copy final frontier to host
    ParetoFrontier *frontier = new ParetoFrontier;
    frontier->sols.resize(frontier_size * NOBJS, 0);
    if (frontier_size > 0) {
        thrust::host_vector<ObjType> h_res(d_frontier_pts.begin(),
                                           d_frontier_pts.begin() + frontier_size * NOBJS);
        std::copy(h_res.begin(), h_res.end(), frontier->sols.begin());
    }

    if (reason)
        reason->clear();
    return frontier;
}
