// ----------------------------------------------------------
// CUDA enumeration public entry points & orchestration
// ----------------------------------------------------------

#include "../../mdd/mdd.hpp"
#include "../multiobj_enum.hpp"
#include "cuda_wrappers.hpp"
#include "enum_types.cuh"

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

ParetoFrontier *enumerate_bdd_topdown(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      std::string *reason, long long gpu_batch_size);

ParetoFrontier *enumerate_mdd_topdown(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                      long long gpu_batch_size);

ParetoFrontier *enumerate_mdd_coupled(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                      long long gpu_batch_size, long long gpu_max_prod);

namespace {

inline bool cuda_enumeration_available(std::string *reason) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return set_reason(reason,
                          std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
    }
    if (device_count <= 0) {
        return set_reason(reason, "No CUDA device found");
    }
    return true;
}

inline bool prepare_cuda_device(std::string *reason) {
    if (!cuda_enumeration_available(reason)) {
        return false;
    }
    return cuda_ok(cudaSetDevice(0), "cudaSetDevice", reason);
}

} // namespace

bool topdown_cuda_available(std::string *reason) { return cuda_enumeration_available(reason); }

bool coupled_cuda_available(std::string *reason) { return cuda_enumeration_available(reason); }

ParetoFrontier *topdown_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                       const int state_dominance, EnumerationStats *stats,
                                       std::string *reason, long long gpu_batch_size) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_bdd_topdown(bdd, maximization, problem_type, state_dominance, stats, reason,
                                 gpu_batch_size);
}

ParetoFrontier *topdown_mdd_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                           long long gpu_batch_size) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_mdd_topdown(mdd, stats, reason, gpu_batch_size);
}

ParetoFrontier *coupled_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                       long long gpu_batch_size, long long gpu_max_prod) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    return enumerate_mdd_coupled(mdd, stats, reason, gpu_batch_size, gpu_max_prod);
}

// ----------------------------------------------------------
// MDD Packing Helper Implementation
// ----------------------------------------------------------

void pack_mdd_layers(MDD *mdd, std::vector<PackedMDDLayer> &packed, bool pack_bottom_up,
                     bool pack_heuristic) {
    const int num_layers = mdd->num_layers;
    packed.resize(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        const int num_nodes = mdd->layers[l].size();
        packed[l].num_nodes = num_nodes;

        // Top-down: incoming arcs
        if (l > 0) {
            std::vector<int> h_off(num_nodes + 1, 0);
            for (int d = 0; d < num_nodes; ++d) {
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->in_arcs_list.size();
            }
            const int num_edges = h_off[num_nodes];
            std::vector<int> h_src(num_edges);
            std::vector<ObjType> h_wt(num_edges * NOBJS);
            int idx = 0;
            for (int d = 0; d < num_nodes; ++d) {
                for (MDDArc *a : mdd->layers[l][d]->in_arcs_list) {
                    h_src[idx] = a->tail->index;
                    for (int o = 0; o < NOBJS; ++o) {
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    }
                    ++idx;
                }
            }
            packed[l].td_num_edges = num_edges;
            packed[l].td_in_edge_offsets = h_off;
            packed[l].td_edge_src = h_src;
            packed[l].td_edge_weights = h_wt;
        } else {
            packed[l].td_num_edges = 0;
        }

        // Bottom-up: outgoing arcs
        if (pack_bottom_up && l < num_layers - 1) {
            std::vector<int> h_off(num_nodes + 1, 0);
            for (int d = 0; d < num_nodes; ++d) {
                h_off[d + 1] = h_off[d] + mdd->layers[l][d]->out_arcs_list.size();
            }
            const int num_edges = h_off[num_nodes];
            std::vector<int> h_src(num_edges);
            std::vector<ObjType> h_wt(num_edges * NOBJS);
            int idx = 0;
            for (int d = 0; d < num_nodes; ++d) {
                for (MDDArc *a : mdd->layers[l][d]->out_arcs_list) {
                    h_src[idx] = a->head->index;
                    for (int o = 0; o < NOBJS; ++o) {
                        h_wt[idx * NOBJS + o] = a->weights[o];
                    }
                    ++idx;
                }
            }
            packed[l].bu_num_edges = num_edges;
            packed[l].bu_in_edge_offsets = h_off;
            packed[l].bu_edge_src = h_src;
            packed[l].bu_edge_weights = h_wt;
        } else {
            packed[l].bu_num_edges = 0;
        }

        // Heuristic arc counts
        if (pack_heuristic) {
            std::vector<int> h_out(num_nodes), h_in(num_nodes);
            for (int d = 0; d < num_nodes; ++d) {
                h_out[d] = mdd->layers[l][d]->out_arcs_list.size();
                h_in[d] = mdd->layers[l][d]->in_arcs_list.size();
            }
            packed[l].out_arc_counts = h_out;
            packed[l].in_arc_counts = h_in;
        }
    }
}

// ----------------------------------------------------------
// MDD GPU Top-Down Enumeration Orchestrator
// ----------------------------------------------------------

ParetoFrontier *enumerate_mdd_topdown(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                      long long gpu_batch_size) {
    if (mdd == NULL) {
        set_reason(reason, "MDD is NULL");
        return NULL;
    }
    if (mdd->num_layers <= 0) {
        set_reason(reason, "MDD has zero layers");
        return NULL;
    }
    long long gpu_mem_baseline_used_bytes = 0;
    long long gpu_mem_peak_used_bytes = 0;
    long long gpu_mem_peak_reserved_bytes = 0;
    if (stats != NULL) {
        if (!capture_gpu_memory_used(reason, &gpu_mem_baseline_used_bytes))
            return NULL;
        gpu_mem_peak_reserved_bytes = gpu_mem_baseline_used_bytes;
        stats->std_candidates_per_layer.assign(mdd->num_layers, 0.0);
        stats->std_frontier_survivors_per_layer.assign(mdd->num_layers, 0.0);
    }

    const int num_layers = mdd->num_layers;

    // 1. Pack top-down layers
    const auto pack_begin = std::chrono::steady_clock::now();
    std::vector<PackedMDDLayer> packed;
    pack_mdd_layers(mdd, packed, false, false);
    if (stats != NULL) {
        stats->wall_pack_transfer_s += std::chrono::duration_cast<std::chrono::duration<double>>(
                                           std::chrono::steady_clock::now() - pack_begin)
                                           .count();
        if (!sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes, &gpu_mem_peak_used_bytes,
                                    &gpu_mem_peak_reserved_bytes))
            return NULL;
    }

    // 2. Initialize frontier at root
    const int root_nodes = packed[0].num_nodes;
    const int root_idx = mdd->get_root()->index;

    thrust::host_vector<int> h_td_sizes(root_nodes, 0);
    h_td_sizes[root_idx] = 1;
    thrust::device_vector<int> d_td_sizes = h_td_sizes;
    thrust::device_vector<int> d_td_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_td_sizes.begin(), d_td_sizes.end(), d_td_offsets.begin());
    d_td_offsets[root_nodes] = 1;
    thrust::device_vector<ObjType> d_td_points(NOBJS, 0); // single point (0,0...)

    // 3. Expand layer by layer
    for (int l = 1; l < num_layers; ++l) {
        thrust::device_vector<int> next_sizes, next_offsets;
        thrust::device_vector<ObjType> next_points;
        long long layer_candidates = 0;
        long long layer_survivors = 0;
        double layer_candidates_std = 0.0;
        double layer_survivors_std = 0.0;

        ScopedCudaEventTimer layer_expand_timer("cudaEventCreate/expand_layer_frontiers", reason);
        if (!layer_expand_timer.ok()) {
            return NULL;
        }
        if (!topdown_expand_mdd_layer(
                packed[l], d_td_offsets, d_td_points, next_sizes, next_offsets, next_points, reason,
                gpu_batch_size, &layer_candidates, &layer_survivors, &layer_candidates_std,
                &layer_survivors_std, stats != NULL ? &gpu_mem_baseline_used_bytes : NULL,
                stats != NULL ? &gpu_mem_peak_used_bytes : NULL,
                stats != NULL ? &gpu_mem_peak_reserved_bytes : NULL)) {
            return NULL;
        }
        if (!layer_expand_timer.finish_and_add(stats != NULL ? &stats->kernel_expand_td_s : NULL,
                                               "cudaEventElapsedTime/expand_layer_frontiers")) {
            return NULL;
        }
        if (stats != NULL) {
            stats->work_candidates_total += layer_candidates;
            stats->work_candidates_peak = std::max(stats->work_candidates_peak, layer_candidates);
            stats->work_frontier_survivors_total += layer_survivors;
            stats->work_frontier_peak_points =
                std::max(stats->work_frontier_peak_points, layer_survivors);
            stats->std_candidates_per_layer[l] = layer_candidates_std;
            stats->std_frontier_survivors_per_layer[l] = layer_survivors_std;
            if (!sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes,
                                        &gpu_mem_peak_used_bytes, &gpu_mem_peak_reserved_bytes))
                return NULL;
        }

        d_td_sizes.swap(next_sizes);
        d_td_offsets.swap(next_offsets);
        d_td_points.swap(next_points);
    }

    // 4. Extract points from terminal node
    const int term_idx = mdd->get_terminal()->index;
    const int term_nodes = packed[num_layers - 1].num_nodes;

    thrust::host_vector<int> h_offsets = d_td_offsets;
    if (term_idx < 0 || term_idx >= term_nodes) {
        set_reason(reason, "Invalid terminal index");
        return NULL;
    }

    const int begin = h_offsets[term_idx];
    const int end = h_offsets[term_idx + 1];
    const int num_points = std::max(0, end - begin);

    ParetoFrontier *frontier = new ParetoFrontier;
    frontier->sols.resize(num_points * NOBJS, 0);
    if (num_points > 0) {
        thrust::host_vector<ObjType> h_points = d_td_points;
        std::copy(h_points.begin() + begin * NOBJS, h_points.begin() + end * NOBJS,
                  frontier->sols.begin());
    }

    if (reason != NULL)
        reason->clear();
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

// ----------------------------------------------------------
// MDD GPU Coupled (Dynamic Layer Cutset) Orchestrator
// ----------------------------------------------------------

ParetoFrontier *enumerate_mdd_coupled(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                      long long gpu_batch_size, long long gpu_max_prod) {
    if (mdd == NULL) {
        set_reason(reason, "MDD is NULL");
        return NULL;
    }
    if (mdd->num_layers <= 0) {
        set_reason(reason, "MDD has zero layers");
        return NULL;
    }
    if (stats != NULL) {
        stats->std_candidates_per_layer.clear();
        stats->std_frontier_survivors_per_layer.clear();
    }

    const int num_layers = mdd->num_layers;

    // 1. Pack all MDD layers into flat GPU arrays
    const auto pack_begin = std::chrono::steady_clock::now();
    std::vector<PackedMDDLayer> packed;
    pack_mdd_layers(mdd, packed, true, true);
    if (stats != NULL) {
        stats->wall_pack_transfer_s += std::chrono::duration_cast<std::chrono::duration<double>>(
                                           std::chrono::steady_clock::now() - pack_begin)
                                           .count();
    }

    // 2. Initialize top-down and bottom-up frontiers
    const int root_nodes = packed[0].num_nodes;
    const int root_idx = mdd->get_root()->index;

    thrust::host_vector<int> h_td_sizes(root_nodes, 0);
    h_td_sizes[root_idx] = 1;
    thrust::device_vector<int> d_td_sizes = h_td_sizes;
    thrust::device_vector<int> d_td_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_td_sizes.begin(), d_td_sizes.end(), d_td_offsets.begin());
    d_td_offsets[root_nodes] = 1;
    thrust::device_vector<ObjType> d_td_points(NOBJS, 0);

    const int term_layer = num_layers - 1;
    const int term_nodes = packed[term_layer].num_nodes;
    const int term_idx = mdd->get_terminal()->index;

    thrust::host_vector<int> h_bu_sizes(term_nodes, 0);
    h_bu_sizes[term_idx] = 1;
    thrust::device_vector<int> d_bu_sizes = h_bu_sizes;
    thrust::device_vector<int> d_bu_offsets(term_nodes + 1, 0);
    thrust::exclusive_scan(d_bu_sizes.begin(), d_bu_sizes.end(), d_bu_offsets.begin());
    d_bu_offsets[term_nodes] = 1;
    thrust::device_vector<ObjType> d_bu_points(NOBJS, 0);

    // 3. Dynamic layer selection loop (all data stays on GPU)
    int layer_td = 0;
    int layer_bu = num_layers - 1;
    int topdown_score = 0;
    int bottomup_score = 0;

    double total_td_time = 0.0, total_bu_time = 0.0;
    int td_iters = 0, bu_iters = 0;

    while (layer_td != layer_bu) {
        clock_t t0 = clock();
        if (topdown_score <= bottomup_score) {
            // Expand top-down
            ++layer_td;
            const int num_nodes = packed[layer_td].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            if (!topdown_expand_mdd_layer(packed[layer_td], d_td_offsets, d_td_points, next_sizes,
                                          next_offsets, next_points, reason, gpu_batch_size,
                                          &layer_candidates, &layer_survivors, NULL, NULL, NULL,
                                          NULL, NULL)) {
                return NULL;
            }
            clock_t t1 = clock();
            double td_elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
            total_td_time += td_elapsed;
            ++td_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak =
                    std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points =
                    std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_td_sizes.swap(next_sizes);
            d_td_offsets.swap(next_offsets);
            d_td_points.swap(next_points);

            // Compute layer value on GPU
            topdown_score =
                compute_expansion_score(d_td_offsets, packed[layer_td].out_arc_counts, num_nodes);
        } else {
            // Expand bottom-up
            --layer_bu;
            const int num_nodes = packed[layer_bu].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            if (!bottomup_expand_mdd_layer(packed[layer_bu], d_bu_offsets, d_bu_points, next_sizes,
                                           next_offsets, next_points, reason, gpu_batch_size,
                                           &layer_candidates, &layer_survivors, NULL, NULL, NULL,
                                           NULL, NULL)) {
                return NULL;
            }
            clock_t t1 = clock();
            double bu_elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
            total_bu_time += bu_elapsed;
            ++bu_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak =
                    std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points =
                    std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_bu_sizes.swap(next_sizes);
            d_bu_offsets.swap(next_offsets);
            d_bu_points.swap(next_points);

            // Compute layer value on GPU (with 1.5x multiplier)
            bottomup_score = 1.5 * compute_expansion_score(
                                       d_bu_offsets, packed[layer_bu].in_arc_counts, num_nodes);
        }
    }

    if (stats != NULL)
        stats->layer_coupling = layer_td;

    // 4. Cutset coupling
    return couple_cutsets_cuda(packed[layer_td].num_nodes, d_td_offsets, d_td_points, d_bu_offsets,
                               d_bu_points, stats, reason, gpu_max_prod);
}

void pack_bdd_layers(BDD *bdd, std::vector<PackedBDDLayer> &packed, bool pack_bottom_up,
                     bool pack_heuristic, const int problem_type, const int state_dominance,
                     bool maximization, EnumerationStats *stats) {
    const int num_layers = bdd->num_layers;
    packed.resize(num_layers);

    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    const bool pack_knapsack_meta = (problem_type == 1 && state_dominance > 0);

    const auto pack_begin = std::chrono::steady_clock::now();

    for (int l = 0; l < num_layers; ++l) {
        const int num_nodes = bdd->layers[l].size();
        packed[l].num_nodes = num_nodes;

        // Top-down: incoming arcs (for layer l > 0)
        if (l > 0) {
            std::vector<int> h_in_offsets(num_nodes + 1, 0);
            for (int dst_idx = 0; dst_idx < num_nodes; ++dst_idx) {
                Node *dst_node = bdd->layers[l][dst_idx];
                int count = 0;
                const int arc_order[2] = {first_arc_type, second_arc_type};
                for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                    count += dst_node->prev[arc_order[arc_pos]].size();
                }
                h_in_offsets[dst_idx + 1] = h_in_offsets[dst_idx] + count;
            }

            const int num_edges = h_in_offsets[num_nodes];
            std::vector<int> h_edge_src;
            std::vector<ObjType> h_edge_weights;
            h_edge_src.reserve(num_edges);
            h_edge_weights.reserve(static_cast<size_t>(num_edges) * NOBJS);

            std::vector<int> h_min_weight;
            std::vector<int> h_parent_id;
            std::vector<int> h_parent_arc;
            if (pack_knapsack_meta) {
                h_min_weight.resize(num_nodes);
                h_parent_id.resize(num_nodes);
                h_parent_arc.resize(num_nodes);
            }

            const int arc_order[2] = {first_arc_type, second_arc_type};
            for (int dst_idx = 0; dst_idx < num_nodes; ++dst_idx) {
                Node *dst_node = bdd->layers[l][dst_idx];
                for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                    const int arc_type = arc_order[arc_pos];
                    for (Node *src_node : dst_node->prev[arc_type]) {
                        h_edge_src.push_back(src_node->index);
                        ObjType *w = src_node->weights[arc_type];
                        for (int o = 0; o < NOBJS; ++o) {
                            h_edge_weights.push_back(w != NULL ? w[o] : 0);
                        }
                    }
                }

                if (pack_knapsack_meta) {
                    h_min_weight[dst_idx] = dst_node->min_weight;
                    const int parents_total = dst_node->prev[0].size() + dst_node->prev[1].size();
                    if (parents_total == 1) {
                        if (dst_node->prev[0].size() == 1) {
                            h_parent_id[dst_idx] = dst_node->prev[0][0]->index;
                            h_parent_arc[dst_idx] = 0;
                        } else {
                            h_parent_id[dst_idx] = dst_node->prev[1][0]->index;
                            h_parent_arc[dst_idx] = 1;
                        }
                    } else {
                        h_parent_id[dst_idx] = -1;
                        h_parent_arc[dst_idx] = -1;
                    }
                }
            }

            packed[l].td_num_edges = num_edges;
            packed[l].td_in_edge_offsets = h_in_offsets;
            packed[l].td_edge_src = h_edge_src;
            packed[l].td_edge_weights = h_edge_weights;
            if (pack_knapsack_meta) {
                packed[l].min_weight = h_min_weight;
                packed[l].single_parent_id = h_parent_id;
                packed[l].single_parent_arc = h_parent_arc;
            }
        } else {
            packed[l].td_num_edges = 0;
        }

        // Bottom-up: outgoing arcs (for layer l < num_layers - 1)
        if (pack_bottom_up && l < num_layers - 1) {
            std::vector<int> h_bu_offsets(num_nodes + 1, 0);
            for (int dst_idx = 0; dst_idx < num_nodes; ++dst_idx) {
                Node *dst_node = bdd->layers[l][dst_idx];
                int count = 0;
                if (dst_node->arcs[first_arc_type] != NULL)
                    count++;
                if (dst_node->arcs[second_arc_type] != NULL)
                    count++;
                h_bu_offsets[dst_idx + 1] = h_bu_offsets[dst_idx] + count;
            }

            const int num_edges = h_bu_offsets[num_nodes];
            std::vector<int> h_edge_src;
            std::vector<ObjType> h_edge_weights;
            h_edge_src.reserve(num_edges);
            h_edge_weights.reserve(static_cast<size_t>(num_edges) * NOBJS);

            const int arc_order[2] = {first_arc_type, second_arc_type};
            for (int dst_idx = 0; dst_idx < num_nodes; ++dst_idx) {
                Node *dst_node = bdd->layers[l][dst_idx];
                for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
                    const int arc_type = arc_order[arc_pos];
                    Node *target = dst_node->arcs[arc_type];
                    if (target != NULL) {
                        h_edge_src.push_back(target->index);
                        ObjType *w = dst_node->weights[arc_type];
                        for (int o = 0; o < NOBJS; ++o) {
                            h_edge_weights.push_back(w != NULL ? w[o] : 0);
                        }
                    }
                }
            }

            packed[l].bu_num_edges = num_edges;
            packed[l].bu_in_edge_offsets = h_bu_offsets;
            packed[l].bu_edge_src = h_edge_src;
            packed[l].bu_edge_weights = h_edge_weights;
        } else {
            packed[l].bu_num_edges = 0;
        }

        // Heuristic arc counts (for coupled selection score)
        if (pack_heuristic) {
            std::vector<int> h_out(num_nodes), h_in(num_nodes);
            for (int d = 0; d < num_nodes; ++d) {
                Node *node = bdd->layers[l][d];
                int out_cnt = 0;
                if (node->arcs[0] != NULL)
                    out_cnt++;
                if (node->arcs[1] != NULL)
                    out_cnt++;
                h_out[d] = out_cnt;
                h_in[d] = node->prev[0].size() + node->prev[1].size();
            }
            packed[l].out_arc_counts = h_out;
            packed[l].in_arc_counts = h_in;
        }
    }

    if (stats != NULL) {
        stats->wall_pack_transfer_s += std::chrono::duration_cast<std::chrono::duration<double>>(
                                           std::chrono::steady_clock::now() - pack_begin)
                                           .count();
    }
}

ParetoFrontier *coupled_bdd_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                           const int state_dominance, EnumerationStats *stats,
                                           std::string *reason, long long gpu_batch_size,
                                           long long gpu_max_prod) {
    if (!prepare_cuda_device(reason)) {
        return NULL;
    }
    if (bdd == NULL) {
        set_reason(reason, "BDD is NULL");
        return NULL;
    }
    if (bdd->num_layers <= 0) {
        set_reason(reason, "BDD has zero layers");
        return NULL;
    }
    if (stats != NULL) {
        stats->std_candidates_per_layer.clear();
        stats->std_frontier_survivors_per_layer.clear();
    }
    const int num_layers = bdd->num_layers;

    // 1. Pack BDD layers
    std::vector<PackedBDDLayer> packed;
    pack_bdd_layers(bdd, packed, true, true, problem_type, state_dominance, maximization, stats);

    // 2. Initialize top-down and bottom-up frontiers
    const int root_nodes = packed[0].num_nodes;
    const int root_idx = bdd->get_root()->index;

    thrust::host_vector<int> h_td_sizes(root_nodes, 0);
    h_td_sizes[root_idx] = 1;
    thrust::device_vector<int> d_td_sizes = h_td_sizes;
    thrust::device_vector<int> d_td_offsets(root_nodes + 1, 0);
    thrust::exclusive_scan(d_td_sizes.begin(), d_td_sizes.end(), d_td_offsets.begin());
    d_td_offsets[root_nodes] = 1;
    thrust::device_vector<ObjType> d_td_points(NOBJS, 0);

    const int term_layer = num_layers - 1;
    const int term_nodes = packed[term_layer].num_nodes;
    const int term_idx = bdd->get_terminal()->index;

    thrust::host_vector<int> h_bu_sizes(term_nodes, 0);
    h_bu_sizes[term_idx] = 1;
    thrust::device_vector<int> d_bu_sizes = h_bu_sizes;
    thrust::device_vector<int> d_bu_offsets(term_nodes + 1, 0);
    thrust::exclusive_scan(d_bu_sizes.begin(), d_bu_sizes.end(), d_bu_offsets.begin());
    d_bu_offsets[term_nodes] = 1;
    thrust::device_vector<ObjType> d_bu_points(NOBJS, 0);

    // 3. Dynamic layer selection loop (all data stays on GPU)
    int layer_td = 0;
    int layer_bu = num_layers - 1;
    int topdown_score = 0;
    int bottomup_score = 0;

    double total_td_time = 0.0, total_bu_time = 0.0;
    int td_iters = 0, bu_iters = 0;

    long long gpu_mem_baseline_used_bytes = 0;
    long long gpu_mem_peak_used_bytes = 0;
    long long gpu_mem_peak_reserved_bytes = 0;
    if (stats != NULL) {
        capture_gpu_memory_used(reason, &gpu_mem_baseline_used_bytes);
        gpu_mem_peak_reserved_bytes = gpu_mem_baseline_used_bytes;
    }

    while (layer_td != layer_bu) {
        clock_t t0 = clock();
        if (topdown_score <= bottomup_score) {
            // Expand top-down
            ++layer_td;
            const int num_nodes = packed[layer_td].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            if (!topdown_expand_bdd_layer(packed[layer_td], d_td_offsets, d_td_points, next_sizes,
                                          next_offsets, next_points, reason, gpu_batch_size,
                                          &layer_candidates, &layer_survivors, NULL, NULL,
                                          stats != NULL ? &gpu_mem_baseline_used_bytes : NULL,
                                          stats != NULL ? &gpu_mem_peak_used_bytes : NULL,
                                          stats != NULL ? &gpu_mem_peak_reserved_bytes : NULL)) {
                return NULL;
            }
            clock_t t1 = clock();
            double td_elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
            total_td_time += td_elapsed;
            ++td_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak =
                    std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points =
                    std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_td_sizes.swap(next_sizes);
            d_td_offsets.swap(next_offsets);
            d_td_points.swap(next_points);

            // Compute layer value on GPU
            topdown_score =
                compute_expansion_score(d_td_offsets, packed[layer_td].out_arc_counts, num_nodes);
        } else {
            // Expand bottom-up
            --layer_bu;
            const int num_nodes = packed[layer_bu].num_nodes;
            thrust::device_vector<int> next_sizes, next_offsets;
            thrust::device_vector<ObjType> next_points;
            long long layer_candidates = 0;
            long long layer_survivors = 0;

            if (!bottomup_expand_bdd_layer(packed[layer_bu], d_bu_offsets, d_bu_points, next_sizes,
                                           next_offsets, next_points, reason, gpu_batch_size,
                                           &layer_candidates, &layer_survivors, NULL, NULL,
                                           stats != NULL ? &gpu_mem_baseline_used_bytes : NULL,
                                           stats != NULL ? &gpu_mem_peak_used_bytes : NULL,
                                           stats != NULL ? &gpu_mem_peak_reserved_bytes : NULL)) {
                return NULL;
            }
            clock_t t1 = clock();
            double bu_elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
            total_bu_time += bu_elapsed;
            ++bu_iters;
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                stats->work_candidates_peak =
                    std::max(stats->work_candidates_peak, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                stats->work_frontier_peak_points =
                    std::max(stats->work_frontier_peak_points, layer_survivors);
            }

            d_bu_sizes.swap(next_sizes);
            d_bu_offsets.swap(next_offsets);
            d_bu_points.swap(next_points);

            // Compute layer value on GPU (with 1.5x multiplier)
            bottomup_score = 1.5 * compute_expansion_score(
                                       d_bu_offsets, packed[layer_bu].in_arc_counts, num_nodes);
        }
    }

    if (stats != NULL)
        stats->layer_coupling = layer_td;

    // 4. Cutset coupling
    ParetoFrontier *res =
        couple_cutsets_cuda(packed[layer_td].num_nodes, d_td_offsets, d_td_points, d_bu_offsets,
                            d_bu_points, stats, reason, gpu_max_prod);

    if (stats != NULL && res != NULL) {
        if (sample_gpu_memory_peak(reason, gpu_mem_baseline_used_bytes, &gpu_mem_peak_used_bytes,
                                   &gpu_mem_peak_reserved_bytes)) {
            stats->gpu_mem_peak_used_bytes = gpu_mem_peak_used_bytes;
            stats->gpu_mem_peak_reserved_bytes = gpu_mem_peak_reserved_bytes;
        }
    }
    return res;
}
