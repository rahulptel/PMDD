// ----------------------------------------------------------
// Shared CUDA enumeration types and layer expansion API
// ----------------------------------------------------------

#ifndef CUDA_ENUM_TYPES_CUH_
#define CUDA_ENUM_TYPES_CUH_

#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>

#include "../../util/util.hpp"

struct EnumerationStats;
struct MDD;
struct BDD;
struct ParetoFrontier;

// Flat representation of one MDD layer's connectivity, on device.
struct PackedMDDLayer {
    int num_nodes;

    // Top-down: incoming arcs grouped by destination node (this layer).
    // edge_src indices refer to nodes in layer-1.
    int td_num_edges;
    thrust::device_vector<int> td_in_edge_offsets;
    thrust::device_vector<int> d_edge_src; // renamed to keep compiler happy if we had collision,
                                           // but let's keep original names:
    thrust::device_vector<int> td_edge_src;
    thrust::device_vector<ObjType> td_edge_weights;

    // Bottom-up: outgoing arcs grouped by node (this layer).
    // edge_src indices refer to head nodes in layer+1.
    int bu_num_edges;
    thrust::device_vector<int> bu_in_edge_offsets;
    thrust::device_vector<int> bu_edge_src;
    thrust::device_vector<ObjType> bu_edge_weights;

    // Per-node arc counts for the layer-value heuristic.
    thrust::device_vector<int> out_arc_counts;
    thrust::device_vector<int> in_arc_counts;
};

// Flat representation of one BDD layer's connectivity, on device.
struct PackedBDDLayer {
    int num_nodes;

    // Top-down: incoming arcs grouped by destination node (this layer).
    int td_num_edges;
    thrust::device_vector<int> td_in_edge_offsets;
    thrust::device_vector<int> td_edge_src;
    thrust::device_vector<ObjType> td_edge_weights;

    // Bottom-up: outgoing arcs grouped by node (this layer).
    int bu_num_edges;
    thrust::device_vector<int> bu_in_edge_offsets;
    thrust::device_vector<int> bu_edge_src;
    thrust::device_vector<ObjType> bu_edge_weights;

    // Per-node arc counts for the layer-value heuristic.
    thrust::device_vector<int> out_arc_counts;
    thrust::device_vector<int> in_arc_counts;

    // BDD Knapsack Dominance metadata (packed top-down)
    thrust::device_vector<int> min_weight;
    thrust::device_vector<int> single_parent_id;
    thrust::device_vector<int> single_parent_arc;
};

// ----------------------------------------------------------
// Common utility functions for CUDA
// ----------------------------------------------------------

inline bool set_reason(std::string *reason, const std::string &message) {
    if (reason != NULL) {
        *reason = message;
    }
    return false;
}

inline bool cuda_ok(cudaError_t err, const char *where, std::string *reason) {
    if (err == cudaSuccess) {
        return true;
    }
    std::string msg = where;
    msg += ": ";
    msg += cudaGetErrorString(err);
    return set_reason(reason, msg);
}

inline bool sync_kernel(const char *where, std::string *reason) {
    if (!cuda_ok(cudaGetLastError(), where, reason))
        return false;
    return cuda_ok(cudaDeviceSynchronize(), where, reason);
}

inline bool capture_gpu_memory_used(std::string *reason, long long *used_bytes_out) {
    if (used_bytes_out == NULL) {
        return set_reason(reason, "capture_gpu_memory_used: output pointer is NULL");
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", reason)) {
        return false;
    }
    *used_bytes_out = static_cast<long long>(total_bytes) - static_cast<long long>(free_bytes);
    return true;
}

inline bool sample_gpu_memory_peak(std::string *reason, const long long baseline_used_bytes,
                                   long long *peak_used_bytes, long long *peak_reserved_bytes) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", reason)) {
        return false;
    }

    const long long used_bytes =
        static_cast<long long>(total_bytes) - static_cast<long long>(free_bytes);
    if (peak_reserved_bytes != NULL && used_bytes > *peak_reserved_bytes) {
        *peak_reserved_bytes = used_bytes;
    }

    long long delta_used_bytes = used_bytes - baseline_used_bytes;
    if (delta_used_bytes < 0) {
        delta_used_bytes = 0;
    }
    if (peak_used_bytes != NULL && delta_used_bytes > *peak_used_bytes) {
        *peak_used_bytes = delta_used_bytes;
    }
    return true;
}

class ScopedCudaEventTimer {
  public:
    ScopedCudaEventTimer(const char *where, std::string *reason)
        : start_(NULL), stop_(NULL), active_(false), reason_(reason) {
        if (!cuda_ok(cudaEventCreate(&start_), where, reason_)) {
            return;
        }
        if (!cuda_ok(cudaEventCreate(&stop_), where, reason_)) {
            return;
        }
        if (!cuda_ok(cudaEventRecord(start_), where, reason_)) {
            return;
        }
        active_ = true;
    }

    bool ok() const { return active_; }

    bool finish_and_add(double *seconds_out, const char *where) {
        if (!active_) {
            return false;
        }
        if (!cuda_ok(cudaEventRecord(stop_), where, reason_)) {
            active_ = false;
            return false;
        }
        if (!cuda_ok(cudaEventSynchronize(stop_), where, reason_)) {
            active_ = false;
            return false;
        }
        float milliseconds = 0.0f;
        if (!cuda_ok(cudaEventElapsedTime(&milliseconds, start_, stop_), where, reason_)) {
            active_ = false;
            return false;
        }
        if (seconds_out != NULL) {
            *seconds_out += static_cast<double>(milliseconds) / 1000.0;
        }
        active_ = false;
        return true;
    }

    ~ScopedCudaEventTimer() {
        if (start_ != NULL) {
            cudaEventDestroy(start_);
        }
        if (stop_ != NULL) {
            cudaEventDestroy(stop_);
        }
    }

  private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    bool active_;
    std::string *reason_;
};

// ----------------------------------------------------------
// Core CUDA functions
// ----------------------------------------------------------

int compute_expansion_score(const thrust::device_vector<int> &offsets,
                            const thrust::device_vector<int> &arc_counts, int num_nodes);

bool expand_layer_frontiers(
    const thrust::device_vector<int> &in_edge_offsets, const thrust::device_vector<int> &edge_src,
    const thrust::device_vector<ObjType> &edge_weights, int num_edges, int next_nodes,
    const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch,
    long long *total_candidates_out = NULL, long long *total_next_out = NULL,
    double *std_candidates_out = NULL, double *std_survivors_out = NULL,
    long long *gpu_mem_baseline_used_bytes = NULL, long long *gpu_mem_peak_used_bytes = NULL,
    long long *gpu_mem_peak_reserved_bytes = NULL);

// MDD layer packing helper
void pack_mdd_layers(MDD *mdd, std::vector<PackedMDDLayer> &packed, bool pack_bottom_up,
                     bool pack_heuristic);

// BDD layer packing helper
void pack_bdd_layers(BDD *bdd, std::vector<PackedBDDLayer> &packed, bool pack_bottom_up,
                     bool pack_heuristic, const int problem_type, const int state_dominance,
                     bool maximization, EnumerationStats *stats = NULL);

// Directional wrappers
bool topdown_expand_mdd_layer(
    const PackedMDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch,
    long long *total_candidates_out = NULL, long long *total_next_out = NULL,
    double *std_candidates_out = NULL, double *std_survivors_out = NULL,
    long long *gpu_mem_baseline_used_bytes = NULL, long long *gpu_mem_peak_used_bytes = NULL,
    long long *gpu_mem_peak_reserved_bytes = NULL);

bool bottomup_expand_mdd_layer(
    const PackedMDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch,
    long long *total_candidates_out = NULL, long long *total_next_out = NULL,
    double *std_candidates_out = NULL, double *std_survivors_out = NULL,
    long long *gpu_mem_baseline_used_bytes = NULL, long long *gpu_mem_peak_used_bytes = NULL,
    long long *gpu_mem_peak_reserved_bytes = NULL);

bool topdown_expand_bdd_layer(
    const PackedBDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch,
    long long *total_candidates_out = NULL, long long *total_next_out = NULL,
    double *std_candidates_out = NULL, double *std_survivors_out = NULL,
    long long *gpu_mem_baseline_used_bytes = NULL, long long *gpu_mem_peak_used_bytes = NULL,
    long long *gpu_mem_peak_reserved_bytes = NULL);

bool bottomup_expand_bdd_layer(
    const PackedBDDLayer &packed_layer, const thrust::device_vector<int> &d_prev_offsets,
    const thrust::device_vector<ObjType> &d_prev_points, thrust::device_vector<int> &d_next_sizes,
    thrust::device_vector<int> &d_next_offsets, thrust::device_vector<ObjType> &d_next_points,
    std::string *reason, long long max_candidate_points_per_batch,
    long long *total_candidates_out = NULL, long long *total_next_out = NULL,
    double *std_candidates_out = NULL, double *std_survivors_out = NULL,
    long long *gpu_mem_baseline_used_bytes = NULL, long long *gpu_mem_peak_used_bytes = NULL,
    long long *gpu_mem_peak_reserved_bytes = NULL);

// General coupling helper
ParetoFrontier *couple_cutsets_cuda(int num_nodes, const thrust::device_vector<int> &d_td_offsets,
                                    const thrust::device_vector<ObjType> &d_td_points,
                                    const thrust::device_vector<int> &d_bu_offsets,
                                    const thrust::device_vector<ObjType> &d_bu_points,
                                    EnumerationStats *stats, std::string *reason,
                                    long long gpu_max_prod);

#endif
