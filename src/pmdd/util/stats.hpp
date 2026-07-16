/*
 * -------------------------------------------------------------------
 * Statistic routines for profiling
 * -------------------------------------------------------------------
 */

#ifndef STATS_HPP_
#define STATS_HPP_

#include <ctime>
#include <string>
#include <vector>

//
// Enumeration stats populated by BDD/CUDA during frontier enumeration.
// This is the canonical runtime stats object.
//
struct EnumerationStats {
    // Time spent in state-dominance filtering (CPU seconds)
    double cpu_state_dominance_s;
    // Solutions filtered by pareto dominance
    int dominance_filtered_total;
    // Layer where coupling happened
    int layer_coupling;
    // Number of Pareto solutions produced by the run
    int num_solutions;

    // End-to-end timing (seconds)
    double cpu_compile_s;
    double cpu_enumeration_s;
    double cpu_total_s;
    double wall_compile_s;
    double wall_enumeration_s;

    // Aggregated wall times (seconds) for CPU phases
    double wall_state_dominance_s;
    double wall_cutset_sort_s;
    double wall_cutset_convolution_s;
    double wall_cutset_partial_merge_s;

    // Aggregated wall times (seconds) for GPU packing and join phases
    double wall_pack_transfer_s;
    double wall_join_s;
    // Aggregated CUDA kernel times captured with cudaEvent_t (seconds)
    double kernel_expand_td_s;
    double kernel_dominance_s;
    double kernel_total_s;
    // Peak CPU memory usage (bytes) sampled with getrusage.
    long long cpu_mem_peak_bytes;

    // Peak GPU memory usage (bytes) sampled with cudaMemGetInfo.
    // used_bytes is baseline-adjusted for this run; reserved_bytes is device-wide used memory.
    long long gpu_mem_peak_used_bytes;
    long long gpu_mem_peak_reserved_bytes;

    // Method-agnostic work counters
    long long work_candidates_total;
    long long work_candidates_peak;
    long long work_frontier_survivors_total;
    long long work_frontier_peak_points;
    long long work_join_products_total;
    // Per-layer population std of node-level candidate counts (methods 1/2 only).
    std::vector<double> std_candidates_per_layer;
    // Per-layer population std of node-level frontier survivor counts (methods 1/2 only).
    std::vector<double> std_frontier_survivors_per_layer;

    // Aggregated counters for CPU phases
    int cpu_layers_td;
    int cpu_layers_bu;
    long long cpu_nodes_expanded;
    int cpu_cutset_size;

    EnumerationStats()
        : cpu_state_dominance_s(0.0), dominance_filtered_total(0), layer_coupling(0),
          num_solutions(0), cpu_compile_s(0.0), cpu_enumeration_s(0.0), cpu_total_s(0.0),
          wall_compile_s(0.0), wall_enumeration_s(0.0), wall_state_dominance_s(0.0),
          wall_cutset_sort_s(0.0), wall_cutset_convolution_s(0.0), wall_cutset_partial_merge_s(0.0),
          wall_pack_transfer_s(0.0), wall_join_s(0.0), kernel_expand_td_s(0.0),
          kernel_dominance_s(0.0), kernel_total_s(0.0), cpu_mem_peak_bytes(0),
          gpu_mem_peak_used_bytes(0), gpu_mem_peak_reserved_bytes(0), work_candidates_total(0),
          work_candidates_peak(0), work_frontier_survivors_total(0), work_frontier_peak_points(0),
          work_join_products_total(0), std_candidates_per_layer(),
          std_frontier_survivors_per_layer(), cpu_layers_td(0), cpu_layers_bu(0),
          cpu_nodes_expanded(0), cpu_cutset_size(0) {}
};

// Backward-compatible alias: existing code can keep using MultiObjectiveStats.
using MultiObjectiveStats = EnumerationStats;

//
// Lightweight run-summary/output view model.
// Keep only metadata and non-instrumentation fields.
//
struct DDStats {
    long original_width;
    long reduced_width;
    long original_num_nodes;
    long reduced_num_nodes;
    std::vector<long> max_num_nodes_per_layer;
    long max_num_in_arcs;
    long total_num_in_arcs;
    std::string status_state;
    std::string status_error_message;

    DDStats()
        : original_width(-1), reduced_width(-1), original_num_nodes(-1), reduced_num_nodes(-1),
          max_num_nodes_per_layer(), max_num_in_arcs(-1), total_num_in_arcs(-1), status_state("ok"),
          status_error_message("") {}
};

#endif /* STATS_HPP_ */
