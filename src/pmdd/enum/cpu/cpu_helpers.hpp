// ----------------------------------------------------------
// CPU Enumeration Helpers
// ----------------------------------------------------------

#ifndef CPU_HELPERS_HPP_
#define CPU_HELPERS_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../bdd/bdd.hpp"
#include "../../mdd/mdd.hpp"
#include "../../util/omp_compat.hpp"
#include "../../util/stats.hpp"
#include "../pareto_frontier.hpp"

typedef std::pair<int, int> intpair;
typedef std::chrono::steady_clock WallClock;

struct CpuTopdownTileTask {
    int node_idx;
    size_t local_begin;
    size_t local_end;
};

inline ParetoFrontier *request_frontier(ParetoFrontierManager *mgmr, const bool parallel_mode) {
    return parallel_mode ? new ParetoFrontier : mgmr->request();
}

inline void recycle_frontier(ParetoFrontierManager *mgmr, ParetoFrontier *frontier,
                             const bool parallel_mode) {
    if (frontier == NULL) {
        return;
    }
    if (parallel_mode) {
        delete frontier;
    } else {
        mgmr->deallocate(frontier);
    }
}

inline ParetoFrontier *parallel_reduce_partial_frontiers(std::vector<ParetoFrontier *> &partial,
                                                         const bool parallel_mode,
                                                         const int threads) {
    size_t active_count = 0;
    for (size_t i = 0; i < partial.size(); ++i) {
        if (partial[i] != NULL) {
            partial[active_count++] = partial[i];
        }
    }

    if (active_count == 0) {
        return new ParetoFrontier;
    }

    while (active_count > 1) {
        const size_t pair_count = active_count / 2;
        const long long pair_count_ll = static_cast<long long>(pair_count);
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (long long p = 0; p < pair_count_ll; ++p) {
            const size_t lhs_idx = static_cast<size_t>(p) * 2;
            const size_t rhs_idx = lhs_idx + 1;
            ParetoFrontier *lhs = partial[lhs_idx];
            ParetoFrontier *rhs = partial[rhs_idx];
            assert(lhs != NULL);
            assert(rhs != NULL);
            lhs->merge(*rhs);
            delete rhs;
            partial[static_cast<size_t>(p)] = lhs;
        }

        if ((active_count % 2) != 0) {
            partial[pair_count] = partial[active_count - 1];
            active_count = pair_count + 1;
        } else {
            active_count = pair_count;
        }
    }

    return partial[0];
}

inline bool IntPairLargestToSmallestComp(intpair l, intpair r) {
    return l.second > r.second; // from largest to smallest
}

inline bool SetPackingStateMinElementSmallestToLargestComp(Node *l, Node *r) {
    return l->setpack_state.find_first() <
           r->setpack_state.find_first(); // from smallest to largest
}

struct CompareNode {
    bool operator()(const Node *nodeA, const Node *nodeB) const {
        return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) >
               (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
    }
};

struct CompareMDDNode {
    bool operator()(const MDDNode *nodeA, const MDDNode *nodeB) const {
        return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) >
               (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
    }
};

inline double wall_elapsed_s(const WallClock::time_point &start) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(WallClock::now() - start)
        .count();
}

inline void reset_cpu_metrics_stats(EnumerationStats *stats) {
    if (stats == NULL) {
        return;
    }
    stats->wall_state_dominance_s = 0.0;
    stats->wall_cutset_sort_s = 0.0;
    stats->wall_cutset_convolution_s = 0.0;
    stats->wall_cutset_partial_merge_s = 0.0;
    stats->wall_pack_transfer_s = 0.0;
    stats->wall_join_s = 0.0;
    stats->kernel_expand_td_s = 0.0;
    stats->kernel_dominance_s = 0.0;
    stats->kernel_total_s = 0.0;
    stats->gpu_mem_peak_used_bytes = 0;
    stats->gpu_mem_peak_reserved_bytes = 0;
    stats->work_candidates_total = 0;
    stats->work_candidates_peak = 0;
    stats->work_frontier_survivors_total = 0;
    stats->work_frontier_peak_points = 0;
    stats->work_join_products_total = 0;
    stats->std_candidates_per_layer.clear();
    stats->std_frontier_survivors_per_layer.clear();
    stats->cpu_layers_td = 0;
    stats->cpu_layers_bu = 0;
    stats->cpu_nodes_expanded = 0;
    stats->cpu_cutset_size = 0;
}

inline void update_peak_points(EnumerationStats *stats, const long long value) {
    if (stats == NULL) {
        return;
    }
    if (value > stats->work_frontier_peak_points) {
        stats->work_frontier_peak_points = value;
    }
}

inline void update_peak_candidates(EnumerationStats *stats, const long long value) {
    if (stats == NULL) {
        return;
    }
    if (value > stats->work_candidates_peak) {
        stats->work_candidates_peak = value;
    }
}

inline double population_std_from_sums(const long double sum, const long double sum_sq,
                                       const long long count) {
    if (count <= 0) {
        return 0.0;
    }
    const long double mean = sum / static_cast<long double>(count);
    long double variance = sum_sq / static_cast<long double>(count) - mean * mean;
    if (variance < 0.0L) {
        variance = 0.0L;
    }
    return std::sqrt(static_cast<double>(variance));
}

inline long long bdd_node_candidates_topdown(const Node *node, const bool maximization) {
    if (node == NULL) {
        return 0;
    }
    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    long long node_candidates = 0;
    for (Node *prev : node->prev[first_arc_type]) {
        if (prev != NULL && prev->pareto_frontier != NULL) {
            node_candidates += prev->pareto_frontier->get_num_sols();
        }
    }
    for (Node *prev : node->prev[second_arc_type]) {
        if (prev != NULL && prev->pareto_frontier != NULL) {
            node_candidates += prev->pareto_frontier->get_num_sols();
        }
    }
    return node_candidates;
}

inline long long bdd_node_candidates_bottomup(const Node *node, const bool maximization) {
    if (node == NULL) {
        return 0;
    }
    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    long long node_candidates = 0;
    if (node->arcs[first_arc_type] != NULL &&
        node->arcs[first_arc_type]->pareto_frontier_bu != NULL) {
        node_candidates += node->arcs[first_arc_type]->pareto_frontier_bu->get_num_sols();
    }
    if (node->arcs[second_arc_type] != NULL &&
        node->arcs[second_arc_type]->pareto_frontier_bu != NULL) {
        node_candidates += node->arcs[second_arc_type]->pareto_frontier_bu->get_num_sols();
    }
    return node_candidates;
}

inline long long bdd_node_survivors_topdown(const Node *node) {
    return (node != NULL && node->pareto_frontier != NULL) ? node->pareto_frontier->get_num_sols()
                                                           : 0;
}

inline long long bdd_node_survivors_bottomup(const Node *node) {
    return (node != NULL && node->pareto_frontier_bu != NULL)
               ? node->pareto_frontier_bu->get_num_sols()
               : 0;
}

inline double std_bdd_candidates_topdown_layer(BDD *bdd, const int layer, const bool maximization) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = bdd->layers[layer].size();
    for (Node *node : bdd->layers[layer]) {
        const long long node_candidates = bdd_node_candidates_topdown(node, maximization);
        const long double node_value = static_cast<long double>(node_candidates);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline double std_bdd_survivors_topdown_layer(BDD *bdd, const int layer) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = bdd->layers[layer].size();
    for (Node *node : bdd->layers[layer]) {
        const long long node_survivors = bdd_node_survivors_topdown(node);
        const long double node_value = static_cast<long double>(node_survivors);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline double std_bdd_candidates_bottomup_layer(BDD *bdd, const int layer,
                                                const bool maximization) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = bdd->layers[layer].size();
    for (Node *node : bdd->layers[layer]) {
        const long long node_candidates = bdd_node_candidates_bottomup(node, maximization);
        const long double node_value = static_cast<long double>(node_candidates);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline double std_bdd_survivors_bottomup_layer(BDD *bdd, const int layer) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = bdd->layers[layer].size();
    for (Node *node : bdd->layers[layer]) {
        const long long node_survivors = bdd_node_survivors_bottomup(node);
        const long double node_value = static_cast<long double>(node_survivors);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline double std_mdd_candidates_topdown_layer(MDD *mdd, const int layer) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = mdd->layers[layer].size();
    for (MDDNode *node : mdd->layers[layer]) {
        long long node_candidates = 0;
        for (MDDArc *arc : node->in_arcs_list) {
            if (arc != NULL && arc->tail != NULL && arc->tail->pareto_frontier != NULL) {
                node_candidates += arc->tail->pareto_frontier->get_num_sols();
            }
        }
        const long double node_value = static_cast<long double>(node_candidates);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline double std_mdd_survivors_topdown_layer(MDD *mdd, const int layer) {
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    const long long node_count = mdd->layers[layer].size();
    for (MDDNode *node : mdd->layers[layer]) {
        const long long node_survivors = (node != NULL && node->pareto_frontier != NULL)
                                             ? node->pareto_frontier->get_num_sols()
                                             : 0;
        const long double node_value = static_cast<long double>(node_survivors);
        sum += node_value;
        sum_sq += node_value * node_value;
    }
    return population_std_from_sums(sum, sum_sq, node_count);
}

inline long long count_bdd_candidates_topdown_layer(BDD *bdd, const int layer,
                                                    const bool maximization) {
    long long total = 0;
    for (Node *node : bdd->layers[layer]) {
        total += bdd_node_candidates_topdown(node, maximization);
    }
    return total;
}

inline long long count_bdd_survivors_topdown_layer(BDD *bdd, const int layer) {
    long long total = 0;
    for (Node *node : bdd->layers[layer]) {
        total += bdd_node_survivors_topdown(node);
    }
    return total;
}

inline long long count_bdd_candidates_bottomup_layer(BDD *bdd, const int layer,
                                                     const bool maximization) {
    long long total = 0;
    for (Node *node : bdd->layers[layer]) {
        total += bdd_node_candidates_bottomup(node, maximization);
    }
    return total;
}

inline long long count_bdd_survivors_bottomup_layer(BDD *bdd, const int layer) {
    long long total = 0;
    for (Node *node : bdd->layers[layer]) {
        total += bdd_node_survivors_bottomup(node);
    }
    return total;
}

inline long long count_mdd_candidates_topdown_layer(MDD *mdd, const int layer) {
    long long total = 0;
    for (MDDNode *node : mdd->layers[layer]) {
        for (MDDArc *arc : node->in_arcs_list) {
            if (arc != NULL && arc->tail != NULL && arc->tail->pareto_frontier != NULL) {
                total += arc->tail->pareto_frontier->get_num_sols();
            }
        }
    }
    return total;
}

inline long long count_mdd_survivors_topdown_layer(MDD *mdd, const int layer) {
    long long total = 0;
    for (MDDNode *node : mdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier != NULL) {
            total += node->pareto_frontier->get_num_sols();
        }
    }
    return total;
}

inline long long count_mdd_candidates_bottomup_layer(MDD *mdd, const int layer) {
    long long total = 0;
    for (MDDNode *node : mdd->layers[layer]) {
        for (MDDArc *arc : node->out_arcs_list) {
            if (arc != NULL && arc->head != NULL && arc->head->pareto_frontier_bu != NULL) {
                total += arc->head->pareto_frontier_bu->get_num_sols();
            }
        }
    }
    return total;
}

inline long long count_mdd_survivors_bottomup_layer(MDD *mdd, const int layer) {
    long long total = 0;
    for (MDDNode *node : mdd->layers[layer]) {
        if (node != NULL && node->pareto_frontier_bu != NULL) {
            total += node->pareto_frontier_bu->get_num_sols();
        }
    }
    return total;
}

// Helper kernel declarations shared across CPU compilation units

void expand_layer_topdown(BDD *bdd, const int l, const bool maximization,
                          ParetoFrontierManager *mgmr, const int cpu_threads);
void expand_layer_bottomup(BDD *bdd, const int l, const bool maximization,
                           ParetoFrontierManager *mgmr, const int cpu_threads);

void expand_layer_topdown_cpu(BDD *bdd, const int layer, const bool maximization,
                              ParetoFrontierManager *mgmr, const bool parallel_mode,
                              const int threads);
void expand_layer_topdown_mdd_cpu(MDD *mdd, const int l, ParetoFrontierManager *mgmr,
                                  const bool parallel_mode, const int threads);
void expand_layer_bottomup_mdd_cpu(MDD *mdd, const int l, ParetoFrontierManager *mgmr,
                                   const bool parallel_mode, const int threads);

int topdown_layer_value(BDD *bdd, Node *node);
int bottomup_layer_value(BDD *bdd, Node *node);
int topdown_layer_value(MDD *mdd, MDDNode *node);
int bottomup_layer_value(MDD *mdd, MDDNode *node);

#endif // CPU_HELPERS_HPP_
