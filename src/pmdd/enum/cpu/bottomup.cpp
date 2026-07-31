// ----------------------------------------------------------
// CPU Bottom-Up BFS Kernels - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

void expand_layer_bottomup(BDD *bdd, const int l, const bool maximization,
                           ParetoFrontierManager *mgmr, const int cpu_threads) {
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    if (maximization) {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node *node = bdd->layers[l][i];

            // Request frontier
            node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

            // add outgoing one arcs
            if (node->arcs[1] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu),
                                                node->weights[1]);
            }

            // add outgoing zero arcs
            if (node->arcs[0] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu),
                                                node->weights[0]);
            }
        }
    } else {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node *node = bdd->layers[l][i];

            // Request frontier
            node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

            // add outgoing zero arcs
            if (node->arcs[0] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu),
                                                node->weights[0]);
            }

            // add outgoing one arcs
            if (node->arcs[1] != NULL) {
                node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu),
                                                node->weights[1]);
            }
        }
    }
    // deallocate next layer
    for (size_t i = 0; i < bdd->layers[l + 1].size(); ++i) {
        recycle_frontier(mgmr, bdd->layers[l + 1][i]->pareto_frontier_bu, parallel_mode);
    }
}

void expand_layer_bottomup_mdd_cpu(MDD *mdd, const int l, ParetoFrontierManager *mgmr,
                                   const bool parallel_mode, const int threads) {
    const int layer_size = mdd->layers[l].size();
    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        MDDNode *node = mdd->layers[l][i];
        node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);

        for (MDDArc *arc : node->out_arcs_list) {
            node->pareto_frontier_bu->merge(*(arc->head->pareto_frontier_bu), arc->weights);
        }
    }

    for (size_t i = 0; i < mdd->layers[l + 1].size(); ++i) {
        recycle_frontier(mgmr, mdd->layers[l + 1][i]->pareto_frontier_bu, parallel_mode);
        delete mdd->layers[l + 1][i];
    }
}
