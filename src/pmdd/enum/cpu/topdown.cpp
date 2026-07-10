// ----------------------------------------------------------
// CPU Top-Down BFS Kernels - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

void expand_layer_topdown_cpu(BDD *bdd, const int layer, const bool maximization,
                              ParetoFrontierManager *mgmr, const bool parallel_mode,
                              const int threads) {
    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;
    const int layer_size = bdd->layers[layer].size();
    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        Node *node = bdd->layers[layer][i];
        node->pareto_frontier = request_frontier(mgmr, parallel_mode);
        for (std::vector<Node *>::iterator prev = node->prev[first_arc_type].begin();
             prev != node->prev[first_arc_type].end(); ++prev) {
            node->pareto_frontier->merge(*((*prev)->pareto_frontier),
                                         (*prev)->weights[first_arc_type]);
        }
        for (std::vector<Node *>::iterator prev = node->prev[second_arc_type].begin();
             prev != node->prev[second_arc_type].end(); ++prev) {
            node->pareto_frontier->merge(*((*prev)->pareto_frontier),
                                         (*prev)->weights[second_arc_type]);
        }
    }
}

void expand_layer_topdown_mdd_cpu(MDD *mdd, const int l, ParetoFrontierManager *mgmr,
                                  const bool parallel_mode, const int threads) {
    const int layer_size = mdd->layers[l].size();
    CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
    for (int i = 0; i < layer_size; ++i) {
        MDDNode *node = mdd->layers[l][i];
        node->pareto_frontier = request_frontier(mgmr, parallel_mode);

        for (MDDArc *arc : node->in_arcs_list) {
            node->pareto_frontier->merge(*(arc->tail->pareto_frontier), arc->weights);
        }
    }

    for (size_t i = 0; i < mdd->layers[l - 1].size(); ++i) {
        recycle_frontier(mgmr, mdd->layers[l - 1][i]->pareto_frontier, parallel_mode);
        delete mdd->layers[l - 1][i];
    }
}

void expand_layer_topdown(BDD *bdd, const int l, const bool maximization,
                          ParetoFrontierManager *mgmr, const int cpu_threads) {
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    if (maximization) {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node *node = bdd->layers[l][i];
            // Request frontier
            node->pareto_frontier = request_frontier(mgmr, parallel_mode);

            // add incoming one arcs
            for (std::vector<Node *>::iterator prev = node->prev[1].begin();
                 prev != node->prev[1].end(); ++prev) {
                node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
            }

            // add incoming zero arcs
            for (std::vector<Node *>::iterator prev = node->prev[0].begin();
                 prev != node->prev[0].end(); ++prev) {
                node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
            }
        }
    } else {
        const int layer_size = bdd->layers[l].size();
        CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
        for (int i = 0; i < layer_size; ++i) {
            Node *node = bdd->layers[l][i];
            // Request frontier
            node->pareto_frontier = request_frontier(mgmr, parallel_mode);

            // add incoming zero arcs
            for (std::vector<Node *>::iterator prev = node->prev[0].begin();
                 prev != node->prev[0].end(); ++prev) {
                node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[0]);
            }

            // add incoming one arcs
            for (std::vector<Node *>::iterator prev = node->prev[1].begin();
                 prev != node->prev[1].end(); ++prev) {
                node->pareto_frontier->merge(*((*prev)->pareto_frontier), (*prev)->weights[1]);
            }
        }
    }

    filter_completion_cpu(bdd, l);

    // deallocate previous layer
    for (size_t i = 0; i < bdd->layers[l - 1].size(); ++i) {
        recycle_frontier(mgmr, bdd->layers[l - 1][i]->pareto_frontier, parallel_mode);
    }
}
