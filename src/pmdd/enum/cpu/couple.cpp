// ----------------------------------------------------------
// CPU Coupled Cutset Expansion Kernels - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

//
// Topdown value of a node (for dynamic layer selection)
//
int topdown_layer_value(BDD *bdd, Node *node) {
    int total = 0;
    for (int t = 0; t < 2; ++t) {
        if (node->arcs[t] != NULL) {
            total += node->pareto_frontier->get_num_sols();
        }
    }
    return total;
}

//
// Bottomup value of a node (for dynamic layer selection)
//
int bottomup_layer_value(BDD *bdd, Node *node) {
    int total = 0;
    for (int t = 0; t < 2; ++t) {
        total += node->pareto_frontier_bu->get_num_sols() * node->prev[t].size();
    }
    return 1.5 * total;
}

//
// Topdown value of a node (for dynamic layer selection)
//
int topdown_layer_value(MDD *mdd, MDDNode *node) {
    return node->pareto_frontier->get_num_sols() * node->out_arcs_list.size();
}

//
// Bottomup value of a node (for dynamic layer selection)
//
int bottomup_layer_value(MDD *mdd, MDDNode *node) {
    return 1.5 * node->pareto_frontier_bu->get_num_sols() * node->in_arcs_list.size();
}
