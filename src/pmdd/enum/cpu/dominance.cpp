// ----------------------------------------------------------
// CPU State Dominance and Completion Filters - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

void filter_dominance_cpu(BDD *bdd, const int layer, const int problem_type,
                          const int state_dominance, EnumerationStats *stats) {
    (void)state_dominance;
    if (problem_type == 1) {
        // Knapsack
        filter_dominance_knapsack_cpu(bdd, layer, stats);
    } else if (problem_type == 2) {
        // Set packing
        filter_dominance_setpacking_cpu(bdd, layer, stats);
    }
}

void filter_dominance_knapsack_cpu(BDD *bdd, const int layer, EnumerationStats *stats) {
    if (bdd->layers[layer].size() > 1) {
        // Compare the nodes based on their min weights, from largest to smallest
        // Ex: 15 <= 11 <= 8 <= 5 <= 0
        std::vector<intpair> NodeOrder_Weight;
        for (size_t i = 0; i < bdd->layers[layer].size(); i++) {
            if (!bdd->layers[layer][i]->pareto_frontier->sols.empty())
                NodeOrder_Weight.push_back(intpair(i, bdd->layers[layer][i]->min_weight));
        }
        std::sort(NodeOrder_Weight.begin(), NodeOrder_Weight.end(), IntPairLargestToSmallestComp);

        // k can be dominated by k+1,k+2,...
        for (size_t i = 0; i < NodeOrder_Weight.size() - 1; i++) {
            int index1 = NodeOrder_Weight[i].first;
            Node *node1 = bdd->layers[layer][index1];

            // Check each label of node at index1 whether it is dominated or not
            int num_dominated = 0;

            for (size_t j = i + 1; j < std::min((size_t)NodeOrder_Weight.size(), i + 3);
                 j++) { // i+3 is just a chosen strategy in order not to try all pairs

                int index2 = NodeOrder_Weight[j].first;
                Node *node2 = bdd->layers[layer][index2];

                // if node1 and node 2 have one parent and they are the same, continue
                if (node1->prev[0].size() + node1->prev[1].size() == 1 &&
                    node2->prev[0].size() + node2->prev[1].size() == 1) {
                    if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 &&
                        node1->prev[0][0] == node2->prev[1][0])
                        continue;
                    if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 &&
                        node1->prev[1][0] == node2->prev[0][0])
                        continue;
                }

                for (size_t s1 = 0; s1 < node1->pareto_frontier->sols.size(); s1 += NOBJS) {
                    if (node1->pareto_frontier->sols[s1] == DOMINATED)
                        continue;

                    bool dominated = true;
                    for (size_t s2 = 0; s2 < node2->pareto_frontier->sols.size(); s2 += NOBJS) {
                        dominated = true;
                        for (int p = 0; p < NOBJS && dominated; ++p)
                            dominated = (node2->pareto_frontier->sols[s2 + p] >=
                                         node1->pareto_frontier->sols[s1 + p]);
                        if (dominated) {
                            node1->pareto_frontier->sols[s1] = DOMINATED;
                            num_dominated++;
                            break;
                        }
                    }
                }
            }

            if (num_dominated > 0) {
                assert(stats != NULL);
                node1->pareto_frontier->remove_dominated();
                stats->dominance_filtered_total += num_dominated;
            }
        }
    }
}

void filter_dominance_setpacking_cpu(BDD *bdd, const int layer, EnumerationStats *stats) {
    int NoNodes = (int)bdd->layers[layer].size();
    if (NoNodes > 1) {
        std::vector<std::vector<Node *>> Bucket_NodeIndices(bdd->num_layers - 1,
                                                            std::vector<Node *>());
        // Put the nodes into buckets
        for (int i = 0; i < NoNodes; i++)
            Bucket_NodeIndices[bdd->layers[layer][i]->setpack_state.count()].push_back(
                bdd->layers[layer][i]);

        // Go over the buckets
        for (size_t b1 = 0; b1 < Bucket_NodeIndices.size() - 1; b1++) {
            if (Bucket_NodeIndices[b1].size() > 1) {
                for (size_t i = 0; i < Bucket_NodeIndices[b1].size(); i++) {
                    Node *node1 = Bucket_NodeIndices[b1][i]; // try to dominate node1
                    int num_dominated = 0;
                    bool Candidate_j_Found = false;
                    for (size_t b2 = b1 + 1; b2 < Bucket_NodeIndices.size();
                         b2++) { // only j in a subsequent bucket can potentially dominate node1
                        for (size_t j = 0; j < Bucket_NodeIndices[b2].size(); j++) {
                            Node *node2 =
                                Bucket_NodeIndices[b2][j]; // node2 can potentially dominate node1

                            // if node1 and node2 do not have same min and max element, then they
                            // cannot be subsets
                            if (node1->setpack_state.find_first() !=
                                node2->setpack_state.find_first())
                                continue;

                            // if node1 and node 2 have one parent and they are the same, continue
                            if (node1->prev[0].size() + node1->prev[1].size() == 1 &&
                                node2->prev[0].size() + node2->prev[1].size() == 1) {
                                if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 &&
                                    node1->prev[0][0] == node2->prev[1][0])
                                    continue;
                                if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 &&
                                    node1->prev[1][0] == node2->prev[0][0])
                                    continue;
                            }

                            if (!node1->setpack_state.is_subset_of(node2->setpack_state))
                                continue;

                            Candidate_j_Found = true;

                            // Check each label of node at index1 whether it is dominated or not
                            for (size_t s1 = 0; s1 < node1->pareto_frontier->sols.size();
                                 s1 += NOBJS) {
                                if (node1->pareto_frontier->sols[s1] == DOMINATED)
                                    continue;

                                bool dominated = true;
                                for (size_t s2 = 0; s2 < node2->pareto_frontier->sols.size();
                                     s2 += NOBJS) {
                                    dominated = true;
                                    for (int p = 0; p < NOBJS && dominated; ++p)
                                        dominated = (node2->pareto_frontier->sols[s2 + p] >=
                                                     node1->pareto_frontier->sols[s1 + p]);
                                    if (dominated) {
                                        node1->pareto_frontier->sols[s1] = DOMINATED;
                                        num_dominated++;
                                        break;
                                    }
                                }
                            }

                            if (Candidate_j_Found)
                                break;
                        }

                        // TRY ONLY THE NEXT NONEMPTY BUCKET
                        if (Bucket_NodeIndices[b2].size() > 0)
                            break;

                        if (Candidate_j_Found)
                            break;
                    }
                    if (num_dominated > 0) {
                        node1->pareto_frontier->remove_dominated();
                        stats->dominance_filtered_total += num_dominated;
                    }
                }
            }
        }
    }
}

void filter_completion_cpu(BDD *bdd, const int layer) {
    (void)bdd;
    (void)layer;
}
