// ----------------------------------------------------------
// CPU Enumeration Orchestration Drivers - Implementation
// ----------------------------------------------------------

#include "cpu_helpers.hpp"
#include "cpu_wrappers.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ctime>
#include <vector>

ParetoFrontier *topdown_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      int cpu_threads) {
    stats->cpu_state_dominance_s = 0.0;
    stats->dominance_filtered_total = 0;
    reset_cpu_metrics_stats(stats);
    if (stats != NULL) {
        stats->std_candidates_per_layer.assign(bdd->num_layers, 0.0);
        stats->std_frontier_survivors_per_layer.assign(bdd->num_layers, 0.0);
    }
    clock_t init;
    const bool metrics_enabled = (stats != NULL);

    ParetoFrontierManager *mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);

    ObjType zero_array[NOBJS];
    std::memset(zero_array, 0, sizeof(ObjType) * NOBJS);
    bdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
    bdd->get_root()->pareto_frontier->add(zero_array);

    for (int l = 1; l < bdd->num_layers; ++l) {
        const long long layer_candidates = count_bdd_candidates_topdown_layer(bdd, l, maximization);
        const double layer_candidates_std = std_bdd_candidates_topdown_layer(bdd, l, maximization);
        const int layer_size = bdd->layers[l].size();

        expand_layer_topdown_cpu(bdd, l, maximization, mgmr, parallel_mode, threads);

        if (state_dominance > 0) {
            const WallClock::time_point dominance_begin =
                metrics_enabled ? WallClock::now() : WallClock::time_point();
            init = clock();
            filter_dominance_cpu(bdd, l, problem_type, state_dominance, stats);
            stats->cpu_state_dominance_s += static_cast<double>(clock() - init) / CLOCKS_PER_SEC;
            if (metrics_enabled) {
                stats->wall_state_dominance_s += wall_elapsed_s(dominance_begin);
            }
        }

        const long long layer_survivors = count_bdd_survivors_topdown_layer(bdd, l);
        if (stats != NULL) {
            stats->work_candidates_total += layer_candidates;
            update_peak_candidates(stats, layer_candidates);
            stats->work_frontier_survivors_total += layer_survivors;
            update_peak_points(stats, layer_survivors);
            stats->std_candidates_per_layer[l] = layer_candidates_std;
            stats->std_frontier_survivors_per_layer[l] = std_bdd_survivors_topdown_layer(bdd, l);
        }

        for (size_t i = 0; i < bdd->layers[l - 1].size(); ++i) {
            recycle_frontier(mgmr, bdd->layers[l - 1][i]->pareto_frontier, parallel_mode);
        }
        if (metrics_enabled) {
            stats->cpu_layers_td += 1;
            stats->cpu_nodes_expanded += layer_size;
        }
    }

    ParetoFrontier *frontier = bdd->get_terminal()->pareto_frontier;
    delete mgmr;
    return frontier;
}

ParetoFrontier *topdown_mdd_cpu_enumerate(MDD *mdd, EnumerationStats *stats, int cpu_threads) {
    stats->cpu_state_dominance_s = 0.0;
    stats->dominance_filtered_total = 0;
    reset_cpu_metrics_stats(stats);
    if (stats != NULL) {
        stats->std_candidates_per_layer.assign(mdd->num_layers, 0.0);
        stats->std_frontier_survivors_per_layer.assign(mdd->num_layers, 0.0);
    }
    const bool metrics_enabled = (stats != NULL);
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);

    ParetoFrontierManager *mgmr = new ParetoFrontierManager(mdd->get_width());

    ObjType zero_array[NOBJS];
    std::memset(zero_array, 0, sizeof(ObjType) * NOBJS);
    mdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
    mdd->get_root()->pareto_frontier->add(zero_array);

    for (int l = 1; l < mdd->num_layers; ++l) {
        const long long layer_candidates = count_mdd_candidates_topdown_layer(mdd, l);
        const double layer_candidates_std = std_mdd_candidates_topdown_layer(mdd, l);
        const int layer_size = mdd->layers[l].size();

        expand_layer_topdown_mdd_cpu(mdd, l, mgmr, parallel_mode, threads);
        const long long layer_survivors = count_mdd_survivors_topdown_layer(mdd, l);
        if (stats != NULL) {
            stats->work_candidates_total += layer_candidates;
            update_peak_candidates(stats, layer_candidates);
            stats->work_frontier_survivors_total += layer_survivors;
            update_peak_points(stats, layer_survivors);
            stats->std_candidates_per_layer[l] = layer_candidates_std;
            stats->std_frontier_survivors_per_layer[l] = std_mdd_survivors_topdown_layer(mdd, l);
        }
        if (metrics_enabled) {
            stats->cpu_layers_td += 1;
            stats->cpu_nodes_expanded += layer_size;
        }
    }

    ParetoFrontier *frontier = mdd->get_terminal()->pareto_frontier;
    delete mgmr;
    return frontier;
}

ParetoFrontier *bottomup_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                       const int state_dominance, EnumerationStats *stats,
                                       int cpu_threads) {
    (void)problem_type;
    (void)state_dominance;
    reset_cpu_metrics_stats(stats);
    if (stats != NULL) {
        stats->std_candidates_per_layer.assign(bdd->num_layers, 0.0);
        stats->std_frontier_survivors_per_layer.assign(bdd->num_layers, 0.0);
    }
    const bool metrics_enabled = (stats != NULL);
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);

    ParetoFrontierManager *mgmr = new ParetoFrontierManager(bdd->get_width());

    ObjType zero_array[NOBJS];
    std::memset(zero_array, 0, sizeof(ObjType) * NOBJS);
    bdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
    bdd->get_terminal()->pareto_frontier_bu->add(zero_array);

    if (maximization) {
        for (int l = bdd->num_layers - 2; l >= 0; --l) {
            const long long layer_candidates =
                count_bdd_candidates_bottomup_layer(bdd, l, maximization);
            const double layer_candidates_std =
                std_bdd_candidates_bottomup_layer(bdd, l, maximization);

            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
            for (int i = 0; i < layer_size; ++i) {
                Node *node = bdd->layers[l][i];
                node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
                if (node->arcs[1] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu),
                                                    node->weights[1]);
                }
                if (node->arcs[0] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu),
                                                    node->weights[0]);
                }
            }

            for (size_t i = 0; i < bdd->layers[l + 1].size(); ++i) {
                recycle_frontier(mgmr, bdd->layers[l + 1][i]->pareto_frontier_bu, parallel_mode);
            }
            const long long layer_survivors = count_bdd_survivors_bottomup_layer(bdd, l);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
                stats->std_candidates_per_layer[l] = layer_candidates_std;
                stats->std_frontier_survivors_per_layer[l] =
                    std_bdd_survivors_bottomup_layer(bdd, l);
            }
            if (metrics_enabled) {
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += layer_size;
            }
        }
    } else {
        for (int l = bdd->num_layers - 2; l >= 0; --l) {
            const long long layer_candidates =
                count_bdd_candidates_bottomup_layer(bdd, l, maximization);
            const double layer_candidates_std =
                std_bdd_candidates_bottomup_layer(bdd, l, maximization);

            const int layer_size = bdd->layers[l].size();
            CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
            for (int i = 0; i < layer_size; ++i) {
                Node *node = bdd->layers[l][i];
                node->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
                if (node->arcs[0] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu),
                                                    node->weights[0]);
                }
                if (node->arcs[1] != NULL) {
                    node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu),
                                                    node->weights[1]);
                }
            }

            for (size_t i = 0; i < bdd->layers[l + 1].size(); ++i) {
                recycle_frontier(mgmr, bdd->layers[l + 1][i]->pareto_frontier_bu, parallel_mode);
            }
            const long long layer_survivors = count_bdd_survivors_bottomup_layer(bdd, l);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
                stats->std_candidates_per_layer[l] = layer_candidates_std;
                stats->std_frontier_survivors_per_layer[l] =
                    std_bdd_survivors_bottomup_layer(bdd, l);
            }
            if (metrics_enabled) {
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += layer_size;
            }
        }
    }

    ParetoFrontier *frontier = bdd->get_root()->pareto_frontier_bu;
    delete mgmr;
    return frontier;
}

ParetoFrontier *coupled_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      int cpu_threads) {
    ParetoFrontierManager *mgmr = new ParetoFrontierManager(bdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    reset_cpu_metrics_stats(stats);
    const bool metrics_enabled = (stats != NULL);

    ObjType sol[NOBJS];
    std::memset(sol, 0, sizeof(ObjType) * NOBJS);

    bdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
    bdd->get_root()->pareto_frontier->add(sol);

    bdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
    bdd->get_terminal()->pareto_frontier_bu->add(sol);

    stats->cpu_state_dominance_s = 0.0;
    stats->dominance_filtered_total = 0;
    clock_t init;

    int layer_topdown = 0;
    int layer_bottomup = bdd->num_layers - 1;

    int val_topdown = 0;
    int val_bottomup = 0;
    while (layer_topdown != layer_bottomup) {
        if (val_topdown <= val_bottomup) {
            const int next_layer = layer_topdown + 1;
            const long long layer_candidates =
                count_bdd_candidates_topdown_layer(bdd, next_layer, maximization);
            ++layer_topdown;
            expand_layer_topdown(bdd, layer_topdown, maximization, mgmr, threads);
            if (metrics_enabled) {
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_topdown].size();
            }
            val_topdown = 0;
            const int layer_size = bdd->layers[layer_topdown].size();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_topdown)
            for (int i = 0; i < layer_size; ++i) {
                val_topdown += topdown_layer_value(bdd, bdd->layers[layer_topdown][i]);
            }
            if (state_dominance > 0) {
                const WallClock::time_point dominance_begin =
                    metrics_enabled ? WallClock::now() : WallClock::time_point();
                init = clock();
                filter_dominance_cpu(bdd, layer_topdown, problem_type, state_dominance, stats);
                stats->cpu_state_dominance_s +=
                    static_cast<double>(clock() - init) / CLOCKS_PER_SEC;
                if (metrics_enabled) {
                    stats->wall_state_dominance_s += wall_elapsed_s(dominance_begin);
                }
            }
            const long long layer_survivors = count_bdd_survivors_topdown_layer(bdd, layer_topdown);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
        } else {
            const int next_layer = layer_bottomup - 1;
            const long long layer_candidates =
                count_bdd_candidates_bottomup_layer(bdd, next_layer, maximization);
            --layer_bottomup;
            expand_layer_bottomup(bdd, layer_bottomup, maximization, mgmr, threads);
            if (metrics_enabled) {
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += bdd->layers[layer_bottomup].size();
            }
            val_bottomup = 0;
            const int layer_size = bdd->layers[layer_bottomup].size();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_bottomup)
            for (int i = 0; i < layer_size; ++i) {
                val_bottomup += bottomup_layer_value(bdd, bdd->layers[layer_bottomup][i]);
            }
            const long long layer_survivors =
                count_bdd_survivors_bottomup_layer(bdd, layer_bottomup);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
        }
    }

    stats->layer_coupling = layer_topdown;

    std::vector<Node *> &cutset = bdd->layers[layer_topdown];
    if (metrics_enabled) {
        stats->cpu_cutset_size = cutset.size();
    }

    const WallClock::time_point cutset_sort_begin =
        metrics_enabled ? WallClock::now() : WallClock::time_point();
    std::sort(cutset.begin(), cutset.end(), CompareNode());
    if (metrics_enabled) {
        stats->wall_cutset_sort_s += wall_elapsed_s(cutset_sort_begin);
    }

    long int expected_size = 0;
    for (size_t i = 0; i < cutset.size(); ++i) {
        expected_size += cutset[i]->pareto_frontier->get_num_sols() *
                         cutset[i]->pareto_frontier_bu->get_num_sols();
    }
    if (stats != NULL) {
        stats->work_join_products_total += expected_size;
    }
    expected_size = 10000;

    ParetoFrontier *paretoFrontier = NULL;

    if (parallel_mode && cutset.size() > 1) {
        const WallClock::time_point join_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        std::vector<ParetoFrontier *> partial(threads, NULL);
        CUMODD_OMP_PARALLEL_NUM_THREADS(threads) {
            const int tid = cumodd_omp_thread_num();
            ParetoFrontier *local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
            CUMODD_OMP_FOR_DYNAMIC
            for (size_t i = 0; i < cutset.size(); ++i) {
                Node *node = cutset[i];
                assert(node->pareto_frontier != NULL);
                assert(node->pareto_frontier_bu != NULL);
                local_frontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
            }
        }
        if (metrics_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        paretoFrontier = parallel_reduce_partial_frontiers(partial, parallel_mode, threads);
        if (metrics_enabled) {
            stats->wall_cutset_partial_merge_s += wall_elapsed_s(partial_merge_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    } else {
        paretoFrontier = new ParetoFrontier;
        paretoFrontier->sols.reserve(expected_size * NOBJS);
        const WallClock::time_point join_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        for (size_t i = 0; i < cutset.size(); ++i) {
            Node *node = cutset[i];
            assert(node->pareto_frontier != NULL);
            assert(node->pareto_frontier_bu != NULL);
            paretoFrontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
        }
        if (metrics_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    }
    if (stats != NULL) {
        const long long join_survivors = paretoFrontier->get_num_sols();
        stats->work_frontier_survivors_total += join_survivors;
        update_peak_points(stats, join_survivors);
    }

    delete mgmr;
    return paretoFrontier;
}

ParetoFrontier *coupled_mdd_cpu_enumerate(MDD *mdd, EnumerationStats *stats, int cpu_threads) {
    ParetoFrontierManager *mgmr = new ParetoFrontierManager(mdd->get_width());
    const int threads = cumodd_normalized_cpu_threads(cpu_threads);
    const bool parallel_mode = cumodd_use_parallel_cpu(threads);
    reset_cpu_metrics_stats(stats);
    const bool metrics_enabled = (stats != NULL);

    ObjType sol[NOBJS];
    std::memset(sol, 0, sizeof(ObjType) * NOBJS);

    mdd->get_root()->pareto_frontier = request_frontier(mgmr, parallel_mode);
    mdd->get_root()->pareto_frontier->add(sol);

    mdd->get_terminal()->pareto_frontier_bu = request_frontier(mgmr, parallel_mode);
    mdd->get_terminal()->pareto_frontier_bu->add(sol);

    stats->cpu_state_dominance_s = 0.0;
    stats->dominance_filtered_total = 0;

    int layer_topdown = 0;
    int layer_bottomup = mdd->num_layers - 1;

    int val_topdown = 0;
    int val_bottomup = 0;
    while (layer_topdown != layer_bottomup) {
        if (val_topdown <= val_bottomup) {
            const int next_layer = layer_topdown + 1;
            const long long layer_candidates = count_mdd_candidates_topdown_layer(mdd, next_layer);
            ++layer_topdown;
            expand_layer_topdown_mdd_cpu(mdd, layer_topdown, mgmr, parallel_mode, threads);
            if (metrics_enabled) {
                stats->cpu_layers_td += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_topdown].size();
            }
            val_topdown = 0;
            const int layer_size = mdd->layers[layer_topdown].size();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_topdown)
            for (int i = 0; i < layer_size; ++i) {
                val_topdown += topdown_layer_value(mdd, mdd->layers[layer_topdown][i]);
            }
            const long long layer_survivors = count_mdd_survivors_topdown_layer(mdd, layer_topdown);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
        } else {
            const int next_layer = layer_bottomup - 1;
            const long long layer_candidates = count_mdd_candidates_bottomup_layer(mdd, next_layer);
            --layer_bottomup;
            expand_layer_bottomup_mdd_cpu(mdd, layer_bottomup, mgmr, parallel_mode, threads);
            if (metrics_enabled) {
                stats->cpu_layers_bu += 1;
                stats->cpu_nodes_expanded += mdd->layers[layer_bottomup].size();
            }
            val_bottomup = 0;
            const int layer_size = mdd->layers[layer_bottomup].size();
            CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, val_bottomup)
            for (int i = 0; i < layer_size; ++i) {
                val_bottomup += bottomup_layer_value(mdd, mdd->layers[layer_bottomup][i]);
            }
            const long long layer_survivors =
                count_mdd_survivors_bottomup_layer(mdd, layer_bottomup);
            if (stats != NULL) {
                stats->work_candidates_total += layer_candidates;
                update_peak_candidates(stats, layer_candidates);
                stats->work_frontier_survivors_total += layer_survivors;
                update_peak_points(stats, layer_survivors);
            }
        }
    }

    stats->layer_coupling = layer_topdown;

    std::vector<MDDNode *> &cutset = mdd->layers[layer_topdown];
    if (metrics_enabled) {
        stats->cpu_cutset_size = cutset.size();
    }
    const WallClock::time_point cutset_sort_begin =
        metrics_enabled ? WallClock::now() : WallClock::time_point();
    std::sort(cutset.begin(), cutset.end(), CompareMDDNode());
    if (metrics_enabled) {
        stats->wall_cutset_sort_s += wall_elapsed_s(cutset_sort_begin);
    }

    long int expected_size = 0;
    for (size_t i = 0; i < cutset.size(); ++i) {
        expected_size += cutset[i]->pareto_frontier->get_num_sols() *
                         cutset[i]->pareto_frontier_bu->get_num_sols();
    }
    if (stats != NULL) {
        stats->work_join_products_total += expected_size;
    }
    expected_size = 10000;

    ParetoFrontier *paretoFrontier = NULL;

    if (parallel_mode && cutset.size() > 1) {
        const WallClock::time_point join_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        std::vector<ParetoFrontier *> partial(threads, NULL);
        CUMODD_OMP_PARALLEL_NUM_THREADS(threads) {
            const int tid = cumodd_omp_thread_num();
            ParetoFrontier *local_frontier = new ParetoFrontier;
            partial[tid] = local_frontier;
            CUMODD_OMP_FOR_DYNAMIC
            for (size_t i = 0; i < cutset.size(); ++i) {
                MDDNode *node = cutset[i];
                assert(node->pareto_frontier != NULL);
                assert(node->pareto_frontier_bu != NULL);
                local_frontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
            }
        }
        if (metrics_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
        }
        const WallClock::time_point partial_merge_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        paretoFrontier = parallel_reduce_partial_frontiers(partial, parallel_mode, threads);
        if (metrics_enabled) {
            stats->wall_cutset_partial_merge_s += wall_elapsed_s(partial_merge_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    } else {
        paretoFrontier = new ParetoFrontier;
        paretoFrontier->sols.reserve(expected_size * NOBJS);
        const WallClock::time_point join_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        const WallClock::time_point convolution_begin =
            metrics_enabled ? WallClock::now() : WallClock::time_point();
        for (size_t i = 0; i < cutset.size(); ++i) {
            MDDNode *node = cutset[i];
            assert(node->pareto_frontier != NULL);
            assert(node->pareto_frontier_bu != NULL);
            paretoFrontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
        }
        if (metrics_enabled) {
            stats->wall_cutset_convolution_s += wall_elapsed_s(convolution_begin);
            stats->wall_join_s += wall_elapsed_s(join_begin);
        }
    }
    if (stats != NULL) {
        const long long join_survivors = paretoFrontier->get_num_sols();
        stats->work_frontier_survivors_total += join_survivors;
        update_peak_points(stats, join_survivors);
    }

    delete mgmr;
    return paretoFrontier;
}
