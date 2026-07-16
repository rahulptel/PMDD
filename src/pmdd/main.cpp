// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "enum/multiobj_enum.hpp"
#include "enum/pareto_frontier.hpp"
#include "util/cli_parser.hpp"
#include "util/omp_compat.hpp"
#include "util/output_utils.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"

// Knapsack includes
#include "bdd/knapsack_bdd.hpp"
#include "instances/knapsack_instance.hpp"

// Set packing / Independent set includes
#include "bdd/indepset_bdd.hpp"
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;

static vector<long> compute_bdd_max_num_nodes_per_layer(const BDD *bdd) {
    vector<long> max_num_nodes_per_layer;
    if (bdd == NULL) {
        return max_num_nodes_per_layer;
    }
    max_num_nodes_per_layer.reserve(bdd->num_layers);
    for (int l = 0; l < bdd->num_layers; ++l) {
        max_num_nodes_per_layer.push_back(static_cast<long>(bdd->layers[l].size()));
    }
    return max_num_nodes_per_layer;
}

static vector<long> compute_mdd_max_num_nodes_per_layer(const MDD *mdd) {
    vector<long> max_num_nodes_per_layer;
    if (mdd == NULL) {
        return max_num_nodes_per_layer;
    }
    max_num_nodes_per_layer.reserve(mdd->num_layers);
    for (int l = 0; l < mdd->num_layers; ++l) {
        max_num_nodes_per_layer.push_back(static_cast<long>(mdd->layers[l].size()));
    }
    return max_num_nodes_per_layer;
}

//
// Main function
//
int main(int argc, char *argv[]) {
    CliOptions options;
    string parse_error;
    if (!parse_cli_args(argc, argv, &options, &parse_error)) {
        if (!parse_error.empty()) {
            cout << parse_error << endl;
        }
        print_usage();
        exit(1);
    }

    const string input_path = options.input_path;
    const int problem_type = options.problem_type;
    const int method = options.method;
    bool maximization = true;
    const int state_dominance = options.state_dominance;
    const Backend backend = options.backend;
    const int cpu_threads = options.cpu_threads;
    const long long gpu_batch_size = options.gpu_batch_size;
    const long long gpu_max_prod = options.gpu_max_prod;
    const bool save_frontier = options.save_frontier;
    const string frontier_out_path = options.frontier_out_path;
    const bool save_stats = options.save_stats;
    const string stats_out_path = options.stats_out_path;

    if (backend == BACKEND_CPU) {
        cumodd_bind_openmp_threads_to_cores_if_applicable(cpu_threads);
    }

    typedef std::chrono::steady_clock WallClock;
    const WallClock::time_point run_wall_begin = WallClock::now();
    long int original_width;
    long int reduced_width;
    long int original_num_nodes;
    long int reduced_num_nodes;
    long int max_num_in_arcs = 0;
    long int total_num_in_arcs = 0;
    vector<long> max_num_nodes_per_layer;

    // Read problem instance and construct BDD
    BDD *bdd = NULL;
    vector<vector<int>> obj_coeffs;
    const WallClock::time_point compilation_wall_begin = WallClock::now();
    const clock_t compilation_cpu_begin = clock();

    // --- Knapsack ---
    if (problem_type == 1) {

        // Read instance
        KnapsackInstance inst;
        inst.read(const_cast<char *>(input_path.c_str()));

        // Construct BDD
        KnapsackBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        // obj_coeffs = inst.obj_coeffs;

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // cout << "Original width: " << original_width << " - number of nodes: " <<
        // original_num_nodes << endl;

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced width: " << reduced_width << " - number of nodes: " << reduced_num_nodes
        // << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced-2 width: " << reduced_width << " - number of nodes: " <<
        // reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);
        max_num_nodes_per_layer = compute_bdd_max_num_nodes_per_layer(bdd);
        max_num_in_arcs = bdd->get_max_num_in_arcs();
        total_num_in_arcs = bdd->get_total_num_in_arcs();

        //        bdd->print();
    }

    // --- Set Packing ---
    else if (problem_type == 2) {

        // read instance
        SetPackingInstance setpack(input_path.c_str());

        // create associated independent set instance
        IndepSetInst *inst = setpack.create_indepset_instance();

        // generate independent set BDD
        IndepSetBDDConstructor bddConstructor(inst, setpack.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
        max_num_nodes_per_layer = compute_bdd_max_num_nodes_per_layer(bdd);
        max_num_in_arcs = bdd->get_max_num_in_arcs();
        total_num_in_arcs = bdd->get_total_num_in_arcs();
    }

    // --- TSP ---
    else if (problem_type == 3) {
        // Read instance
        TSPInstance inst;
        inst.read(input_path.c_str());

        // Construct MDD
        const WallClock::time_point compilation_tsp_wall_begin = WallClock::now();
        clock_t compilation_tsp = clock();

        MDDTSPConstructor mddCons(&inst);
        MDD *mdd = mddCons.generate_exact();
        assert(mdd != NULL);
        const vector<long> mdd_max_num_nodes_per_layer = compute_mdd_max_num_nodes_per_layer(mdd);
        long mdd_max_num_in_arcs = mdd->get_max_num_in_arcs();
        long mdd_total_num_in_arcs = mdd->get_total_num_in_arcs();

        compilation_tsp = clock() - compilation_tsp;
        const double compilation_tsp_wall_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(WallClock::now() -
                                                                      compilation_tsp_wall_begin)
                .count();

        // Generate frontier (timed region excludes final lexicographic sort)
        const WallClock::time_point pareto_tsp_wall_begin = WallClock::now();
        clock_t pareto_tsp_cpu = clock();

        // Solver-owned stats populated during frontier enumeration.
        EnumerationStats *enumeration_stats = new EnumerationStats;
        ParetoFrontier *pareto_frontier = NULL;

        if (method == 1) { // Top-down
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = MultiobjEnum::pareto_frontier_topdown_cuda(
                    mdd, enumeration_stats, &cuda_reason, gpu_batch_size);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but top-down enumeration failed";
                    if (!cuda_reason.empty())
                        cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                try {
                    pareto_frontier =
                        MultiobjEnum::pareto_frontier_topdown(mdd, enumeration_stats, cpu_threads);
                } catch (const std::exception &e) {
                    cout << "Error - CPU backend enumeration failed: " << e.what() << endl;
                    exit(1);
                }
            }
        } else if (method == 3) { // Coupled
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = MultiobjEnum::pareto_frontier_dynamic_layer_cutset_cuda(
                    mdd, enumeration_stats, &cuda_reason, gpu_batch_size, gpu_max_prod);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but coupled enumeration failed";
                    if (!cuda_reason.empty())
                        cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                try {
                    pareto_frontier = MultiobjEnum::pareto_frontier_dynamic_layer_cutset(
                        mdd, enumeration_stats, cpu_threads);
                } catch (const std::exception &e) {
                    cout << "Error - CPU backend enumeration failed: " << e.what() << endl;
                    exit(1);
                }
            }
        } else {
            cout << "Error - method " << method << " not valid for TSP" << endl;
            exit(1);
        }
        pareto_tsp_cpu = clock() - pareto_tsp_cpu;
        const double pareto_tsp_wall_enumeration_s =
            std::chrono::duration_cast<std::chrono::duration<double>>(WallClock::now() -
                                                                      pareto_tsp_wall_begin)
                .count();

        assert(pareto_frontier != NULL);
        pareto_frontier->sort_lexicographic_ascending();

        if (save_frontier) {
            string save_error;
            if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error)) {
                cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
                if (!save_error.empty()) {
                    cout << ": " << save_error;
                }
                cout << endl;
                exit(1);
            }
        }

        // Run-level summary assembled in main for reporting/output only.
        DDStats run_summary;
        run_summary.original_width = -1;
        run_summary.reduced_width = -1;
        run_summary.original_num_nodes = -1;
        run_summary.reduced_num_nodes = -1;
        run_summary.max_num_nodes_per_layer = mdd_max_num_nodes_per_layer;
        run_summary.max_num_in_arcs = mdd_max_num_in_arcs;
        run_summary.total_num_in_arcs = mdd_total_num_in_arcs;

        enumeration_stats->num_solutions = pareto_frontier->get_num_sols();
        enumeration_stats->cpu_compile_s = ((double)compilation_tsp) / CLOCKS_PER_SEC;
        enumeration_stats->cpu_enumeration_s = ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC;
        enumeration_stats->cpu_total_s =
            enumeration_stats->cpu_compile_s + enumeration_stats->cpu_enumeration_s;
        enumeration_stats->wall_compile_s = compilation_tsp_wall_s;
        enumeration_stats->wall_enumeration_s = pareto_tsp_wall_enumeration_s;
        enumeration_stats->cpu_mem_peak_bytes = get_cpu_peak_memory_bytes();

        print_and_save_run_summary(options, enumeration_stats, run_summary, pareto_frontier);

        return 0;
    } else {
        cout << "Error - problem type not recognized" << endl;
        exit(1);
    }

    const clock_t compilation_cpu_elapsed = clock() - compilation_cpu_begin;
    const double compilation_wall_s = std::chrono::duration_cast<std::chrono::duration<double>>(
                                          WallClock::now() - compilation_wall_begin)
                                          .count();

    // cout << "\nBDD Info:\n";
    // cout << "\tOriginal width: " << original_width << endl;
    // cout << "\tOriginal number of nodes: " << original_num_nodes << endl;
    // cout << "\n\tReduced width: " << reduced_width << endl;
    // cout << "\tReduced number of nodes: " << reduced_num_nodes << endl;
    // cout << "\n\tBDD compilation total time: " <<
    // ((double)compilation_cpu_elapsed)/CLOCKS_PER_SEC << endl;

    // Initialize enumeration stats
    // Solver-owned stats populated during frontier enumeration.
    EnumerationStats *enumeration_stats = new EnumerationStats;

    // Compute pareto frontier based on methodology
    // cout << "\n\nComputing pareto frontier..." << endl;
    ParetoFrontier *pareto_frontier = NULL;
    const WallClock::time_point pareto_wall_begin = WallClock::now();
    const clock_t pareto_cpu_begin = clock();

    if (method == 1) {
        // -- Optimal BFS algorithm: top-down --
        if (backend == BACKEND_GPU) {
            string cuda_reason;
            pareto_frontier = MultiobjEnum::pareto_frontier_topdown_cuda(
                bdd, maximization, problem_type, state_dominance, enumeration_stats, &cuda_reason,
                gpu_batch_size);
            if (pareto_frontier == NULL) {
                cout << "Error - GPU backend requested but top-down enumeration failed";
                if (!cuda_reason.empty()) {
                    cout << ": " << cuda_reason;
                }
                cout << endl;
                exit(1);
            }
        } else {
            try {
                pareto_frontier = MultiobjEnum::pareto_frontier_topdown(
                    bdd, maximization, problem_type, state_dominance, enumeration_stats,
                    cpu_threads);
            } catch (const std::exception &e) {
                cout << "Error - CPU backend enumeration failed: " << e.what() << endl;
                exit(1);
            }
        }
    } else if (method == 2) {
        // -- Optimal BFS algorithm: bottom-up --
        if (backend == BACKEND_GPU) {
            cout << "Error - GPU backend is unsupported for method 2." << endl;
            exit(1);
        }
        pareto_frontier = MultiobjEnum::pareto_frontier_bottomup(
            bdd, maximization, problem_type, state_dominance, enumeration_stats, cpu_threads);
    } else if (method == 3) {
        // -- Dynamic layer cutset --
        if (backend == BACKEND_GPU) {
            string cuda_reason;
            pareto_frontier = MultiobjEnum::pareto_frontier_dynamic_layer_cutset_cuda(
                bdd, maximization, problem_type, state_dominance, enumeration_stats, &cuda_reason,
                gpu_batch_size, gpu_max_prod);
            if (pareto_frontier == NULL) {
                cout << "Error - GPU backend requested but coupled enumeration failed";
                if (!cuda_reason.empty())
                    cout << ": " << cuda_reason;
                cout << endl;
                exit(1);
            }
        } else {
            try {
                pareto_frontier = MultiobjEnum::pareto_frontier_dynamic_layer_cutset(
                    bdd, maximization, problem_type, state_dominance, enumeration_stats,
                    cpu_threads);
            } catch (const std::exception &e) {
                cout << "Error - CPU backend enumeration failed: " << e.what() << endl;
                exit(1);
            }
        }
    } else {
        cout << "Error - method not recognized" << endl;
        exit(1);
    }

    if (pareto_frontier == NULL) {
        cout << "\nError - pareto frontier not computed" << endl;
        exit(1);
    }
    const clock_t pareto_cpu_elapsed = clock() - pareto_cpu_begin;
    const double pareto_wall_enumeration_s =
        std::chrono::duration_cast<std::chrono::duration<double>>(WallClock::now() -
                                                                  pareto_wall_begin)
            .count();

    pareto_frontier->sort_lexicographic_ascending();

    if (save_frontier) {
        string save_error;
        if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error)) {
            cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
            if (!save_error.empty()) {
                cout << ": " << save_error;
            }
            cout << endl;
            exit(1);
        }
    }

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << ((double)compilation_cpu_elapsed)/CLOCKS_PER_SEC << endl;
    // cout << "\tPareto time: " << ((double)pareto_cpu_elapsed)/CLOCKS_PER_SEC << endl;
    // cout << "\tTotal time: " << (((double)(compilation_cpu_elapsed +
    // pareto_cpu_elapsed))/CLOCKS_PER_SEC) << endl; cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << method;
    // stats << "\t" << pareto_frontier->get_num_sols();
    // stats << "\t" << original_width;
    // stats << "\t" << original_num_nodes;
    // stats << "\t" << reduced_width;
    // stats << "\t" << reduced_num_nodes;
    // stats << "\t" << ((double)compilation_cpu_elapsed)/CLOCKS_PER_SEC;
    // stats << "\t" << ((double)pareto_cpu_elapsed)/CLOCKS_PER_SEC;
    // stats << "\t" << (((double)(compilation_cpu_elapsed + pareto_cpu_elapsed))/CLOCKS_PER_SEC);
    // stats << endl;
    // stats.close();

    // Run-level summary assembled in main for reporting/output only.
    DDStats run_summary;
    run_summary.original_width = original_width;
    run_summary.reduced_width = reduced_width;
    run_summary.original_num_nodes = original_num_nodes;
    run_summary.reduced_num_nodes = reduced_num_nodes;
    run_summary.max_num_nodes_per_layer = max_num_nodes_per_layer;
    run_summary.max_num_in_arcs = max_num_in_arcs;
    run_summary.total_num_in_arcs = total_num_in_arcs;

    enumeration_stats->num_solutions = pareto_frontier->get_num_sols();
    enumeration_stats->cpu_compile_s = ((double)compilation_cpu_elapsed) / CLOCKS_PER_SEC;
    enumeration_stats->cpu_enumeration_s = ((double)pareto_cpu_elapsed) / CLOCKS_PER_SEC;
    enumeration_stats->cpu_total_s =
        enumeration_stats->cpu_compile_s + enumeration_stats->cpu_enumeration_s;
    enumeration_stats->wall_compile_s = compilation_wall_s;
    enumeration_stats->wall_enumeration_s = pareto_wall_enumeration_s;
    enumeration_stats->cpu_mem_peak_bytes = get_cpu_peak_memory_bytes();

    print_and_save_run_summary(options, enumeration_stats, run_summary, pareto_frontier);

    return 0;
}
