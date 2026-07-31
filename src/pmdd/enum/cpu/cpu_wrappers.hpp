// ----------------------------------------------------------
// CPU Enumeration Wrappers
// ----------------------------------------------------------

#ifndef CPU_WRAPPERS_HPP_
#define CPU_WRAPPERS_HPP_

#include "../../bdd/bdd.hpp"
#include "../../mdd/mdd.hpp"
#include "../../util/stats.hpp"
#include "../pareto_frontier.hpp"
#include <string>

// CPU wrapper functions that implement the enumeration methods

ParetoFrontier *topdown_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      int cpu_threads);

ParetoFrontier *topdown_mdd_cpu_enumerate(MDD *mdd, EnumerationStats *stats, int cpu_threads);

ParetoFrontier *bottomup_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                       const int state_dominance, EnumerationStats *stats,
                                       int cpu_threads);

ParetoFrontier *coupled_cpu_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                      const int state_dominance, EnumerationStats *stats,
                                      int cpu_threads);

ParetoFrontier *coupled_mdd_cpu_enumerate(MDD *mdd, EnumerationStats *stats, int cpu_threads);

// CPU state dominance filtering functions

void filter_dominance_cpu(BDD *bdd, const int layer, const int problem_type,
                          const int state_dominance, EnumerationStats *stats);

void filter_dominance_knapsack_cpu(BDD *bdd, const int layer, EnumerationStats *stats);

void filter_dominance_setpacking_cpu(BDD *bdd, const int layer, EnumerationStats *stats);

void filter_completion_cpu(BDD *bdd, const int layer);

#endif // CPU_WRAPPERS_HPP_
