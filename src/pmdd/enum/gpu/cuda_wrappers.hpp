#pragma once

#include <string>

#include "../../bdd/bdd.hpp"
#include "../../mdd/mdd.hpp"
#include "../multiobj_enum.hpp"
#include "../pareto_frontier.hpp"

// Declarations of CUDA functions that are either real or stubs depending on ENABLE_CUDA.
ParetoFrontier *topdown_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                       const int state_dominance, EnumerationStats *stats,
                                       std::string *reason, long long gpu_batch_size);
ParetoFrontier *topdown_mdd_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                           long long gpu_batch_size);
ParetoFrontier *coupled_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                       long long gpu_batch_size, long long gpu_max_prod);
ParetoFrontier *coupled_bdd_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                           const int state_dominance, EnumerationStats *stats,
                                           std::string *reason, long long gpu_batch_size,
                                           long long gpu_max_prod);
