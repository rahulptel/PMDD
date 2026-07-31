// Stub implementations of CUDA wrappers used when compiling without USE_CUDA

#include "../../bdd/bdd.hpp"
#include "../../mdd/mdd.hpp"
#include "../multiobj_enum.hpp"
#include "../pareto_frontier.hpp"
#include "cuda_wrappers.hpp"

ParetoFrontier *topdown_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                       const int state_dominance, EnumerationStats *stats,
                                       std::string *reason, long long gpu_batch_size) {
    (void)bdd;
    (void)maximization;
    (void)problem_type;
    (void)state_dominance;
    (void)stats;
    (void)gpu_batch_size;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier *topdown_mdd_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                           long long gpu_batch_size) {
    (void)mdd;
    (void)stats;
    (void)gpu_batch_size;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier *coupled_cuda_enumerate(MDD *mdd, EnumerationStats *stats, std::string *reason,
                                       long long gpu_batch_size, long long gpu_max_prod) {
    (void)mdd;
    (void)stats;
    (void)gpu_batch_size;
    (void)gpu_max_prod;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}

ParetoFrontier *coupled_bdd_cuda_enumerate(BDD *bdd, bool maximization, const int problem_type,
                                           const int state_dominance, EnumerationStats *stats,
                                           std::string *reason, long long gpu_batch_size,
                                           long long gpu_max_prod) {
    (void)bdd;
    (void)maximization;
    (void)problem_type;
    (void)state_dominance;
    (void)stats;
    (void)gpu_batch_size;
    (void)gpu_max_prod;
    if (reason != NULL) {
        *reason = "GPU backend requested but binary was built without CUDA support";
    }
    return NULL;
}
