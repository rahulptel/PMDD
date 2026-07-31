// --------------------------------------------------
// OpenMP compatibility facade
// --------------------------------------------------

#ifndef OMP_COMPAT_HPP_
#define OMP_COMPAT_HPP_

#if defined(_OPENMP)
#define CUMODD_HAS_OPENMP 1
#include <omp.h>
#else
#define CUMODD_HAS_OPENMP 0
#endif

#define CUMODD_OMP_PRAGMA_IMPL(x) _Pragma(#x)
#define CUMODD_OMP_PRAGMA(x) CUMODD_OMP_PRAGMA_IMPL(x)

#if CUMODD_HAS_OPENMP
#define CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)                                 \
    CUMODD_OMP_PRAGMA(omp parallel for if(parallel_mode) num_threads(threads) schedule(dynamic))
#define CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, var)                      \
    CUMODD_OMP_PRAGMA(omp parallel for if(parallel_mode) num_threads(threads) reduction(+:var))
#define CUMODD_OMP_PARALLEL_NUM_THREADS(threads)                                                   \
    CUMODD_OMP_PRAGMA(omp parallel num_threads(threads))
#define CUMODD_OMP_FOR_DYNAMIC                                                                     \
    CUMODD_OMP_PRAGMA(omp for schedule(dynamic))
#else
#define CUMODD_OMP_PARALLEL_FOR_DYNAMIC_IF(parallel_mode, threads)
#define CUMODD_OMP_PARALLEL_FOR_REDUCTION_SUM_IF(parallel_mode, threads, var)
#define CUMODD_OMP_PARALLEL_NUM_THREADS(threads)
#define CUMODD_OMP_FOR_DYNAMIC
#endif

inline int cumodd_normalized_cpu_threads(const int cpu_threads) {
    return cpu_threads > 0 ? cpu_threads : 1;
}

inline bool cumodd_use_parallel_cpu(const int cpu_threads) {
#if CUMODD_HAS_OPENMP
    return cumodd_normalized_cpu_threads(cpu_threads) > 1;
#else
    (void)cpu_threads;
    return false;
#endif
}

inline int cumodd_omp_thread_num() {
#if CUMODD_HAS_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

#include "cpu_affinity.hpp"

#endif
