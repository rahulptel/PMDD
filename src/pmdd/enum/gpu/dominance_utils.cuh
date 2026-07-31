// ----------------------------------------------------------
// Shared CUDA dominance helpers
// ----------------------------------------------------------

#ifndef CUDA_DOMINANCE_UTILS_CUH_
#define CUDA_DOMINANCE_UTILS_CUH_

#include "../../util/util.hpp"

static __device__ __forceinline__ bool
dominates_or_tie_before(const ObjType *lhs, const ObjType *rhs, bool tie_before) {
    bool strict = false;
#pragma unroll
    for (int o = 0; o < NOBJS; ++o) {
        const ObjType a = lhs[o];
        const ObjType b = rhs[o];
        if (a < b) {
            return false;
        }
        strict = strict || (a > b);
    }
    return strict || tie_before;
}

#endif
