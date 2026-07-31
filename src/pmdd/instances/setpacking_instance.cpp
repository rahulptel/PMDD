#include "setpacking_instance.hpp"
#include <cstdlib>

namespace {
int compute_bandwidth(const vector<vector<int>> &vars_cons) {
    int bw = 0;
    for (size_t c = 0; c < vars_cons.size(); ++c) {
        for (size_t i = 0; i < vars_cons[c].size(); ++i) {
            for (size_t j = i + 1; j < vars_cons[c].size(); ++j) {
                bw = std::max(bw, std::abs(vars_cons[c][i] - vars_cons[c][j]));
            }
        }
    }
    return bw;
}
} // namespace

// Apply heuristic to minimize bandwidth
void SetPackingInstance::minimize_bandwidth() {
    // CPLEX-based variable reordering has been removed.
    // Keep current ordering and only report current bandwidth.
    bandwidth = compute_bandwidth(vars_cons);
}
