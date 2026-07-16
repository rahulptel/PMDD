#include "util.hpp"
#include <boost/unordered_map.hpp>
#include <sys/resource.h>

using namespace boost;

long long get_cpu_peak_memory_bytes() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
#ifdef __APPLE__
    return r_usage.ru_maxrss;
#else
    return r_usage.ru_maxrss * 1024LL; // ru_maxrss is in KB on Linux
#endif
}

//
// Equality functions to dynamic_bitset pointer
//
bool bitset_equal_to::operator()(const boost::dynamic_bitset<> *const x,
                                 const boost::dynamic_bitset<> *const y) const {
    return (*x == *y);
}

//
// Hash functions to dynamic_bitset pointer
//
std::size_t bitset_hash::operator()(const boost::dynamic_bitset<> *const x) const {
    return boost::hash_value(x->m_bits);
}
