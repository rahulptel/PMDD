#ifndef OUTPUT_UTILS_HPP_
#define OUTPUT_UTILS_HPP_

#include "../enum/pareto_frontier.hpp"
#include "cli_parser.hpp"
#include "stats.hpp"
#include <string>

//
// Output utilities
//

bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const std::string &out_path,
                             std::string *error);

bool write_stats_jsonl(const std::string &out_path, const CliOptions &opts,
                       const EnumerationStats *stats, const DDStats &record, std::string *error);

void print_and_save_run_summary(const CliOptions &opts, const EnumerationStats *enumeration_stats,
                                const DDStats &run_summary, const ParetoFrontier *pareto_frontier);

#endif /* OUTPUT_UTILS_HPP_ */
