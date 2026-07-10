#include "output_utils.hpp"
#include "util.hpp"
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

static string shell_single_quote(const string &value) {
    string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\"'\"'";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static string json_escape(const string &value) {
    string escaped;
    escaped.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c) {
        case '\"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                static const char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[(c >> 4) & 0x0F];
                escaped += hex[c & 0x0F];
            } else {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }
    return escaped;
}

static void write_json_long_array(ostream &out, const vector<long> &values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
}

static void write_json_double_array(ostream &out, const vector<double> &values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
}

bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const string &out_path,
                             string *error) {
    if (frontier == NULL) {
        if (error != NULL) {
            *error = "frontier is null";
        }
        return false;
    }

    if (out_path.empty()) {
        if (error != NULL) {
            *error = "output path is empty";
        }
        return false;
    }

    if (frontier->sols.size() % NOBJS != 0) {
        if (error != NULL) {
            *error = "frontier has invalid dimension";
        }
        return false;
    }

    const string command = "gzip -c > " + shell_single_quote(out_path);
    FILE *pipe = popen(command.c_str(), "w");
    if (pipe == NULL) {
        if (error != NULL) {
            *error = "could not launch gzip";
        }
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < frontier->sols.size() && ok; i += NOBJS) {
        for (int o = 0; o < NOBJS; ++o) {
            if (o > 0 && fputc(',', pipe) == EOF) {
                ok = false;
                break;
            }
            if (fprintf(pipe, "%d", frontier->sols[i + o]) < 0) {
                ok = false;
                break;
            }
        }
        if (ok && fputc('\n', pipe) == EOF) {
            ok = false;
        }
    }

    int close_status = pclose(pipe);
    if (!ok) {
        if (error != NULL) {
            *error = "failed while writing compressed frontier";
        }
        return false;
    }

    if (close_status != 0) {
        if (error != NULL) {
            *error = "gzip exited with non-zero status";
        }
        return false;
    }

    return true;
}

bool write_stats_jsonl(const string &out_path, const CliOptions &opts,
                       const EnumerationStats *stats, const DDStats &record, string *error) {
    ofstream out(out_path.c_str(), ios::out | ios::app);
    if (!out.is_open()) {
        if (error != NULL) {
            *error = "could not open output path";
        }
        return false;
    }

    out << fixed << setprecision(6);
    const double cpu_compile_s = stats != NULL ? stats->cpu_compile_s : 0.0;
    const double cpu_enumeration_s = stats != NULL ? stats->cpu_enumeration_s : 0.0;
    const double cpu_total_s = stats != NULL ? stats->cpu_total_s : 0.0;
    const double cpu_state_dominance_s = stats != NULL ? stats->cpu_state_dominance_s : 0.0;
    const double wall_compile_s = stats != NULL ? stats->wall_compile_s : 0.0;
    const double wall_enumeration_s = stats != NULL ? stats->wall_enumeration_s : 0.0;
    const double kernel_expand_td_s = stats != NULL ? stats->kernel_expand_td_s : 0.0;
    const double kernel_dominance_s = stats != NULL ? stats->kernel_dominance_s : 0.0;
    const double kernel_total_s = stats != NULL ? stats->kernel_total_s : 0.0;
    const long long cpu_mem_peak_bytes = stats != NULL ? stats->cpu_mem_peak_bytes : 0;
    const long long gpu_mem_peak_used_bytes = stats != NULL ? stats->gpu_mem_peak_used_bytes : 0;
    const long long gpu_mem_peak_reserved_bytes =
        stats != NULL ? stats->gpu_mem_peak_reserved_bytes : 0;
    const int layer_coupling = stats != NULL ? stats->layer_coupling : 0;
    const int dominance_filtered_total = stats != NULL ? stats->dominance_filtered_total : 0;
    const long long work_candidates_total = stats != NULL ? stats->work_candidates_total : 0;
    const long long work_candidates_peak = stats != NULL ? stats->work_candidates_peak : 0;
    const long long work_frontier_survivors_total =
        stats != NULL ? stats->work_frontier_survivors_total : 0;
    const long long work_frontier_peak_points =
        stats != NULL ? stats->work_frontier_peak_points : 0;
    const long long work_join_products_total = stats != NULL ? stats->work_join_products_total : 0;
    static const vector<double> kEmptyDoubleVector;
    const vector<double> &std_candidates_per_layer =
        stats != NULL ? stats->std_candidates_per_layer : kEmptyDoubleVector;
    const vector<double> &std_frontier_survivors_per_layer =
        stats != NULL ? stats->std_frontier_survivors_per_layer : kEmptyDoubleVector;

    out << "{";
    out << "\"schema_version\":1,";
    out << "\"identity\":{";
    out << "\"input_path\":\"" << json_escape(opts.input_path) << "\",";
    out << "\"problem_type\":" << opts.problem_type << ",";
    out << "\"method\":" << opts.method << ",";
    out << "\"state_dominance\":" << opts.state_dominance << ",";
    out << "\"backend\":\"" << backend_to_string(opts.backend) << "\",";
    out << "\"cpu_threads\":" << opts.cpu_threads;
    out << "},";

    out << "\"outputs\":{";
    out << "\"num_solutions\":" << (stats != NULL ? stats->num_solutions : 0) << ",";
    out << "\"save_frontier\":" << (opts.save_frontier ? "true" : "false") << ",";
    out << "\"frontier_out_path\":\"" << json_escape(opts.frontier_out_path) << "\"";
    out << "},";

    out << "\"timing\":{";
    out << "\"cpu\":{";
    out << "\"cpu_compile_s\":" << cpu_compile_s << ",";
    out << "\"cpu_enumeration_s\":" << cpu_enumeration_s << ",";
    out << "\"cpu_total_s\":" << cpu_total_s << ",";
    out << "\"cpu_state_dominance_s\":" << cpu_state_dominance_s;
    out << "},";
    out << "\"wall\":{";
    out << "\"wall_compile_s\":" << wall_compile_s << ",";
    out << "\"wall_enumeration_s\":" << wall_enumeration_s;
    out << "},";
    out << "\"kernel\":{";
    out << "\"kernel_expand_td_s\":" << kernel_expand_td_s << ",";
    out << "\"kernel_dominance_s\":" << kernel_dominance_s << ",";
    out << "\"kernel_total_s\":" << kernel_total_s;
    out << "}";
    out << "},";

    out << "\"memory\":{";
    out << "\"cpu\":{";
    out << "\"cpu_mem_peak_bytes\":" << cpu_mem_peak_bytes;
    out << "},";
    out << "\"gpu\":{";
    out << "\"gpu_mem_peak_used_bytes\":" << gpu_mem_peak_used_bytes << ",";
    out << "\"gpu_mem_peak_reserved_bytes\":" << gpu_mem_peak_reserved_bytes;
    out << "}";
    out << "},";

    out << "\"work\":{";
    out << "\"work_candidates_total\":" << work_candidates_total << ",";
    out << "\"work_candidates_peak\":" << work_candidates_peak << ",";
    out << "\"work_frontier_survivors_total\":" << work_frontier_survivors_total << ",";
    out << "\"work_frontier_peak_points\":" << work_frontier_peak_points << ",";
    out << "\"work_join_products_total\":" << work_join_products_total;
    out << "},";

    out << "\"dominance\":{";
    out << "\"dominance_filtered_total\":" << dominance_filtered_total;
    out << "},";

    out << "\"structure\":{";
    out << "\"is_tsp_branch\":" << (opts.problem_type == 3 ? "true" : "false") << ",";
    out << "\"postprocess_sort_applied\":true,";
    out << "\"original_width\":" << record.original_width << ",";
    out << "\"reduced_width\":" << record.reduced_width << ",";
    out << "\"original_num_nodes\":" << record.original_num_nodes << ",";
    out << "\"reduced_num_nodes\":" << record.reduced_num_nodes << ",";
    out << "\"max_num_nodes_per_layer\":";
    write_json_long_array(out, record.max_num_nodes_per_layer);
    out << ",";
    out << "\"max_num_in_arcs\":" << record.max_num_in_arcs << ",";
    out << "\"total_num_in_arcs\":" << record.total_num_in_arcs << ",";
    out << "\"layer_coupling\":" << layer_coupling;
    out << "},";

    out << "\"metrics\":{";
    out << "\"wall_state_dominance_s\":" << (stats != NULL ? stats->wall_state_dominance_s : 0.0)
        << ",";
    out << "\"wall_cutset_sort_s\":" << (stats != NULL ? stats->wall_cutset_sort_s : 0.0) << ",";
    out << "\"wall_cutset_convolution_s\":"
        << (stats != NULL ? stats->wall_cutset_convolution_s : 0.0) << ",";
    out << "\"wall_cutset_partial_merge_s\":"
        << (stats != NULL ? stats->wall_cutset_partial_merge_s : 0.0) << ",";
    out << "\"wall_pack_transfer_s\":" << (stats != NULL ? stats->wall_pack_transfer_s : 0.0)
        << ",";
    out << "\"wall_join_s\":" << (stats != NULL ? stats->wall_join_s : 0.0) << ",";
    out << "\"cpu_layers_td\":" << (stats != NULL ? stats->cpu_layers_td : 0) << ",";
    out << "\"cpu_layers_bu\":" << (stats != NULL ? stats->cpu_layers_bu : 0) << ",";
    out << "\"cpu_nodes_expanded\":" << (stats != NULL ? stats->cpu_nodes_expanded : 0) << ",";
    out << "\"cpu_cutset_size\":" << (stats != NULL ? stats->cpu_cutset_size : 0) << ",";
    out << "\"std_candidates_per_layer\":";
    write_json_double_array(out, std_candidates_per_layer);
    out << ",";
    out << "\"std_frontier_survivors_per_layer\":";
    write_json_double_array(out, std_frontier_survivors_per_layer);
    out << "},";

    out << "\"status\":{";
    out << "\"status_state\":\"" << json_escape(record.status_state) << "\",";
    out << "\"status_error_message\":\"" << json_escape(record.status_error_message) << "\"";
    out << "}";
    out << "}\n";

    if (!out.good()) {
        if (error != NULL) {
            *error = "failed while writing JSONL output";
        }
        return false;
    }
    return true;
}

void print_and_save_run_summary(const CliOptions &opts, const EnumerationStats *enumeration_stats,
                                const DDStats &run_summary, const ParetoFrontier *pareto_frontier) {
    if (opts.problem_type == 3) {
        const double cpu_total_s = enumeration_stats != NULL ? enumeration_stats->cpu_total_s : 0.0;
        const double cpu_compile_s =
            enumeration_stats != NULL ? enumeration_stats->cpu_compile_s : 0.0;
        const double cpu_enumeration_s =
            enumeration_stats != NULL ? enumeration_stats->cpu_enumeration_s : 0.0;
        const double wall_compile_s =
            enumeration_stats != NULL ? enumeration_stats->wall_compile_s : 0.0;
        const double wall_enumeration_s =
            enumeration_stats != NULL ? enumeration_stats->wall_enumeration_s : 0.0;

        cout << (enumeration_stats != NULL ? enumeration_stats->num_solutions
                                           : pareto_frontier->get_num_sols())
             << endl;
        cout << cpu_total_s << endl;
        cout << cpu_compile_s;
        cout << "\t" << cpu_enumeration_s;
        cout << "\t" << wall_compile_s;
        cout << "\t" << wall_enumeration_s;
        cout << endl;
    } else {
        const int layer_coupling =
            enumeration_stats != NULL ? enumeration_stats->layer_coupling : 0;
        const int dominance_filtered_total =
            enumeration_stats != NULL ? enumeration_stats->dominance_filtered_total : 0;
        const double cpu_state_dominance_s =
            enumeration_stats != NULL ? enumeration_stats->cpu_state_dominance_s : 0.0;
        const double cpu_compile_s =
            enumeration_stats != NULL ? enumeration_stats->cpu_compile_s : 0.0;
        const double cpu_enumeration_s =
            enumeration_stats != NULL ? enumeration_stats->cpu_enumeration_s : 0.0;
        const double cpu_total_s = enumeration_stats != NULL ? enumeration_stats->cpu_total_s : 0.0;
        const double wall_compile_s =
            enumeration_stats != NULL ? enumeration_stats->wall_compile_s : 0.0;
        const double wall_enumeration_s =
            enumeration_stats != NULL ? enumeration_stats->wall_enumeration_s : 0.0;

        cout << (enumeration_stats != NULL ? enumeration_stats->num_solutions
                                           : pareto_frontier->get_num_sols())
             << endl;
        cout << cpu_total_s << endl;

        cout << opts.method;
        cout << "\t" << opts.state_dominance;
        cout << "\t" << run_summary.original_width;
        cout << "\t" << run_summary.reduced_width;
        cout << "\t" << run_summary.original_num_nodes;
        cout << "\t" << run_summary.reduced_num_nodes;
        cout << "\t" << cpu_compile_s;
        cout << "\t" << cpu_enumeration_s;
        cout << "\t" << layer_coupling;
        cout << "\t" << dominance_filtered_total;
        cout << "\t" << cpu_state_dominance_s;
        cout << "\t" << wall_compile_s;
        cout << "\t" << wall_enumeration_s;
        cout << endl;
    }

    if (opts.save_stats) {
        string stats_error;
        if (!write_stats_jsonl(opts.stats_out_path, opts, enumeration_stats, run_summary,
                               &stats_error)) {
            cerr << "Error - failed to save stats to '" << opts.stats_out_path << "'";
            if (!stats_error.empty()) {
                cerr << ": " << stats_error;
            }
            cerr << '\n';
            exit(1);
        }
    }
}
