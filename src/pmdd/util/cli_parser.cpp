// --------------------------------------------------
// CLI parser
// --------------------------------------------------

#include "cli_parser.hpp"
#include "omp_compat.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>

using namespace std;

CliOptions::CliOptions()
    : problem_type(0), method(0), state_dominance(0), backend(BACKEND_CPU), cpu_threads(1),
      gpu_batch_size(DEFAULT_GPU_BATCH_SIZE), gpu_max_prod(DEFAULT_GPU_MAX_PROD),
      save_frontier(false), save_stats(false) {}

static string derive_default_frontier_path(const string &input_path) {
    string base = input_path;
    size_t slash_pos = base.find_last_of("/\\");
    if (slash_pos != string::npos) {
        base = base.substr(slash_pos + 1);
    }

    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != string::npos) {
        base = base.substr(0, dot_pos);
    }

    if (base.empty()) {
        base = "frontier";
    }

    return base + ".frontier.csv.gz";
}

static string derive_default_stats_path(const string &input_path) {
    string base = input_path;
    size_t slash_pos = base.find_last_of("/\\");
    if (slash_pos != string::npos) {
        base = base.substr(slash_pos + 1);
    }

    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != string::npos) {
        base = base.substr(0, dot_pos);
    }

    if (base.empty()) {
        base = "stats";
    }

    return base + ".stats.jsonl";
}

static bool parse_positive_int(const string &value, int *out_value) {
    if (value.empty()) {
        return false;
    }

    errno = 0;
    char *end_ptr = NULL;
    long parsed = strtol(value.c_str(), &end_ptr, 10);
    if (errno != 0 || end_ptr == value.c_str() || *end_ptr != '\0') {
        return false;
    }
    if (parsed <= 0 || parsed > INT_MAX) {
        return false;
    }

    if (out_value != NULL) {
        *out_value = static_cast<int>(parsed);
    }
    return true;
}

static bool parse_gpu_batch_size(const string &value, long long *out_value) {
    if (value.empty()) {
        return false;
    }

    long long multiplier = 1;
    string digits = value;
    const char last = value[value.size() - 1];
    if (isalpha(static_cast<unsigned char>(last))) {
        digits = value.substr(0, value.size() - 1);
        const char suffix = static_cast<char>(toupper(static_cast<unsigned char>(last)));
        if (suffix == 'K') {
            multiplier = 1000LL;
        } else if (suffix == 'M') {
            multiplier = 1000000LL;
        } else if (suffix == 'B') {
            multiplier = 1000000000LL;
        } else {
            return false;
        }
    }

    if (digits.empty()) {
        return false;
    }
    for (size_t i = 0; i < digits.size(); ++i) {
        if (!isdigit(static_cast<unsigned char>(digits[i]))) {
            return false;
        }
    }

    errno = 0;
    char *end_ptr = NULL;
    long long parsed = strtoll(digits.c_str(), &end_ptr, 10);
    if (errno != 0 || end_ptr == digits.c_str() || *end_ptr != '\0') {
        return false;
    }
    if (parsed <= 0 || parsed > LLONG_MAX / multiplier) {
        return false;
    }

    if (out_value != NULL) {
        *out_value = parsed * multiplier;
    }
    return true;
}

const char *backend_to_string(const Backend backend) {
    return backend == BACKEND_GPU ? "gpu" : "cpu";
}

void print_usage() {
    cout << '\n';
    cout << "Usage: multiobj_nobjs<NUM_OBJS> [input file] [problem type] [method] "
            "[state_dominance] [options]\n";

    cout << "\n\twhere:";

    cout << "\n";
    cout << "\t\tproblem_type = 1: knapsack\n";
    cout << "\t\tproblem_type = 2: set packing\n";
    cout << "\t\tproblem_type = 3: TSP\n";

    cout << "\n";
    cout << "\t\tmethod = 1: top-down BFS\n";
    cout << "\t\tmethod = 2: bottom-up BFS\n";
    cout << "\t\tmethod = 3: dynamic layer cutset\n";

    cout << "\n";
    cout << "\t\tstate_dominance = 0:  disable state dominance\n";
    cout << "\t\tstate_dominance = 1:  state dominance strategy 1\n";

    cout << "\n";
    cout << "\t\tNamed backend options:\n";
    cout << "\t\t\t--backend cpu|gpu\n";
    cout << "\t\t\t--cpu-threads <N>   (cpu only)\n";
    cout << "\t\tShorthand backend options:\n";
    cout << "\t\t\tcpu [N]\n";
    cout << "\t\t\tgpu\n";
    cout << "\t\tbackend omitted defaults to cpu\n";
#if CUMODD_HAS_OPENMP
    cout << "\t\tcpu threads default to OMP_NUM_THREADS if valid, otherwise 1\n";
#else
    cout << "\t\tthis binary was built without OpenMP: CPU runs serially (1 thread)\n";
    cout << "\t\t--cpu-threads and cpu [N] are unsupported; rebuild with ENABLE_OPENMP=1\n";
#endif

    cout << "\n";
    cout << "\t\t--save-frontier: save Pareto frontier to <input_stem>.frontier.csv.gz\n";
    cout << "\t\t--frontier-out <path>: save Pareto frontier to explicit gzip CSV path\n";
    cout << "\t\t--save-stats: save one JSONL record with run statistics\n";
    cout << "\t\t--stats-out <path>: save JSONL statistics to explicit path\n";
    cout
        << "\t\t--max-cand <N>: GPU candidate batch cap; supports K, M, B suffixes (default 20M)\n";
    cout << "\t\t--max-prod <N>: GPU coupled batch product cap; supports K, M, B suffixes (default "
            "625K)\n";
    cout << "\t\toptional arguments can be provided in any order\n";

    cout << endl;
}

bool parse_cli_args(int argc, char *argv[], CliOptions *out, string *error) {
    if (out == NULL) {
        if (error != NULL) {
            *error = "internal parser error";
        }
        return false;
    }

    if (argc < 5) {
        if (error != NULL) {
            *error = "";
        }
        return false;
    }

    CliOptions opts;
    opts.input_path = argv[1];
    opts.problem_type = atoi(argv[2]);
    opts.method = atoi(argv[3]);
    opts.state_dominance = atoi(argv[4]);

    if (opts.problem_type < 1 || opts.problem_type > 3) {
        if (error != NULL) {
            *error = "Error - invalid problem type '" + string(argv[2]) +
                     "'. Valid problem types are 1 (knapsack), 2 (set packing), 3 (TSP).";
        }
        return false;
    }

    bool backend_set = false;
    bool backend_from_named = false;
    bool backend_from_shorthand = false;
    bool cpu_threads_set = false;
    bool gpu_batch_set = false;
    bool gpu_max_prod_set = false;

    for (int i = 5; i < argc; ++i) {
        string token(argv[i]);
        if (token == "--backend") {
            if (backend_from_shorthand) {
                if (error != NULL) {
                    *error = "Error - cannot mix --backend with shorthand backend token.";
                }
                return false;
            }
            if (backend_set) {
                if (error != NULL) {
                    *error = "Error - backend provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --backend requires a value (cpu or gpu).";
                }
                return false;
            }
            string backend_token(argv[++i]);
            if (backend_token == "cpu") {
                opts.backend = BACKEND_CPU;
            } else if (backend_token == "gpu") {
                opts.backend = BACKEND_GPU;
            } else if (backend_token == "cuda") {
                if (error != NULL) {
                    *error = "Error - backend token 'cuda' is unsupported; use 'gpu'.";
                }
                return false;
            } else {
                if (error != NULL) {
                    *error = "Error - invalid backend '" + backend_token + "'. Use cpu or gpu.";
                }
                return false;
            }
            backend_set = true;
            backend_from_named = true;
        } else if (token == "--cpu-threads") {
            if (cpu_threads_set) {
                if (error != NULL) {
                    *error = "Error - cpu thread count provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --cpu-threads requires a positive integer.";
                }
                return false;
            }
            string value(argv[++i]);
            if (!parse_positive_int(value, &opts.cpu_threads)) {
                if (error != NULL) {
                    *error = "Error - invalid --cpu-threads value '" + value +
                             "' (expected positive integer).";
                }
                return false;
            }
            cpu_threads_set = true;
        } else if (token == "--max-cand") {
            if (gpu_batch_set) {
                if (error != NULL) {
                    *error = "Error - GPU candidate batch cap provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --max-cand requires a positive integer, optionally using K, "
                             "M, or B suffix.";
                }
                return false;
            }
            string value(argv[++i]);
            if (!parse_gpu_batch_size(value, &opts.gpu_batch_size)) {
                if (error != NULL) {
                    *error = "Error - invalid --max-cand value '" + value +
                             "' (expected positive integer with optional K, M, or B suffix).";
                }
                return false;
            }
            gpu_batch_set = true;
        } else if (token == "--max-prod") {
            if (gpu_max_prod_set) {
                if (error != NULL) {
                    *error = "Error - GPU coupled batch product cap provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --max-prod requires a positive integer, optionally using K, "
                             "M, or B suffix.";
                }
                return false;
            }
            string value(argv[++i]);
            if (!parse_gpu_batch_size(value, &opts.gpu_max_prod)) {
                if (error != NULL) {
                    *error = "Error - invalid --max-prod value '" + value +
                             "' (expected positive integer with optional K, M, or B suffix).";
                }
                return false;
            }
            gpu_max_prod_set = true;
        } else if (token == "cpu" || token == "gpu") {
            if (backend_from_named) {
                if (error != NULL) {
                    *error = "Error - cannot mix shorthand backend token with --backend.";
                }
                return false;
            }
            if (backend_set) {
                if (error != NULL) {
                    *error = "Error - backend provided multiple times.";
                }
                return false;
            }

            opts.backend = (token == "cpu" ? BACKEND_CPU : BACKEND_GPU);
            backend_set = true;
            backend_from_shorthand = true;

            if (i + 1 < argc) {
                string next_token(argv[i + 1]);
                errno = 0;
                char *end_ptr = NULL;
                long parsed_raw = strtol(next_token.c_str(), &end_ptr, 10);
                const bool next_is_integer =
                    (errno == 0 && end_ptr != next_token.c_str() && *end_ptr == '\0');
                if (next_is_integer && token == "gpu") {
                    if (error != NULL) {
                        *error = "Error - gpu shorthand does not accept a kernel value; use 'gpu'.";
                    }
                    return false;
                }
                if (next_is_integer && token == "cpu") {
                    if (cpu_threads_set) {
                        if (error != NULL) {
                            *error = "Error - cpu threads provided multiple times.";
                        }
                        return false;
                    }
                    if (parsed_raw <= 0 || parsed_raw > INT_MAX) {
                        if (error != NULL) {
                            *error = "Error - invalid cpu shorthand thread count '" + next_token +
                                     "' (expected positive integer).";
                        }
                        return false;
                    }
                    opts.cpu_threads = static_cast<int>(parsed_raw);
                    cpu_threads_set = true;
                    ++i;
                }
            }
        } else if (token == "cuda") {
            if (error != NULL) {
                *error = "Error - backend token 'cuda' is unsupported; use 'gpu'.";
            }
            return false;
        } else if (token == "--save-frontier") {
            opts.save_frontier = true;
        } else if (token == "--frontier-out") {
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --frontier-out requires a file path.";
                }
                return false;
            }
            opts.frontier_out_path = argv[++i];
            if (opts.frontier_out_path.empty()) {
                if (error != NULL) {
                    *error = "Error - --frontier-out path cannot be empty.";
                }
                return false;
            }
            opts.save_frontier = true;
        } else if (token == "--save-stats") {
            opts.save_stats = true;
        } else if (token == "--stats-out") {
            if (i + 1 >= argc) {
                if (error != NULL) {
                    *error = "Error - --stats-out requires a file path.";
                }
                return false;
            }
            opts.stats_out_path = argv[++i];
            if (opts.stats_out_path.empty()) {
                if (error != NULL) {
                    *error = "Error - --stats-out path cannot be empty.";
                }
                return false;
            }
            opts.save_stats = true;
        } else {
            int parsed_numeric = 0;
            if (parse_positive_int(token, &parsed_numeric)) {
                if (error != NULL) {
                    *error = "Error - positional numeric argument '" + token +
                             "' must immediately follow shorthand backend token cpu|gpu.";
                }
            } else {
                if (error != NULL) {
                    *error = "Error - unrecognized optional argument '" + token + "'.";
                }
            }
            return false;
        }
    }

    if (opts.save_frontier && opts.frontier_out_path.empty()) {
        opts.frontier_out_path = derive_default_frontier_path(opts.input_path);
    }
    if (opts.save_stats && opts.stats_out_path.empty()) {
        opts.stats_out_path = derive_default_stats_path(opts.input_path);
    }

    if (opts.backend == BACKEND_GPU && cpu_threads_set) {
        if (error != NULL) {
            *error = "Error - cpu thread options are not valid with backend=gpu.";
        }
        return false;
    }
    if (opts.backend == BACKEND_CPU && gpu_batch_set) {
        if (error != NULL) {
            *error = "Error - --max-cand is only valid with backend=gpu.";
        }
        return false;
    }
    if (opts.backend == BACKEND_CPU && gpu_max_prod_set) {
        if (error != NULL) {
            *error = "Error - --max-prod is only valid with backend=gpu.";
        }
        return false;
    }
    if (opts.backend == BACKEND_CPU) {
#if CUMODD_HAS_OPENMP
        if (!cpu_threads_set) {
            const char *env_threads = getenv("OMP_NUM_THREADS");
            int parsed_env_threads = 0;
            if (env_threads != NULL &&
                parse_positive_int(string(env_threads), &parsed_env_threads)) {
                opts.cpu_threads = parsed_env_threads;
            } else {
                opts.cpu_threads = 1;
            }
        }
#else
        if (cpu_threads_set) {
            if (error != NULL) {
                *error = "Error - cpu thread options are unsupported in this build (OpenMP "
                         "disabled). Rebuild with ENABLE_OPENMP=1.";
            }
            return false;
        }
        opts.cpu_threads = 1;
#endif
    }

    if (opts.backend == BACKEND_GPU && opts.method != 1 && opts.method != 3) {
        if (error != NULL) {
            *error =
                "Error - GPU backend is unsupported for method " + to_string(opts.method) + ".";
        }
        return false;
    }

    *out = opts;
    return true;
}
