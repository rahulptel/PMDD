# PMDD Agent Guide

## 0) Working Style
- Be direct, technical, and concise. This is research code, not product code.
- Make the smallest code change that solves the requested problem.
- Prefer readable C++ over clever abstractions. Match local style even when it is imperfect.
- If a request is ambiguous in a way that changes the implementation, ask before editing.
- Do not refactor adjacent code, reformat unrelated files, or add speculative features.
- If you notice unrelated dead code or cleanup opportunities, mention them instead of silently changing them.

## 1) Repository Purpose
- This is a C++ decision-diagram codebase for multiobjective optimization.
- The active parallel decision-diagram implementation lives in `src/pmdd/`.
- Benchmark baselines live in `src/baseline/`:
  - `dd/`: older decision-diagram/network-model baseline.
  - `dpa/`: defining-point algorithm baseline.
- Main PMDD executable pattern: `multiobj_nobjs<NUM_OBJS>`, for example `multiobj_nobjs3`.
- Core idea: build an exact BDD or MDD, then enumerate the Pareto frontier by dynamic-programming style frontier propagation.
- Supported PMDD problem types in `src/pmdd/main.cpp`:
  - `1`: Knapsack, represented with a BDD.
  - `2`: Set packing, converted to an independent-set BDD.
  - `3`: TSP, represented with an MDD.
- Older problem types from related branches, such as set covering, portfolio, and absolute-value models, are not wired into the PMDD executable.

## 2) Build and Environment
- PMDD build system: `src/pmdd/makefile`.
- Run PMDD builds from `src/pmdd`, or from the repo root with `make -C src/pmdd ...`.
- Host compiler: `g++` with C++11 flags.
- CUDA compiler: `nvcc` when `ENABLE_CUDA=1`; CUDA builds require detected `nvcc >= 12`.
- Boost headers are expected under `/opt/boost/include` by default.
- On `machine=cc`, `BOOSTDIR` is taken from `BOOST_ROOT`.
- Gurobi include/library settings exist in the makefile but are commented out.
- CPLEX/CP Optimizer are not linked by the PMDD executable.
- Objective dimension is compile-time:
  - `NUM_OBJS` defines macro `NOBJS`.
  - Input files may carry an objective count, but core containers and loops assume the binary was built with the matching `NOBJS`.

Common PMDD commands from the repo root:
- `make -C src/pmdd NUM_OBJS=3`
- `make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=0`
- `make -C src/pmdd NUM_OBJS=3 ENABLE_OPENMP=1`
- `make -C src/pmdd clean`
- `make -n -C src/pmdd ENABLE_CUDA=0 NUM_OBJS=3`
- `src/pmdd/compile_all`

`src/pmdd/compile_all` defaults:
- Builds `NUM_OBJS=3..7`.
- Uses `ENABLE_CUDA=1` and `ENABLE_OPENMP=1` unless overridden.
- Runs `make clean` first unless `CLEAN_FIRST=0`.
- Stores binaries in `resources/bin/pmdd/`.

Baseline commands:
- `make -C src/baseline/dd NUM_OBJS=3`
- `make -C src/baseline/dpa`
- Baseline builds require their own solver dependencies; do not assume CPLEX is installed.

## 3) Runtime Interface
Usage from the repo root:

```bash
resources/bin/pmdd/multiobj_nobjs3 <input-file> <problem-type> <method> <state_dominance> [options]
```

Problem types:
- `1`: knapsack.
- `2`: set packing.
- `3`: TSP.

Methods:
- `1`: top-down BFS frontier propagation.
- `2`: bottom-up BFS frontier propagation.
- `3`: dynamic layer cutset coupling.

Backend options:
- Default backend is CPU.
- Named form: `--backend cpu|gpu`.
- Shorthand form: `cpu [num_threads]` or `gpu`.
- `cuda` is rejected; use `gpu`.
- `--cpu-threads <N>` requires an OpenMP-enabled build.
- GPU candidate batch cap can be overridden with `--max-cand <N>`.
- `--max-cand` accepts plain positive integers or suffixes `K`, `M`, `B`
  for decimal thousands/millions/billions; default is `20M`.
- GPU coupled batch product cap can be overridden with `--max-prod <N>`.
- `--max-prod` accepts plain positive integers or suffixes `K`, `M`, `B`
  for decimal thousands/millions/billions; default is `625K`.
- `--ideal-point-prune` (GPU only, method=3 coupled path, default off): skips
  whole cutset nodes in the join once the running frontier already
  dominates-or-ties the node's ideal/utopia point (`max(td) + max(bu)`
  componentwise). Exact, not a heuristic. Net win on instances with a lot of
  cross-node dominance (e.g. set packing); small overhead (~3%) on instances
  where nothing is ever prunable (e.g. TSP), hence opt-in rather than default.

Output options:
- `--save-frontier` writes `<input_stem>.frontier.csv.gz`.
- `--frontier-out <path>` writes a gzip-compressed CSV frontier to the explicit path.
- `--save-stats` appends one JSONL stats record.
- `--stats-out <path>` implies `--save-stats`; default is `<input_stem>.stats.jsonl`.

GPU support in current PMDD dispatch:
- BDD knapsack/set-packing: GPU is implemented for methods `1` and `3`.
- BDD method `2` rejects GPU.
- TSP/MDD: GPU is implemented for methods `1` and `3`.
- TSP method `2` is not accepted in `main`.
- CPU methods support OpenMP threading when built with `ENABLE_OPENMP=1`.

### Common Enumeration Runs
Use a binary whose `NUM_OBJS` matches the input file. The examples below use
`NUM_OBJS=3`, `8` CPU threads, no state dominance, and a knapsack input.

Build an OpenMP CPU binary:

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

Run top-down enumeration on CPU with threads (`method=1`):

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend cpu --cpu-threads 8
```

Run coupled enumeration on CPU with threads (`method=3`, dynamic layer cutset):

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 3 0 --backend cpu --cpu-threads 8
```

Build a CUDA-enabled binary:

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

Run GPU-based top-down enumeration (`method=1`):

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend gpu
```

For set packing, keep the same method/backend pattern and change
`problem-type` to `2` with a set-packing input. For TSP, change `problem-type`
to `3`; TSP accepts CPU/GPU top-down with `method=1` and CPU/GPU coupled
enumeration with `method=3`.

## 4) Program Output
Stdout is always three lines.

For BDD problem types `1` and `2`:
- Line 1: number of Pareto solutions.
- Line 2: CPU total time, `cpu_compile_s + cpu_enumeration_s`.
- Line 3: tab-separated fields:
  - `method`
  - `state_dominance`
  - `original_width`
  - `reduced_width`
  - `original_num_nodes`
  - `reduced_num_nodes`
  - `cpu_compile_s`
  - `cpu_enumeration_s`
  - `layer_coupling`
  - `dominance_filtered_total`
  - `cpu_state_dominance_s`
  - `wall_compile_s`
  - `wall_enumeration_s`

For TSP problem type `3`:
- Line 1: number of Pareto solutions.
- Line 2: CPU total time, `cpu_compile_s + cpu_enumeration_s`.
- Line 3: `cpu_compile_s<TAB>cpu_enumeration_s<TAB>wall_compile_s<TAB>wall_enumeration_s`.

JSONL stats are written by `src/pmdd/util/output_utils.cpp` and include identity, output paths, timing, memory, work counters, dominance counters, structure, metrics, and status.

## 5) Codebase Map
- `src/pmdd/main.cpp`
  - CLI dispatch, instance loading, BDD/MDD construction, method/backend selection, output calls.
- `src/pmdd/util/`
  - `cli_parser.*`: Positional CLI and optional backend/output parsing.
  - `output_utils.*`: Three-line stdout, gzip frontier CSV, JSONL stats.
  - `stats.hpp`: `EnumerationStats` and `DDStats` structures.
  - `omp_compat.hpp`, `cpu_affinity.*`: OpenMP compatibility and CPU thread pinning.
  - `util.hpp/.cpp`: Common math and print helpers.
- `src/pmdd/bdd/`
  - `bdd.hpp`: BDD node/arc structure and maintenance methods.
  - `bdd_alg.hpp`: BDD reduction logic; `bdd_alg.cpp` is effectively empty.
  - `knapsack_bdd.*`, `indepset_bdd.*`: exact BDD constructors.
- `src/pmdd/mdd/`
  - `mdd.hpp`: MDD node/arc structure.
  - `tsp_mdd.*`: exact TSP MDD constructor.
- `src/pmdd/enum/`
  - `multiobj_enum.hpp/.cpp`: Central multiobjective frontier enumeration dispatch hub (`MultiobjEnum`).
  - `pareto_frontier.hpp`: Nondominated frontier container, and frontier merge/convolution logic.
  - `cpu/`: CPU-based frontier propagation:
    - `enum.cpp`: High-level CPU frontier propagation orchestrator.
    - `topdown.cpp`, `bottomup.cpp`, `couple.cpp`: Implementation of CPU top-down, bottom-up, and dynamic layer cutset algorithms.
    - `dominance.cpp`: CPU-based state dominance filters for knapsack and set packing.
    - `cpu_helpers.hpp`, `cpu_wrappers.hpp`: Threading helper functions and internal CPU wrapper routines.
  - `gpu/`: GPU-based frontier propagation (compiled when `ENABLE_CUDA=1`):
    - `enum.cu`: High-level GPU frontier propagation dispatch.
    - `topdown.cu`, `bottomup.cu`, `couple.cu`: CUDA kernels and host logic for top-down, bottom-up (stub), and coupled algorithms.
    - `dominance_utils.cuh`: CUDA state dominance helper routines.
    - `enum_types.cuh`: CUDA device-side structures.
    - `cuda_wrappers.hpp`, `cuda_stubs.cpp`: Outer wrappers for CUDA launches, plus stub routines used when CUDA is disabled.
- `src/pmdd/instances/`
  - Parsers for knapsack, set packing, independent set, and TSP.
  - `assignment_instance.*` is stubbed and not integrated in `main`.
- `src/baseline/dd/`
  - Decision-diagram/network-model baseline. It builds `multiobj`, writes binaries to `resources/bin/baseline/dd`, and has separate legacy problem support and dependencies.
- `src/baseline/dpa/`
  - Defining-point algorithm baseline. It builds `main`, writes it to `resources/bin/baseline/dpa`, and reads CPLEX `.lp` files.
- `data/`
  - Included benchmark/input data for objective dimensions `3..7`.
- `kb/`
  - Repository-local knowledge base.
  - Some notes may mention older paths such as `src/enum/*` or `src/cuda/*`; when they conflict with this checkout, use current `src/pmdd/*` paths.

## 6) Input Format Cheat Sheet
- Knapsack (`src/pmdd/instances/knapsack_instance.cpp`):
  - `n_vars n_cons num_objs`
  - `num_objs` rows of `n_vars` objective coefficients
  - For each constraint: `n_vars` coefficients followed by one RHS.
- Set packing (`src/pmdd/instances/setpacking_instance.hpp`):
  - `n_vars n_cons n_objs`
  - Objective matrix of shape `n_objs x n_vars`
  - For each constraint: `count` followed by `count` 1-based variable ids.
  - The parser converts constraint variable ids to 0-based indices.
- TSP (`src/pmdd/instances/tsp_instance.cpp`):
  - `n_objs n_cities`
  - For each objective: full `n_cities x n_cities` cost matrix.
- Independent-set DIMACS parsing exists in `IndepSetInst`/`Graph`, but `main` uses it indirectly through set packing.

## 7) Algorithm Flow
BDD path for problem types `1` and `2`:
1. Read instance.
2. Build exact BDD.
3. For knapsack, reduce the BDD, update node weights, reduce again, then update weights again.
4. Compute structure stats such as width, node counts, layer sizes, and arc counts.
5. Enumerate the Pareto frontier using method `1`, `2`, or `3`.
6. Sort the frontier lexicographically before saving/printing summaries.

TSP path for problem type `3`:
1. Read TSP instance.
2. Build exact MDD using `MDDTSPConstructor`.
3. Enumerate with top-down (`method=1`) or dynamic layer cutset (`method=3`).
4. Sort the frontier lexicographically before saving/printing summaries.

State dominance:
- `state_dominance=0` disables state dominance.
- `state_dominance=1` enables available problem-specific filters.
- CPU filters exist for knapsack and set packing.
- CUDA state dominance is implemented for knapsack in the top-down BDD path.

## 8) Important Caveats
- `NOBJS` consistency matters. Rebuild when changing objective dimension.
- There is no dedicated unit-test suite in this repo.
- `data/` contains sample benchmark files, so prefer small checked-in instances for smoke tests.
- `write_frontier_gzip_csv` shells out to `gzip`; frontier saving requires `gzip` on `PATH`.
- Memory ownership is mixed:
  - BDD arc weights are often allocated with `new ObjType[NOBJS]`.
  - MDD arc weights are owned by `MDDArc` and freed in `~MDDArc`.
- If removing arcs or nodes, preserve `prev`/arc consistency and layer indices; use existing cleanup helpers where applicable.
- Do not assume CPLEX, set-covering, portfolio, or absval code exists in the PMDD executable.

## 9) Agent Workflow
- Start by reading `src/pmdd/main.cpp`, `src/pmdd/util/cli_parser.*`, and the file directly related to the requested change.
- Run `make -n -C src/pmdd` before full PMDD builds when changing build flags or source lists.
- For CPU-only verification, use `ENABLE_CUDA=0` to avoid requiring `nvcc`.
- For GPU changes, verify the CUDA build path and the relevant runtime branch if hardware/tooling is available.
- For baseline changes, work inside `src/baseline/dd` or `src/baseline/dpa` and verify their separate dependencies before building.
- For GPU TSP autoresearch or cleanup work, read the relevant `kb/` knowledge-base files before editing, but re-check all source paths against the current tree.
- When adding a PMDD problem type:
  - Add or complete the parser in `src/pmdd/instances/`.
  - Add a BDD or MDD constructor.
  - Extend CLI validation and usage text.
  - Extend `src/pmdd/main.cpp` dispatch.
  - Decide whether state dominance and GPU support apply.
  - Update stdout/JSONL structure only deliberately, since scripts may depend on it.
