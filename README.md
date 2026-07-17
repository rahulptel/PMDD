# Parallel Multiobjective Decision Diagrams

Parallel Multiobjective Decision Diagrams (`PMDD`) computes exact Pareto frontiers
for multiobjective optimization by combining decision-diagram representations
with parallel frontier enumeration. Supported problem classes are represented as
binary or multivalued decision diagrams, and the enumeration algorithms run on
parallel CPU and GPU backends.

The active parallel decision-diagram implementation now lives under `src/pmdd`.
Baseline implementations used for comparison live under `src/baseline`.

The main PMDD executable is compiled for a fixed number of objectives:

```bash
multiobj_nobjs<NUM_OBJS>
```

For example, a 3-objective build produces `multiobj_nobjs3`. Use a binary whose
`NUM_OBJS` matches the objective count in the input instance.

## Project Layout

```text
.
|-- data/                         # Benchmark/test instances grouped by objective count
|-- src/
|   |-- pmdd/                      # Active parallel decision-diagram implementation
|   |   |-- makefile              # PMDD build file
|   |   |-- compile_all           # Helper script for NUM_OBJS=3..7 builds
|   |   |-- main.cpp              # CLI entry point and problem dispatch
|   |   |-- bdd/                  # BDD structures and constructors
|   |   |-- mdd/                  # MDD structures and TSP constructor
|   |   |-- instances/            # Knapsack, set packing, independent set, TSP parsers
|   |   |-- enum/                 # Pareto frontier enumeration
|   |   |   |-- cpu/              # CPU top-down, bottom-up, and coupled algorithms
|   |   |   `-- gpu/              # CUDA kernels/wrappers and CPU-only stubs
|   |   `-- util/                 # CLI parsing, output, stats, OpenMP helpers
|   `-- baseline/
|       |-- dd/                   # Decision-diagram/network-model baseline
|       `-- dpa/                  # Defining-point algorithm baseline
|-- resources/bin/pmdd/            # PMDD binaries produced by make/compile_all
|-- resources/bin/baseline/dd/    # DD baseline binaries produced by make/compile_all.sh
|-- resources/bin/baseline/dpa/   # DPA baseline binary produced by make
|-- kb/                           # Repository-local notes and run context
`-- results/                      # Generated/checked-in result plots
```

The currently wired PMDD problem types are:

- `1`: multiobjective knapsack, represented as a BDD.
- `2`: multiobjective set packing, converted to an independent-set BDD.
- `3`: multiobjective TSP, represented as an MDD.

The available PMDD enumeration methods are:

- `1`: top-down BFS frontier propagation.
- `2`: bottom-up BFS frontier propagation.
- `3`: dynamic layer cutset coupling.

## PMDD Build

Builds are controlled by `src/pmdd/makefile`. Run `make` from `src/pmdd`, or use
`make -C src/pmdd` from the repository root.

Important options:

- `NUM_OBJS=<N>`: compile-time objective dimension. Default: `3`.
- `ENABLE_CUDA=0|1`: include CUDA kernels. Default: `1`.
- `ENABLE_OPENMP=0|1`: include OpenMP CPU parallelism. Default: `1`.
- `machine=cc`: use `BOOST_ROOT` as the Boost location.
- `NVCC=<path>`: override CUDA compiler auto-detection.

The makefile uses `g++` for C++ sources and expects Boost headers under
`/opt/boost/include` by default. CUDA builds require detected `nvcc >= 12`.

CPU-only build:

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

CUDA build:

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=1 ENABLE_OPENMP=1
```

If Boost is installed elsewhere:

```bash
make -C src/pmdd BOOSTDIR=/path/to/boost-prefix NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

The makefile and helper script write binaries to `resources/bin/pmdd`. The helper
script builds `multiobj_nobjs3` through `multiobj_nobjs7` by default:

```bash
src/pmdd/compile_all
```

Useful overrides:

```bash
ENABLE_CUDA=0 ENABLE_OPENMP=1 src/pmdd/compile_all
NUM_OBJS_MIN=4 NUM_OBJS_MAX=6 src/pmdd/compile_all
CLEAN_FIRST=0 MAKE_JOBS=-j8 src/pmdd/compile_all
```

On Compute Canada-style environments using `machine=cc`, set `BOOST_ROOT`:

```bash
BOOST_ROOT=/path/to/boost machine=cc ENABLE_CUDA=0 src/pmdd/compile_all
```

## PMDD Run

General CLI:

```bash
resources/bin/pmdd/multiobj_nobjs3 <input-file> <problem-type> <method> <state_dominance> [options]
```

Arguments:

- `<input-file>`: path to an instance under `data/` or another compatible file.
- `<problem-type>`: `1` knapsack, `2` set packing, `3` TSP.
- `<method>`: `1` top-down, `2` bottom-up, `3` dynamic layer cutset.
- `<state_dominance>`: `0` disabled, `1` enabled where implemented.

Backend options:

```bash
--backend cpu|gpu
cpu [num_threads]
gpu
```

CPU options:

```bash
--cpu-threads <N>
```

GPU options:

```bash
--max-cand <N>
--max-prod <N>
--ideal-point-prune
--row-prune
--col-prune
```

`--max-cand` defaults to `20M`; `--max-prod` defaults to `625K`. Both accept
plain positive integers or `K`, `M`, `B` decimal suffixes. The token `cuda` is
intentionally rejected; use `gpu`. `--ideal-point-prune` (method=3 coupled path
only, default off) skips whole cutset nodes once the running frontier already
dominates their ideal point; exact, but only a net win on instances with heavy
cross-node dominance. `--row-prune` and `--col-prune` (also method=3, gpu-only,
default off) are the finer-grained variants: they drop individual dominated
product rows (`t + max(bu)`) or columns (`max(td) + bu`) before materializing
them. All three are exact and independently composable; the row/column variants
help on instances where the single node-level ideal point is too loose to prune.

Output options:

```bash
--save-frontier
--frontier-out <path>
--save-stats
--stats-out <path>
```

`--save-frontier` writes a gzip-compressed CSV frontier to
`<input_stem>.frontier.csv.gz` unless `--frontier-out` is provided.
`--save-stats` appends one JSONL record to `<input_stem>.stats.jsonl` unless
`--stats-out` is provided. Passing `--frontier-out` or `--stats-out` implies the
corresponding save option.

### Supported Backend/Method Combinations

For knapsack and set packing BDDs:

- CPU supports methods `1`, `2`, and `3`.
- GPU supports methods `1` and `3`.
- GPU method `2` fails fast as unsupported.

For TSP MDDs:

- CPU supports methods `1` and `3`.
- GPU supports methods `1` and `3`.
- Method `2` is not accepted for TSP.

## Small PMDD Tests

The commands below use 3-objective instances included in `data/`.

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
```

CPU top-down knapsack:

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 \
  --backend cpu --cpu-threads 4 \
  --save-stats --stats-out test.cpu.stats.jsonl
```

CPU dynamic layer cutset knapsack:

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 3 0 \
  --backend cpu --cpu-threads 4 \
  --save-frontier --frontier-out test.cpu.frontier.csv.gz
```

TSP MDD:

```bash
resources/bin/pmdd/multiobj_nobjs3 data/3/tsp/tsp-nobj3-ncities5-seed495.dat 3 1 0 \
  --backend cpu --cpu-threads 4
```

Successful PMDD runs always print three lines:

1. Number of Pareto solutions.
2. CPU total time, equal to compile time plus enumeration time.
3. Tab-separated run statistics. BDD problem types include BDD structure fields;
   TSP prints compile/enumeration timing fields.

## PMDD Code Map

- `src/pmdd/main.cpp`: CLI dispatch, instance loading, BDD/MDD construction,
  method/backend selection, output calls.
- `src/pmdd/util/`: CLI parsing, gzip frontier output, JSONL stats, OpenMP
  compatibility, CPU affinity, common helpers.
- `src/pmdd/bdd/`: BDD node/arc structure, reduction logic, knapsack and
  independent-set BDD constructors.
- `src/pmdd/mdd/`: MDD node/arc structure and exact TSP MDD constructor.
- `src/pmdd/enum/`: central multiobjective frontier enumeration dispatch and
  Pareto frontier container.
- `src/pmdd/enum/cpu/`: CPU top-down, bottom-up, coupled, and state-dominance
  implementations.
- `src/pmdd/enum/gpu/`: CUDA top-down/coupled implementations, dominance helpers,
  device-side types, wrappers, and CPU-only stubs.
- `src/pmdd/instances/`: parsers for knapsack, set packing, independent set, and
  TSP. `assignment_instance.*` is stubbed and not integrated in `main`.

## Baselines

The benchmark baselines are under `src/baseline`.

- `dd/`: the older decision-diagram/network-model baseline. It builds an
  executable named `multiobj` with `NUM_OBJS` compiled in. This code depends on
  CPLEX, Concert, CP Optimizer, and Boost. Its makefile and `compile_all.sh`
  write binaries to `resources/bin/baseline/dd`.
- `dpa/`: the defining-point algorithm baseline. It builds an executable named
  `main`, writes it to `resources/bin/baseline/dpa`, and reads CPLEX `.lp`
  instances. This code depends on CPLEX and Concert.

Baseline build examples:

```bash
make -C src/baseline/dd NUM_OBJS=3
make -C src/baseline/dpa
```

See the baseline source directories for baseline-specific CLI details.

## Input Format Cheat Sheet

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

## Notes

- `NOBJS` consistency matters. Rebuild when changing objective dimension.
- There is no dedicated unit-test suite in this repo.
- `data/` contains sample benchmark files, so prefer small checked-in instances
  for smoke tests.
- `write_frontier_gzip_csv` shells out to `gzip`; frontier saving requires
  `gzip` on `PATH`.
- The `kb/` directory is a repository-local knowledge base. Some notes may
  mention older source paths such as `src/enum/*`; in this checkout, use the
  current `src/pmdd/enum/*` paths as authoritative.
