# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Primary reference: AGENTS.md

**Read `AGENTS.md` first.** It is the authoritative, detailed agent guide for this
repository — build commands, CLI/runtime interface, output formats, full codebase
map, input file formats, algorithm flow, and caveats. It is kept in sync with the
code by an automated hook (`.codex/hooks/`) that regenerates it after source
changes, so treat it as current rather than this summary.

This file only adds the high-signal orientation that isn't already in AGENTS.md.

## What this repo is

`PMDD` (Parallel Multiobjective Decision Diagrams) computes exact Pareto frontiers
for multiobjective optimization by building an exact BDD/MDD and enumerating the
frontier with parallel CPU/GPU frontier propagation. This is research code, not
product code: prefer the smallest change that solves the problem, match local
style, and don't refactor adjacent code or add speculative features.

- Active implementation: `src/pmdd/` (executable pattern `multiobj_nobjs<N>`,
  e.g. `multiobj_nobjs3`).
- Comparison baselines: `src/baseline/dd/` (older DD/network-model baseline,
  depends on CPLEX/Concert/CP Optimizer) and `src/baseline/dpa/` (defining-point
  algorithm baseline, depends on CPLEX, reads `.lp` files).
- `NOBJS` (objective count) is compile-time; rebuild when it changes.
- No unit-test suite exists — verification is done by running small instances
  under `data/` and checking the three-line stdout / JSONL stats output (format
  documented in AGENTS.md §4).

## Quick build/run (see AGENTS.md for full option reference)

```bash
make -C src/pmdd clean
make -C src/pmdd NUM_OBJS=3 ENABLE_CUDA=0 ENABLE_OPENMP=1
resources/bin/pmdd/multiobj_nobjs3 data/3/knapsack/KP_p-3_n-10_ins-1.dat 1 1 0 --backend cpu --cpu-threads 4
```

`src/pmdd/compile_all` builds `NUM_OBJS=3..7` at once (CUDA+OpenMP by default).

## Repo-specific things not in AGENTS.md

- `.clang-format`: LLVM-based style, 4-space indent, no tabs, 100-col limit —
  format touched C++ files accordingly.
- `scripts/summarize_pareto_enumeration.py`: post-processes run/stats output for
  reporting; check it before writing a new analysis script over frontier/stats
  output.
- `kb/`: repo-local knowledge base (`GPU_NOTES.md`, `MEMORY.md`,
  `autoresearch/`). Cross-check any path it mentions against the current tree —
  some notes predate the `src/pmdd/` reorganization.
- `reports/`, `outputs/`, `run/`, `obj/`: generated experiment artifacts (per
  paper/venue subfolders like `aaai`, `cp26`, `local`, `cc`). Treat as build
  output, not source, unless a task specifically asks you to edit a report.
- A Codex `Stop` hook (`.codex/hooks/`) auto-regenerates `README.md`/`AGENTS.md`
  from uncommitted `src/`/build-file changes when run under Codex. This doesn't
  fire under Claude Code, so if you make a change that affects build/CLI/output
  behavior, update `AGENTS.md`/`README.md` yourself in the same change.
