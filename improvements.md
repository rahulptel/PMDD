# GPU Coupled Enumeration (method=3): Performance Improvement Plan

Scope: the GPU bidirectional / dynamic-layer-cutset path — `enumerate_mdd_coupled` and
`coupled_bdd_cuda_enumerate` in `src/pmdd/enum/gpu/enum.cu`, the shared layer expansion
`expand_layer_frontiers` in `src/pmdd/enum/gpu/bottomup.cu`, and the cutset join
`couple_cutsets_cuda` in `src/pmdd/enum/gpu/couple.cu`. The BDD top-down twin in
`src/pmdd/enum/gpu/topdown.cu` shares almost all of these patterns, so most items apply
there too.

Items are grouped by theme and tagged **[P0]** (do first, high impact / low risk),
**[P1]** (high impact, more work), **[P2]** (worthwhile, lower or situational payoff).
All suggestions preserve exactness of the enumeration; anything that changes the
layer-selection *heuristic* (not correctness) is flagged explicitly.

---

## Status log — updated 2026-07-14

Four items have been attempted, on two branches cut from `aaai` (each in its own
worktree). The branches are **disjoint and have not been benchmarked together**.
All numbers are from the local RTX A2000 (6 GB, shared with the desktop's Xorg) —
read the measurement-protocol addendum below before trusting any delta under ~10%.

**Headline finding (2026-07-14 re-measure):** §1.1+§1.2 measured together, with a
clean same-session interleaved A/B (§3.1 reverted so it cannot contaminate), give
only **~0–3% and mostly within noise** — TSP-3 +2.6%, set-pack bp-100 +1.7%,
bp-150 −0.3%/+3.1%, TSP-5 seed13095 +0.8%. Not the "large speedup on
layer-dominated instances" §1 predicts. Root cause: none of these instances is
launch/sync-bound at the tested scale. Even the 0.10s TSP-3 spends its time in
dominance/join compute, not in the ~10 µs/launch overhead, because layer counts
(10–150) are small and per-node frontiers are large — kernel execution dominates
launch latency everywhere. **Corollary: killing the syncs does not unlock §3.1** —
that was the hypothesis and the answer is no; §3.1's own check-schedule (below) is
the lever, not sync overhead. §1.1+§1.2 are correct, free, and worth keeping as a
~1–3% cleanup, but they are not an enabler. The earlier "§1.1 alone: TSP-3
2.74s → 0.14s" figure was a **cold-JIT artifact** (no `-arch` in the build ⇒ the
first run of a freshly-built binary JIT-compiles PTX; the driver caches it after,
so only run #1 pays it). Do §8.1 (`-arch=native`) so first-run numbers stop lying,
and never trust a single un-warmed run.

### Build prerequisites (any new branch cut from `aaai`/`main`)

- `.gitignore` line 4 (`multiobj_*`, meant for the built binaries) also matches
  `src/pmdd/enum/multiobj_enum.{hpp,cpp}`, which is why commit af9c827
  ("Restructure repo") silently dropped them — **HEAD does not compile**. Both
  experiment branches carry a restore commit; cherry-pick one of them (or anchor
  the pattern to `/multiobj_*`, restore the files from `af9c827~1:src/enum/`, and
  `git add -f`) before starting anything new.
- Local build needs a Boost prefix and an nvcc-compatible host compiler
  (system gcc 13 is too new for CUDA 12.0). The makefile hardcodes `CUDAFLAGS`,
  so override it wholesale:
  `make -C src/pmdd NUM_OBJS=<N> ENABLE_CUDA=1 ENABLE_OPENMP=1 BOOSTDIR=<prefix>
  CUDAFLAGS="-std=c++14 -O3 -DUSE_CUDA -DNOBJS=<N> -I<prefix>/include -ccbin g++-11"`.

### §4.1 skip same-arc pairs — DONE, keep (branch `exp/skip-same-edge-dominance`, commit 3725c44)

Implemented exactly as described: `materialize_edge_candidates_kernel` emits a
per-candidate arc id; `mark_local_dominated_kernel` skips same-arc comparisons
(which also subsumes the self-comparison check). Exact frontier equality verified
against the CPU reference.

- TSP-5 top-down GPU (seed11093): 19.47s → 18.53s (~5%).
- TSP-5 coupled GPU (seed13095): 163.7s → 160.4s (~2% — `layer_coupling=9`;
  the wide middle layers bypass this kernel and land in the join instead).
- Set packing 150/30/7, 10 seeds, coupled GPU: net +0.8%, per-instance −17%…+8%
  (single runs — this straddle is noise-level). Verdict: real for MDD/TSP where
  a destination sees few arcs each carrying a large frontier; neutral for the
  set-packing BDD (many arcs × small per-arc frontiers, so the skip saves little
  while the arc-id loads still cost).

### §1.1 kill per-kernel sync — DONE (branch `exp/ideal-point-pruning`, commit dc579ba)

`sync_kernel()` now does only the async `cudaGetLastError()` check; the old
behavior is behind `PMDD_CUDA_SYNC_DEBUG`. The redundant explicit sync before
`thrust::reduce` in `compute_expansion_score` is also gone. Measured **together
with §1.2** — see that entry; alone it is not separable and not worth measuring
(removing the explicit sync just relocates the wait into §1.2's blocking reads).

Note still open: the BDD top-down path (`enumerate_bdd_topdown`, topdown.cu) and
the MDD/BDD coupled drivers (enum.cu) wrap each layer's expansion in a
`ScopedCudaEventTimer` whose `finish_and_add()` calls `cudaEventSynchronize` —
a per-*layer* sync (not per-kernel, so much coarser than the pathology §1.1
removed, and an acceptable granularity). It is always active because main.cpp
always passes stats. Not a bottleneck at current scale, but if a future
streams/overlap item (§5) needs the layers truly async, make these timers
opt-in.

### §1.2 eliminate single-element device_vector reads/writes — DONE (branch `exp/ideal-point-pruning`, commit 249e65a)

Converted every per-layer `exclusive_scan(counts→offsets)` + `offsets[n-1]` /
`counts[n-1]` readback + `offsets[n]=total` write into one
`inclusive_scan(counts, offsets.begin()+1)` (offsets[0] stays 0 from zero-init,
offsets[n] gets the total on-device); the total is read back at most once where
the host must branch, and not at all where it was already known host-side. Six
sites in `bottomup.cu` (`expand_layer_frontiers`, the shared MDD/BDD coupled hot
path) + five frontier-init sites in `enum.cu`; also dropped two redundant
`thrust::reduce`-to-host block-count recomputes. The two surviving
`exclusive_scan`s (`d_alive_prefix`) genuinely need exclusive semantics for the
compaction scatter. Exact frontier equality + solution counts re-verified.

**Combined §1.1+§1.2 measurement** (both binaries built from the same tree
differing *only* in these two items, §3.1 reverted out, same-session interleaved
A/B, median of 3–5 reps):

| instance | scale | A baseline | B 1.1+1.2 | Δ |
|---|---|---|---|---|
| TSP-3 seed12870 coupled | 0.10 s | 0.103 | 0.100 | +2.6% |
| set-pack bp-100 coupled | 1.4 s | 1.431 | 1.408 | +1.7% |
| set-pack bp-150-15095 coupled | 11 s | 11.380 | 11.411 | −0.3% |
| set-pack bp-150-19703 coupled | 13 s | 13.044 | 12.636 | +3.1% |
| TSP-5 seed13095 coupled | 175 s | 174.61 | 173.17 | +0.8% |

Consistent small positive, all within run-to-run noise; B's spread was tighter
on TSP-5 (A 173.9–180.1 vs B 172.9–173.5). **Keep it** (free, correct, removes
real syncs) but do not expect it to move the needle or to enable §3.1 — the
headline finding at the top explains why (nothing here is launch-bound). Verdict
matches the theory but *not* the doc's optimistic "large speedup" estimate;
update that expectation for anyone doing §5 later.

### §3.1 ideal-point pruning — DONE, check-schedule fixed (branch `exp/ideal-point-incremental-check`, cut from `exp/ideal-point-pruning`)

Implemented as: `compute_node_ub_kernel` computes `ub_v = max_td_v + max_bu_v`
per cutset node once, after the largest-first sort. Exactness verified
throughout (frontier diff on TSP-3; solution counts everywhere else, including
after the rework below).

**2026-07-14 update: fix item 1 (incremental check) implemented and it works.**
The naive version (`couple.cu:477` in commit 489726e) re-tested every remaining
ub point against the *entire* running frontier every batch iteration —
`O(remaining × frontier)` even when nothing prunes, which is where TSP-5's
~30% regression came from. Replaced with: move the check to *after*
`self_prune_points` and *before* `merge_clean_points_into_frontier` inside the
batch loop, and test remaining nodes' ub points only against **this batch's**
clean survivor set (`d_batch_points`, captured before `merge_clean_points_
into_frontier` clears it), not the accumulated frontier. This is exact by the
same transitivity argument as before (if a removed frontier point
dominated-or-tied ub, its strictly-dominating replacement — the only way a
point gets removed, see `mark_frontier_dominated_by_batch_kernel` — also
dominates-or-ties ub), plus induction: every still-remaining node has already
been checked once against every earlier batch's survivors, so checking it once
against *this* batch's survivors extends the same guarantee the full-frontier
check gave, at `O(remaining × batch_survivors)` instead of
`O(remaining × frontier)` — the latter grows unboundedly over a run, the
former does not. Item 2 (back-off) and item 3 (persistent buffers) were **not**
needed to get most of the win and were not implemented — worth revisiting only
if profiling shows the remaining check cost still matters.

Results (correctness: solution counts unchanged everywhere —
117171 / 81339 / 7916 / 171 exact-frontier):

| instance | old naive check | **new incremental check** | no-pruning baseline |
|---|---:|---:|---:|
| TSP-5 seed13095 (clean, isolated) | 218.9s (+26% vs baseline) | **179.3s (+2.7–3.5% vs baseline)** | 173.2–174.6s |
| set-pack bp-150 seed20824 (clean, isolated, cross-day — see caveat) | 113.6s | **101.5s** | 118.7–121.2s |
| set-pack bp-150-14816 (same-session, GPU under external contention — relative only) | 28.4s | **24.8s (13% faster than baseline under the same load)** | 28.5s |
| set-pack bp-150-19703 (same-session, GPU under external contention — relative only) | 26.8s | **17.2s (36% faster than baseline under the same load)** | 26.6s |

The TSP-5 overhead dropped from +26% to +3ish% (small residual cost from the
checks that still run every batch even though nothing there is ever prunable —
this is exactly what item 2's back-off would target if it turns out to matter)
while the set-packing win **grew** (10–35%+ vs the old scheme's inconsistent
0.6–9%/instance and even net-negative aggregate). Verdict: keep this. Net
result across both problem classes is now unambiguously positive, unlike the
naive version which was a wash-to-regression once launch overhead was already
addressed by §1.

Caveat on the numbers above: final validation was disrupted by ~98% external
GPU utilization from something outside this session's processes (desktop/
remote-session activity, not `multiobj_*` — `ps aux` showed nothing) — could
not get more than one clean TSP-5 repetition after that started. The clean,
isolated single-run figures (TSP-5, bp-150-20824) were captured *before* the
contention began and are trustworthy; the two "same-session" set-packing rows
were measured *during* contention but interleaved A/B/C under identical load,
so the relative comparison should still hold even though absolute times are
inflated (28s/26s here vs ~13s these same instances got in an earlier
uncontended session). Re-run for tighter confidence intervals when the GPU is
free, but the direction and rough magnitude of the win are not in doubt.

**Third problem class — MOKP knapsack (BDD, `coupled_bdd_cuda_enumerate`), 2026-07-14.**
`coupled_bdd_cuda_enumerate` (enum.cu:746) calls the same shared
`couple_cutsets_cuda`, so this fix applies to knapsack coupled runs too, not
just MDD/TSP and set-packing. Tested `data/6/knapsack/knapsack-50-6-1000-*.dat`
(50 vars, 1 constraint, 6 objectives; `layer_coupling≈28`,
`work_join_products_total≈400M` — ~8x smaller than TSP-5's 3.2B). Solution
counts match exactly (33305, CPU vs GPU vs baseline vs fixed all agree).

First (single-run) comparison suggested +14.5% (13.96s baseline → 11.93s
fixed) — but a same-session interleaved re-measurement (3 reps, same binaries,
alternating A/C) came back **11.58–11.65s for both configs, i.e. no
measurable difference**. The initial 13.96s baseline run was a noise outlier
(exactly the trap the methodology addendum exists for); trust the interleaved
number. Plausible reason this instance doesn't benefit: at only ~400M
theoretical join products (vs TSP-5's 3.2B, vs the ~1.2B on the hardest
150/7 set-packing seed), the join is a much smaller fraction of this
instance's ~12s total than the two-sided BDD layer-expansion sweeps that build
the td/bu half-frontiers in the first place — sweeps this change doesn't
touch. **Verdict: neutral on this instance size, not a regression, not a win.**
Whether MOKP benefits at all depends on finding an instance where the join
actually dominates wall time; larger `n_vars` (100+) or more objectives would
grow the cutset width and might cross that threshold — untested.

### Measurement-protocol addendum (learned the hard way)

The comparisons above mixed baselines measured on different days. On this
machine (GPU also drives the display) cross-day drift was ~±5% — larger than
several effects being measured. For every future A/B: re-measure the baseline
in the same session, interleave configs (A,B,A,B), ≥3 reps each, report the
median, and treat single-run per-instance deltas under ~10% as noise.

### Recommended order from here

§1.1, §1.2, and now §3.1's check-schedule are done. §1.1+§1.2 measured at
~0–3% (noise); §3.1 with the incremental check is a real win on set packing
(10–35%+) and near-neutral on TSP (down from the naive version's +26%
regression). What's left, in priority order:

1. **§2.1 pooled/caching allocator** — the hot loops (`couple.cu` batch loop,
   §3.1's own compaction) still `cudaMalloc`/`cudaFree` fresh `device_vector`s
   every iteration, and those are implicit syncs. Likely explains both why
   §1.1+§1.2 alone were flat (sync moved from explicit calls into the
   allocator) and why §3.1's residual ~3% TSP-5 overhead exists (the check's
   own per-batch allocations). Do this before concluding the join is
   compute-bound rather than alloc-bound.
2. §8.1 `-arch=native` — trivial, and stops first-run JIT from poisoning
   single-run measurements (see headline finding). Do it early; also re-run
   any single-rep measurement in this doc once it lands, in case JIT was
   silently inflating some of them.
3. §3.2 segmented self-prune — same construction-based argument as §4.1, applied
   to the join; independent of the above.
4. Merge `exp/skip-same-edge-dominance` (§4.1) — verified but never benchmarked
   in combination with the rest. `exp/ideal-point-incremental-check` (this
   branch) and `exp/skip-same-edge-dominance` should be combined and
   re-measured together — neither has seen the other's changes.
5. If TSP-5's residual ~3% check overhead still matters after §2.1: item 2
   from the original §3.1 fix list (back-off after consecutive no-prune
   checks) was never implemented — do it then, not before, since §2.1 may
   remove the overhead's actual cause (allocation, not the check itself).

Do NOT invest more in §1 (sync/round-trip elimination) expecting speed: it is
done, correct, and ~noise. The bottleneck is join compute + allocation, not
launch/sync latency, on every instance tested.

---

## 1. Kill per-kernel synchronization and 4-byte device round-trips

### 1.1 [P0] `sync_kernel` calls `cudaDeviceSynchronize()` after every launch

> **Status 2026-07-14: DONE** (`exp/ideal-point-pruning`, dc579ba), measured with
> §1.2 (249e65a). Combined ~0–3%, within noise — the "large speedup" claim below
> did not hold; nothing tested is launch-bound. See status log. Kept as free
> cleanup; not an enabler.

`enum_types.cuh:93` — every kernel in the pipeline is followed by
`cudaGetLastError()` + `cudaDeviceSynchronize()`. A single layer expansion issues
~6–10 kernels plus several thrust calls, each fully serialized with the host. For
instances with many layers and small/medium frontiers, launch+sync latency (5–20 µs
each) dominates over actual compute.

Fix:
- Keep `cudaGetLastError()` after launches (cheap, async), but synchronize only at
  points where the host actually consumes a device value (the readback of
  `total_candidates`, `total_next`, batch planning, final copy-out).
- Put the whole enumeration on one non-default stream (see §5 for using two).
- Optionally guard the paranoid per-kernel sync behind a debug macro
  (`PMDD_CUDA_SYNC_DEBUG`) so it stays available for debugging.

Expected: large speedup on layer-dominated instances; essentially free.

### 1.2 [P0] Eliminate single-element `device_vector` reads/writes

> **Status 2026-07-14: DONE** (`exp/ideal-point-pruning`, 249e65a). All
> `bottomup.cu` + `enum.cu` sites converted to `inclusive_scan` into
> `begin()+1`. `topdown.cu`'s BDD-top-down-only sites (642-646, 720-723) are
> NOT done — they are off the coupled path and were not measured. See status log.

Each `d_vec[i]` access on a `thrust::device_vector` is a blocking 4-byte
`cudaMemcpy`. Hot examples per layer / per batch:

- `bottomup.cu:283-289` — `d_edge_offsets[num_edges - 1]`, `d_edge_counts[num_edges - 1]`,
  then `d_edge_offsets[num_edges] = total_candidates` (2 D2H + 1 H2D per layer).
  Same pattern again inside the batched path (`bottomup.cu:355-358`) and in
  `topdown.cu:642-646, 720-723`.
- `bottomup.cu:404, 491` — `d_block_offsets[n] = reduce(...)` (a reduce that returns to
  host, then a 4-byte H2D).
- `enum.cu:209, 331, 343, 629, 641` — `d_*_offsets[n] = 1` after an exclusive scan.

Fix: use `thrust::inclusive_scan` into `offsets.begin() + 1` (offsets[0] already 0) so
the final slot is produced on device, and read back the single total once (or fold it
into the readback you already need for batch planning). Replace
`reduce` + element-store with a scan that covers `n+1` entries.

Expected: removes 5–10 hidden syncs per layer; combines with 1.1.

### 1.3 [P1] Replace the per-step `compute_expansion_score` round trip

`bottomup.cu:536-546` allocates a temp vector, launches a kernel,
`cudaDeviceSynchronize()`s, then runs a `thrust::reduce` — once per expansion step
(`enum.cu:390, 425, 698, 735`). Replace with a single fused
`thrust::transform_reduce` over a zip of (offsets, arc_counts) — no temp vector, no
explicit sync — or compute it inside the expansion epilogue where `d_next_sizes` is
already resident. Also note the score is `int`; `sizes[i] * arc_counts[i]` summed over
a wide layer can overflow — accumulate in `long long` (the CPU version has the same
hazard, but on GPU frontiers get larger before coupling).

### 1.4 [P2] CUDA Graphs for the fixed per-layer kernel chain

Once 1.1/1.2 are done, the remaining launch overhead of the count→scan→materialize→
dominate→scan→compact chain can be captured into a CUDA graph per layer shape. The
sizes are data-dependent so this needs re-instantiation per layer (or conditional
nodes); only worth it if profiling still shows launch-bound behavior on small layers.

---

## 2. Stop allocating device memory in the hot loop

### 2.1 [P0] Use a pooled/async allocator instead of fresh `device_vector`s

Every layer allocates and frees ~8 device vectors (`d_edge_counts`, `d_edge_offsets`,
`d_cand_points`, `d_alive`, `d_alive_prefix`, `d_next_*`, per-batch copies), and every
join batch allocates 6 more (`couple.cu:429-435`). `cudaMalloc`/`cudaFree` are
device-synchronizing and serialize with kernels.

Options, in increasing effort:
1. Set a `cudaMemPool` with a high release threshold and switch allocations to
   `cudaMallocAsync` (via a custom thrust allocator) — smallest code change, removes
   both the sync and the OS-level cost.
2. Thrust caching allocator (`thrust::cuda::par(alloc)` execution policy) — also fixes
   the *hidden* temp allocations inside every `thrust::reduce`/`scan`/`sort` call,
   which currently malloc/free per invocation.
3. Preallocated double-buffered arenas sized by `--max-cand`/`--max-prod` for the big
   arrays (`d_cand_points` is bounded by `max_cand * NOBJS * 4` bytes anyway).

Expected: one of the top two wins together with §1; also reduces peak fragmentation
(fewer spurious OOMs near the memory limit).

### 2.2 [P1] Reuse frontier buffers across layers

`enumerate_mdd_coupled` (`enum.cu:361-363, 396-398`) constructs fresh
`next_sizes/next_offsets/next_points` every iteration and swaps. With an arena or
ping-pong pair sized to the running maximum, the loop does zero allocations at steady
state.

### 2.3 [P2] Batch metadata: one transfer instead of five

In `couple_cutsets_cuda` each batch copies five small host vectors to device
separately (`couple.cu:429-433`). Pack them into one pinned staging buffer with a
single `cudaMemcpyAsync`; or better, compute the batch offset arrays on device
directly from `d_td_offsets`/`d_bu_offsets` + the node list (they are pure gathers).

---

## 3. Cutset join algorithmics (`couple_cutsets_cuda`) — the O(n²) hot spot

The join materializes up to `--max-prod` (625K default) product points per batch, then:
filter vs. running frontier → **whole-batch O(B²) self-prune** → filter frontier vs.
batch → concatenate. `mark_globally_dominated_kernel` (`couple.cu:99`) at B = 625K is
~4·10¹¹ point comparisons in the worst case. This phase is where large instances burn
time.

### 3.1 [P0] Ideal-point pruning: skip whole nodes and rows before materializing

> **Status 2026-07-14: node-level variant DONE, check-schedule fixed.**
> Node-level pruning first landed on `exp/ideal-point-pruning` (489726e) with a
> naive full-frontier check (−30% on TSP). The check was made incremental on
> `exp/ideal-point-incremental-check` (cut from the above) — now 10–35%+ faster
> on set packing and only ~3% overhead on TSP (down from −30%). See status log
> for the fix and results. Row-level variant (`t + max_bu_v` per td point) not
> yet attempted.

For each cutset node `v`, compute `ub_v[o] = max_td_v[o] + max_bu_v[o]` (two cheap
segmented max passes). If any running-frontier point `f` satisfies `f[o] >= ub_v[o]`
for all `o`, then *every* product of `v` is dominated-or-equal — skip the node entirely
(the merge already discards ties, `couple.cu:145`). Same test one level down: for a
fixed td point `t`, if `t + max_bu_v` is dominated-or-equal by the frontier, skip that
whole row of the product. Because the batch loop already processes the largest nodes
first (`couple.cu:381`), the running frontier is strong early, so later (numerous,
small) nodes get pruned wholesale without ever materializing their products.

Expected: often order-of-magnitude reduction in `work_join_products_total` actually
materialized; exactness preserved (only provably dominated points are skipped).

### 3.2 [P0] Structure-aware self-prune instead of whole-batch all-pairs

Two exact facts the current global kernel ignores:

- Products sharing the same td point (a "row") are a translation of the bu Pareto set,
  hence mutually nondominated. Same for a "column" (shared bu point). All
  intra-row/intra-column comparisons are wasted work.
- Pruning per node first, then merging survivor sets, does strictly fewer comparisons
  than one flat O(B²) pass over the concatenated batch: sum of per-node n² is much
  smaller than (sum n)², and cross-node comparisons then run only on survivors.

Fix: replace `self_prune_points` over the whole batch with (a) a segmented per-node
prune that skips same-row/same-column pairs, then (b) the existing merge machinery
across nodes. The segmented kernel can reuse the block-offset + binary-search
load-balancing pattern from `mark_dominated_by_dst_dynamic_1d_kernel`.

### 3.3 [P1] Keep the running frontier sorted by objective-sum for early exit

Dominance requires the dominator's component sum to be ≥ the candidate's. If the
running frontier is maintained sorted by decreasing sum (merge two sorted lists at
append time, `couple.cu:323-327`), then:

- `mark_dominated_or_equal_by_frontier_kernel` can break out of the tile loop as soon
  as the tile's max sum drops below the candidate's sum (precompute per-tile sums or
  read the first element of each tile).
- The same ordering inside the self-prune means point `i` only needs to be checked
  against earlier points — halves comparisons and shortens the tail.

Cheap to add on top of the existing shared-memory tiling; no change to results.

### 3.4 [P1] Load-balance `materialize_cutset_products_kernel`

`couple.cu:73-96` assigns one 128-thread block per node and strides over that node's
whole product. A batch consisting of one huge node (common right at the cutset) runs on
a single block — ~1% of a modern GPU. Flatten to a 1D grid over product indices with
binary search into `prod_off` (the pattern already exists in
`mark_dominated_by_dst_dynamic_1d_kernel` via `find_dst_node`), or launch
`ceil_div(products, threads)` blocks per node via a block-offset array.

### 3.5 [P2] Revisit `--max-prod` default and make batching memory-driven

625K products per batch is very conservative: 625K × NOBJS × 4B ≈ 17 MB at NOBJS=7.
Each batch costs a full merge round against the running frontier, so small batches
multiply frontier-filter passes. After 3.1/3.2 shrink per-batch survivor counts, raise
the default (e.g. derive from `cudaMemGetInfo` minus the resident packed layers), and
expose the choice in stats so experiments can correlate batch size with
`wall_join_s`.

---

## 4. Per-destination dominance kernel (`mark_local_dominated_kernel`) — the other O(n²)

Used every expansion step on both sweeps (`bottomup.cu:144`, and its twin
`mark_dominated_by_dst_dynamic_1d_kernel` in `topdown.cu:163`).

### 4.1 [P1] Skip same-edge pairs

> **Status 2026-07-14: implemented and verified** (`exp/skip-same-edge-dominance`,
> 3725c44). ~5% on TSP top-down, ~2% on TSP coupled, wash on set packing.
> Not yet merged with the `exp/ideal-point-pruning` work. See status log.

Candidates arriving over the same arc are a translated copy of the source node's
Pareto set — mutually nondominated by construction (and duplicate-free, since the
previous layer's filter removes ties). The kernel currently compares them anyway. For
the expensive case — a destination with a few arcs each carrying a large frontier — up
to 1/k of all comparisons (k = in-degree) are provably useless, and for k = 1 the
whole kernel can be skipped for that segment. Implement by passing a per-candidate
edge id (already implicit in `edge_offsets`; a binary search or a materialized id
array works) and skipping `j` in the same edge range.

### 4.2 [P1] Sum-ordering + early block exit

Same trick as 3.3, applied per segment: if candidates in a segment are processed in
decreasing objective-sum order, each point only needs comparing against the prefix,
and the tile loop gains a natural early-exit. Additionally, the current kernel keeps
`__syncthreads()`-looping over all tiles even after every thread in the block is
dominated; a `__syncthreads_and(dominated || !valid_i)` check per tile lets fully-dead
blocks retire early.

Note: sorting a segment reorders points within a destination node, which is fine —
per-node point order is not semantically meaningful downstream (offsets/sizes are per
node).

### 4.3 [P2] Fuse materialize + prune for small segments

The full candidate array is written to global memory (`d_cand_points`), read back by
the dominance kernel, then compacted — three passes of `total_candidates × NOBJS × 4`
bytes. For segments that fit in shared memory (len ≤ a few thousand points), one
kernel can materialize candidates directly into shared memory, prune there, and write
only survivors. Since survivors are typically a small fraction of candidates, this
removes most of the pipeline's global-memory traffic. Keep the current path as
fallback for oversized segments.

### 4.4 [P2] Tune block size / precompute per-point sums

`kThreadsPerBlock = 128` everywhere. With `sh[128 * NOBJS]` ints the kernels are far
from shared-memory limits (3.5 KB at NOBJS=7); try 256 on the dominance kernels.
Precomputing an `int` sum per point (stored alongside or in a parallel array) makes
the 3.3/4.2 early exits branch on one scalar instead of NOBJS loads.

---

## 5. Overlap the two sweeps

### 5.1 [P1] Run top-down and bottom-up expansions concurrently on two streams

The TD and BU passes touch disjoint data until they meet (`enum.cu:355-428`), yet run
strictly alternately today. Two options:

- **Semantics-preserving:** when the heuristic will expand the same side several times
  in a row (score stays lower), the *other* side's next expansion is known to happen
  eventually as long as `layer_td + 1 < layer_bu`; speculatively run one step of it
  concurrently on a second stream and consume the result when the heuristic actually
  chooses that side. One-step lookahead never overshoots the meeting layer if you only
  speculate while the sweeps are ≥ 2 layers apart.
- **Heuristic change (flag it):** expand both sides one layer per iteration in
  parallel until the scores cross. Simpler, changes the coupling layer slightly, still
  exact — the cutset is exact wherever the sweeps meet.

Either way this hides the cheaper sweep entirely; combined with §1 the GPU stays
busy instead of ping-ponging with the host. Note the `clock()`-based `td/bu` timers
(`enum.cu:356, 372`) measure CPU time and only work today because everything
synchronizes; switch them to the existing `ScopedCudaEventTimer` when streams land.

---

## 6. Packing and transfers

### 6.1 [P1] Pack layers into one contiguous pinned buffer, copy async

`pack_mdd_layers` (`enum.cu:85-158`) does 3–4 separate pageable H2D copies *per
layer* (`offsets`, `src`, `weights`, arc counts), each synchronous. For coupled runs it
packs both directions up front, doubling this. Build one contiguous host buffer per
array kind across all layers (sizes are known after one counting pass), stage it in
pinned memory, and issue a handful of `cudaMemcpyAsync`s. This directly shrinks
`wall_pack_transfer_s`, which is pure overhead before enumeration starts.

Bonus: keeping the host-side offsets around removes the D2H copy of
`in_edge_offsets` that the batched expansion path re-does every layer
(`bottomup.cu:312`).

### 6.2 [P2] Lazy/streamed packing for the far side

The BU sweep usually stops well before layer 0, so the eagerly-packed TD arrays for
the untouched middle layers (and vice versa) waste transfer time and GPU memory. Pack
per-layer on demand (double-buffered, prefetch next layer on a copy stream), or at
least skip the direction not needed beyond the meeting region. This also lowers
`gpu_mem_peak_reserved_bytes`, indirectly allowing bigger `--max-cand`/`--max-prod`.

---

## 7. Data layout

### 7.1 [P2] SoA (objective-major) point storage, or padded vector loads

Points are AoS: thread `i` reads `points[i*NOBJS + o]`, so a warp's loads for
objective `o` are strided by `NOBJS*4` bytes — at NOBJS=3..7 each 128-byte transaction
carries only 1/NOBJS useful data in the uncoalesced kernels (`compact`,
`materialize`, and the `point_i` loads in dominance kernels). Two options:

- Switch device-side point arrays to SoA (`points[o * n + i]`): fully coalesced loads
  in every kernel; shared-memory tiles change layout accordingly. Moderate refactor of
  every kernel indexing, but purely mechanical.
- Cheaper: pad points to 4 (NOBJS=3) or 8 (NOBJS=5..7) ints and load via `int4` — 1–2
  vector loads per point, minor memory overhead, minimal code churn.

Worth doing after §3/§4 reduce total comparisons; the dominance kernels are compute-
and shared-memory-friendly already, so measure before investing in the full SoA
refactor.

---

## 8. Build configuration

### 8.1 [P0] Add a real GPU architecture to `CUDAFLAGS`

`src/pmdd/makefile:67` has no `-arch`/`-gencode`, so nvcc 12 targets sm_52 and the
runtime JIT-compiles PTX at startup (and misses newer-arch codegen). Add
`-arch=native` for local builds (nvcc ≥ 11.6) and an overridable
`CUDA_ARCH ?= native` variable for cluster builds (`machine=cc`). Also add
`-lineinfo` so Nsight Compute maps hot instructions to source.

### 8.2 [P2] Try `-maxrregcount`/`__launch_bounds__` on the dominance kernels

The dominance kernels keep `point_i[NOBJS]` in registers plus loop state; at NOBJS=7
check achieved occupancy and, if register-limited, add
`__launch_bounds__(kThreadsPerBlock)` hints.

---

## 9. Housekeeping that indirectly helps performance work

- **[P2] Deduplicate the kernel twins.** `topdown.cu` and `bottomup.cu` contain
  near-identical copies of `count_edge_candidates_kernel`,
  `count_destination_candidates_kernel`, `materialize_edge_candidates_kernel`,
  `find_dst_node`, `compact_alive_points_kernel`, and the dominance kernel
  (`mark_dominated_by_dst_dynamic_1d_kernel` vs `mark_local_dominated_kernel` differ
  only in name). Every optimization above must currently be applied twice; fold them
  into a shared `.cuh` before starting §3/§4. Same for the BDD batched-expansion block
  in `enumerate_bdd_topdown` (`topdown.cu:671-861`), which re-implements
  `expand_layer_frontiers` inline.
- **[P2] `int` size audit.** `total_candidates`, offsets, and `alive_prefix` are
  `int`; `--max-cand` accepts values up to billions (`B` suffix) which would overflow
  offsets at > 2³¹ candidates. Either clamp `--max-cand` or move candidate indexing to
  `long long` before raising batch caps (§3.5).
- Dead field in `PackedMDDLayer`: `d_edge_src` (`enum_types.cuh:29`) is allocated per
  layer but never used — free memory and a copy per layer.

---

## Suggested order of attack

| Step | Items | Why first |
|------|-------|-----------|
| 1 | 1.1, 1.2, 8.1 | Hours of work, no algorithmic risk, benefits every later measurement |
| 2 | 2.1 (+2.2) | Removes alloc/sync noise so kernel timings become trustworthy |
| 3 | 3.1, 3.2 | The join is the asymptotic bottleneck of method=3 specifically |
| 4 | 4.1, 4.2 | Same techniques as step 3, applied to every expansion step |
| 5 | 5.1, 6.1 | Overlap and transfer wins once compute is lean |
| 6 | 3.3–3.5, 4.3, 7.1, 1.4 | Measure-driven refinements |

Measurement protocol (matches existing tooling):
- Metric: `wall_enumeration_s` plus `wall_join_s`, `kernel_expand_td_s`,
  `work_join_products_total`, `work_candidates_total`, `gpu_mem_peak_used_bytes` from
  `--save-stats` JSONL — never CPU total time.
- Correctness gate per change: identical solution count (and ideally identical sorted
  frontier CSV via `--frontier-out`) against the CPU coupled run on small instances,
  e.g. `data/3/tsp/tsp-nobj3-ncities10-seed12870.dat 3 3 0` and a knapsack instance
  for the BDD path; then the kb/GPU_NOTES.md reference instance
  (`data/5/tsp/tsp-nobj5-ncities15-seed13095.dat`, 117171 solutions).
- Profile with `nsys` first (sync/alloc gaps → §1/§2 validation), then `ncu` on
  `mark_local_dominated_kernel` and `mark_globally_dominated_kernel` (→ §3/§4).
