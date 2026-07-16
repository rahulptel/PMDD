# Kernel Selection Cleanup Memory

## Current Main Cleanup

- Branch `legacy-v1` preserves the former user-selectable kernel API.
- Current `main` removed `--kernel`, `--cpu-kernel`, and shorthand `gpu 3`.
- CPU enumeration now uses the original CPU expansion implementation directly; the CPU kernel-3 implementation was removed.
- GPU enumeration keeps the active dynamic CUDA implementation directly, without a user-visible kernel selector.

# CUDA Variant Cleanup Memory

This repository now keeps only CUDA kernel variant v3 in the active GPU implementation.

## Retired CUDA v1

The v1 CUDA implementation used one thread block per destination node during per-node dominance filtering.

- Top-down BDD/MDD path: `mark_dominated_by_dst_single_block_kernel` in `src/cuda/topdown_cuda.cu`.
- Coupled MDD path: `mark_dominated_by_dst_single_block_kernel` in `src/cuda/coupled_cuda.cu`.
- Behavior: each block owned one destination node and each thread checked candidates for that node with a strided loop over the node-local frontier.
- Limitation: large destination-node frontiers could underutilize the GPU because a node was limited to one block.

## Retired CUDA v2

The v2 CUDA implementation used a fixed dense two-dimensional grid during per-node dominance filtering.

- Top-down BDD/MDD path: `mark_dominated_by_dst_tiled_kernel` in `src/cuda/topdown_cuda.cu`.
- Coupled MDD path: `mark_dominated_by_dst_tiled_kernel` in `src/cuda/coupled_cuda.cu`.
- Behavior: the grid dimensions were destination nodes by fixed candidate tiles, using `ceil(max_segment / kThreadsPerBlock)` tiles for every node.
- Limitation: nodes with small frontiers still received the dense tile allocation implied by the largest node frontier.

## Kept CUDA v3

The active CUDA implementation uses dynamic one-dimensional block scheduling.

- Top-down BDD/MDD path: `mark_dominated_by_dst_dynamic_1d_kernel` in `src/cuda/topdown_cuda.cu`.
- Coupled MDD path: `mark_dominated_1d_kernel` in `src/cuda/coupled_cuda.cu`.
- Supporting pieces: `compute_dst_candidate_counts_kernel` computes per-destination candidate counts and block counts; `find_dst_node` maps a 1D block index back to a destination node using the block-offset prefix sums.
- Behavior: only the blocks required by each destination node are launched, and each block handles one tile of that node's local candidate frontier.

## Cleanup Performed

- Removed user-selectable CUDA variants v1 and v2 from the active source implementation.
- Removed the `kernel_version` argument from internal CUDA wrapper APIs; GPU calls now use v3 directly.
- Removed the public `--kernel 3` and shorthand `gpu 3` compatibility forms in the later kernel-selection cleanup.
- Updated GPU experiment generation so new `cc/PMDD-gpu/table.dat` entries use `--backend gpu` without `--kernel 3`.
- CPU `--cpu-kernel 1|3` survived this earlier CUDA cleanup, then was removed in the later kernel-selection cleanup.
