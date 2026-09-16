# Solver-Aware Precision Allocation Among Stored Observations in Robust Matrix Completion

## Title
Solver-Aware Precision Allocation Among Stored Observations in Robust Matrix Completion

## Authors
Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, José R. Herrero

## Abstract
In robust matrix completion at scale, mixed precision plays two distinct roles: the format used to store observed entries, and the arithmetic precision of the solver. We decouple these two effects for an iteratively reweighted least-squares (IRLS) solver on the Stiefel manifold. A small fraction of observations is protected in FP32, while the bulk is compressed into low-precision formats. To isolate the structural impact of storage precision, sensitivity-based allocation scores and all solver arithmetic are evaluated in FP64.

We analyze how storage errors perturb the frozen-weight surrogate gradient with respect to U and derive three solver-aware allocation scores based on gradient sensitivity. On eight random test problems, quantization-error-aware allocation is the most effective static choice for FP8 and INT8, while a residual-based score is more robust when coarse FP4 storage is reallocated dynamically. With 16.70 active bits per entry (accounting for value payload, a one-bit allocation mask, and shared-scale metadata), the static error-aware penalty is 1.080 for FP8 and 1.138 for INT8, falling to 1.003 and 1.007 at 26.30 bits. For FP8 and INT8, updating the protected mask every 5 to 10 iterations retains the dynamic accuracy benefit while costing less than one percent of solver time. In a semi-synthetic corruption benchmark on the real-world Caltrans PeMS traffic sensor network (170×4032), protecting 30 percent of entries in FP8 E4M3 matches the FP64 reference to the reported three-decimal precision (penalty 1.000), while two-phase column-balanced allocation under 4-bit storage (FP4 E2M1) limits relative degradation to 1.093 while storing 70 percent of entries in 4-bit format (56.6 percent active storage reduction relative to FP32, or 78.3 percent relative to FP64). Arithmetic with FP8 or lower precision is much less forgiving in our hybrid pipeline and is therefore not used for the allocation study; static allocation provides a single-copy archivable representation, while dynamic allocation represents a compact active tier.

## Overview
This repository contains code and documentation related to solver-aware precision allocation for robust matrix completion. The goal is to assign storage precision across observed entries in a way that minimizes final-solution degradation while greatly reducing storage cost. The work decomposes storage precision (how observed entries are archived) from arithmetic precision used by the solver and studies both static and dynamic allocation strategies guided by sensitivity measures.

## Key contributions
- Decoupling of storage-format precision from solver arithmetic precision in robust matrix completion and measurement of each effect in isolation.
- Three solver-aware allocation scores derived from gradient sensitivity for the IRLS solver on the Stiefel manifold.
- Empirical results showing where static allocation (quantization-error-aware) and dynamic allocation (residual-based mask updates) are most effective across FP4 / FP8 / INT8 regimes.
- A hybrid pipeline demonstrating how a small protected subset in FP32 plus a low-precision bulk can match high-precision references with large storage reductions.

## Results highlights
- Static error-aware penalty: 1.080 (FP8) and 1.138 (INT8) at 16.70 active bits/entry; drops near 1.00 at 26.30 bits.
- Dynamic protected-mask updates every 5–10 iterations retain accuracy while costing <1% solver time.
- On a Caltrans PeMS benchmark, protecting 30% in FP8 E4M3 matches FP64 to three-decimal precision; FP4 two-phase allocation stores 70% in 4-bit with penalty 1.093.

## Getting started
- Add solver code or link to experiments in the repo (placeholder).
- Example: implement a small script to compute gradient sensitivities for a toy matrix completion instance, then derive static allocation scores.
- The docs/ page includes an interactive conceptual visualization that can be adapted to display real sensitivity numbers.

## Citation
If you use this code or ideas from this repository, please cite:
N. Sahraneshinsamani, J. I. Aliaga, S. Catalán, J. R. Herrero. "Solver-Aware Precision Allocation Among Stored Observations in Robust Matrix Completion." (Add full publication reference if available.)

## License
Apache-2.0
