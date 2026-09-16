# Solver-Aware Precision Allocation

Repository for the project:

**Solver-Aware Precision Allocation Among Stored Observations in Robust Matrix Completion**

## Authors

- Nima Sahraneshinsamani (Universitat Jaume I)
- José I. Aliaga (Universitat Jaume I)
- Sandra Catalán (Universitat Jaume I)
- José R. Herrero (Universitat Politècnica de Catalunya)

## Abstract

In robust matrix completion at scale, mixed precision plays two distinct roles: the format used to store observed entries, and the arithmetic precision of the solver. We decouple these two effects for an iteratively reweighted least-squares (IRLS) solver on the Stiefel manifold. A small fraction of observations is protected in FP32, while the bulk is compressed into low-precision formats. To isolate the structural impact of storage precision, sensitivity-based allocation scores and all solver arithmetic are evaluated in FP64.

We analyze how storage errors perturb the frozen-weight surrogate gradient with respect to U and derive three solver-aware allocation scores based on gradient sensitivity. On eight random test problems, quantization-error-aware allocation is the most effective static choice for FP8 and INT8, while a residual-based score is more robust when coarse FP4 storage is reallocated dynamically. With 16.70 active bits per entry (accounting for value payload, a one-bit allocation mask, and shared-scale metadata), the static error-aware penalty is 1.080 for FP8 and 1.138 for INT8, falling to 1.003 and 1.007 at 26.30 bits.

For FP8 and INT8, updating the protected mask every 5 to 10 iterations retains the dynamic accuracy benefit while costing less than one percent of solver time. In a semi-synthetic corruption benchmark on the real-world Caltrans PeMS traffic sensor network (170×4032), protecting 30 percent of entries in FP8 E4M3 matches the FP64 reference to the reported three-decimal precision (penalty 1.000), while two-phase column-balanced allocation under 4-bit storage (FP4 E2M1) limits relative degradation to 1.093 while storing 70 percent of entries in 4-bit format (56.6 percent active storage reduction relative to FP32, or 78.3 percent relative to FP64).

Arithmetic with FP8 or lower precision is much less forgiving in our hybrid pipeline and is therefore not used for the allocation study; static allocation provides a single-copy archivable representation, while dynamic allocation represents a compact active tier.

## Status

More project details (code, experiments, and supplementary material) will be added after paper publication.

## Project Page

A GitHub Pages site is available at:

<https://snima.github.io/Solver_Aware_Precision_Allocation/>
