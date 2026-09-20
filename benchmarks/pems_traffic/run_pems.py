#!/usr/bin/env python3
"""
====================================================================================================
Caltrans PeMS08 Highway Traffic Benchmark Runner
====================================================================================================

Reproduces Table 6, Section 5.2, and extended multi-week/full-scale evaluations from the paper:
"Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion"
(Preprint / Under review).

Physical Sensing Context:
  - Source: Caltrans Performance Measurement System (PeMS), District 8 (San Bernardino, CA).
  - Network: 170 loop detector stations across major freeway corridors (I-15, I-10, SR-60, SR-91).
  - Sampling: 5-minute aggregation intervals (288 timestamps per 24 hours).
  - Channels: Traffic Flow (vehicles / 5 min), Occupancy (%), and Speed (mph).

Controlled Benchmark Protocol:
  1. Low-Rank Ground-Truth Reference:
     To evaluate mathematical recovery error ||X_hat - X*||_F / ||X*||_F with exact precision,
     we extract the empirical spatial-temporal traffic structure via a rank-10 truncated SVD of the
     complete normalized sensor recording, yielding reference X*.
  2. Sensing Deficiencies:
     - 30% Missing Observations: Removed uniformly at random (p_obs = 0.70).
     - 5% Outliers: Corrupted with heavy-tailed gross noise (+5.5 std) simulating detector loop faults,
       transient power surges, and communication packet drops.
  3. Precision & Routing Schemes:
     - Uncompressed FP64 reference baseline.
     - 8-bit FP8 E4M3 and 4-bit FP4 E2M1 storage formats.
     - Static & Dynamic routing with Column-Balanced fairness quotas to prevent station starvation.

Usage:
  python3 run_pems.py                  # Standard 24h traffic flow (Table 6, ~8s)
  python3 run_pems.py --mode 2weeks    # 2-week continuous flow (170 x 4032, ~20s)
  python3 run_pems.py --mode full      # Massive 62-day network (3.03M entries, ~35s)
  python3 run_pems.py --mode speed     # Non-linear speed shockwaves (170 x 4032, ~20s)
====================================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

# Robust import of package-internal modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.mixed_precision_mc import Problem, SolverConfig, solve


def load_pems_data(mode: str = "24h") -> tuple[np.ndarray, str]:
    """Loads and slices the Caltrans PeMS08 dataset according to the requested evaluation mode."""
    candidates = [
        SCRIPT_DIR / "data" / "PEMS08.npz",
        REPO_ROOT / "data" / "PEMS08.npz",
    ]
    npz_path = None
    for cand in candidates:
        if cand.exists():
            npz_path = cand
            break

    if npz_path is None:
        raise FileNotFoundError(
            f"PeMS08 dataset not found. Please ensure PEMS08.npz is placed in "
            f"{SCRIPT_DIR / 'data'} or {REPO_ROOT / 'data'}."
        )

    with np.load(npz_path) as loaded:
        raw_data = loaded["data"]  # (17856, 170, 3): [time, detectors, features]

    flow = raw_data[:, :, 0].T  # (170 detectors, 17856 timestamps)
    speed = raw_data[:, :, 2].T

    if mode == "24h":
        desc = "PeMS08 Traffic Flow (24 Hours: 170 detectors x 288 timestamps) [Table 6]"
        matrix = flow[:, :288]
    elif mode == "2weeks":
        desc = "PeMS08 Traffic Flow (2 Weeks: 170 detectors x 4,032 timestamps)"
        matrix = flow[:, :4032]
    elif mode == "full":
        desc = "PeMS08 Full Network (62 Days: 170 detectors x 17,856 timestamps = 3,035,520 entries)"
        matrix = flow
    elif mode == "speed":
        desc = "PeMS08 Traffic Speed Shockwaves (2 Weeks: 170 detectors x 4,032 timestamps)"
        matrix = speed[:, :4032]
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from '24h', '2weeks', 'full', 'speed'.")

    return matrix.astype(np.float64), desc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caltrans PeMS08 Real-World Highway Traffic Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, default="24h", choices=["24h", "2weeks", "full", "speed"],
                        help="Benchmark evaluation horizon (default: 24h, reproducing Table 6)")
    parser.add_argument("--rank", type=int, default=10, help="Low-rank factor dimension r (default: 10)")
    parser.add_argument("--outer-iter", type=int, default=40, help="Outer IRLS iterations (default: 40)")
    parser.add_argument("--inner-iter", type=int, default=3, help="Inner Riemannian steps (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for missingness and outliers (default: 42)")

    args = parser.parse_args()

    matrix_raw, description = load_pems_data(args.mode)
    m, n = matrix_raw.shape

    print("=" * 90)
    print(f"  CALTRANS PeMS08 BENCHMARK: {description}")
    print(f"  Dimensions: {m} detectors x {n} timesteps ({m * n:,} entries) | Rank: {args.rank}")
    print("=" * 90)

    # 1. Normalize data and form low-rank ground-truth X* via SVD
    mean_val = float(np.mean(matrix_raw))
    std_val = float(np.std(matrix_raw))
    norm_matrix = (matrix_raw - mean_val) / std_val

    u_full, s_full, vt_full = np.linalg.svd(norm_matrix, full_matrices=False)
    clean_norm = (u_full[:, :args.rank] * s_full[:args.rank]) @ vt_full[:args.rank, :]
    cond = float(s_full[0] / s_full[args.rank - 1])

    # 2. Synthesize sensing challenges: 30% missing entries + 5% gross outliers
    rng = np.random.default_rng(args.seed)
    p_obs = 0.70
    p_corrupt = 0.05
    obs_mask = rng.random((m, n)) < p_obs
    corrupt_mask = obs_mask & (rng.random((m, n)) < p_corrupt)

    noisy_norm = clean_norm.copy()
    noisy_norm += 0.02 * rng.standard_normal((m, n))  # 2% background Gaussian noise
    noisy_norm[corrupt_mask] += 5.5 * rng.standard_normal(np.count_nonzero(corrupt_mask))

    rows, cols = np.nonzero(obs_mask)
    prob = Problem(
        clean=clean_norm,
        observed=obs_mask,
        rows=rows,
        cols=cols,
        values=noisy_norm[rows, cols].astype(np.float64),
        clean_condition_number=cond,
        observation_probability=float(np.mean(obs_mask)),
        oversampling_ratio=rows.size / (args.rank * (m + n - args.rank)),
    )

    config = SolverConfig(
        rank=args.rank,
        outer_iterations=args.outer_iter,
        inner_iterations=args.inner_iter,
    )

    # 3. Reference Solver Run (FP64 baseline)
    print("\n[1/3] Solving FP64 uncompressed baseline reference...")
    t0 = time.perf_counter()
    ref_res = solve(prob, config=config, storage_format="fp64", protected_fraction=0.0, seed=args.seed)
    ref_time = time.perf_counter() - t0
    ref_err = ref_res.clean_relative_error
    print(f"      Reference FP64 Relative Error: {ref_err:.5f} (Time: {ref_time:.2f}s)")

    # 4. Experimental Matrix Configuration (Table 6 Configurations)
    experiments = [
        # (Format, Beta, Policy, Balanced, Description)
        ("fp64",     0.0,  "random",      False, "Uncompressed Double (FP64 Baseline)"),
        ("fp8_e4m3", 0.0,  "random",      False, "Uniform FP8 E4M3 (Zero FP32 Protection)"),
        ("fp8_e4m3", 0.10, "random",      False, "FP8 E4M3 + 10% Random Protection"),
        ("fp8_e4m3", 0.10, "quant_error", False, "FP8 E4M3 + 10% Quant-Error Protection"),
        ("fp8_e4m3", 0.30, "quant_error", False, "FP8 E4M3 + 30% Quant-Error Protection"),
        ("fp4_e2m1", 0.0,  "random",      False, "Uniform FP4 E2M1 (Zero FP32 Protection)"),
        ("fp4_e2m1", 0.30, "random",      False, "FP4 E2M1 + 30% Random Protection"),
        ("fp4_e2m1", 0.30, "influence",   False, "FP4 E2M1 + 30% Residual-Influence Protection"),
        ("fp4_e2m1", 0.30, "quant_error", True,  "FP4 E2M1 + 30% Column-Balanced Quant-Error"),
    ]

    print("\n[2/3] Evaluating Mixed-Precision Storage & Allocation Rules...")
    header = f"{'Storage Scheme':<46} | {'Bits/Obs':<9} | {'RelErr':<9} | {'Penalty':<8} | {'Savings':<8} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))

    for fmt, beta, pol, balanced, label in experiments:
        t_start = time.perf_counter()
        res = solve(
            prob,
            config=config,
            storage_format=fmt,  # type: ignore[arg-type]
            protected_fraction=beta,
            policy=pol,  # type: ignore[arg-type]
            column_balanced=balanced,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - t_start

        penalty = res.clean_relative_error / ref_err
        # Storage savings relative to FP32 (32 bits)
        savings = (1.0 - (res.active_bits_per_observation / 32.0)) * 100.0

        print(
            f"{label:<46} | "
            f"{res.active_bits_per_observation:<9.2f} | "
            f"{res.clean_relative_error:<9.5f} | "
            f"{penalty:<8.3f} | "
            f"{savings:<7.1f}% | "
            f"{elapsed:<8.2f}"
        )

    print("\n[3/3] Benchmark Complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
