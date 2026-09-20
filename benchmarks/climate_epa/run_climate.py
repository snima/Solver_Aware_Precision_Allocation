#!/usr/bin/env python3
"""
====================================================================================================
US EPA Continental Air Quality & Climate Grid Benchmark Runner
====================================================================================================

Evaluates Influence-Routed Mixed-Precision Robust Matrix Completion on continental-scale
environmental sensing data:
    - Dataset: US Environmental Protection Agency (EPA) Air Quality System (AQS).
    - Measurement: Fine Particulate Matter (PM2.5, micrograms per cubic meter).
    - Grid Dimensions: 710 ground monitoring stations x 365 days (Year 2023).
    - Total Data Points: 259,150 spatiotemporal entries.
    - Physical Phenomenon: Captures extreme smoke transport plumes from the June 2023 Canadian
      wildfire season, providing natural heavy-tailed distributions and spatially localized spikes.

Benchmark Protocol:
    1. Low-Rank Ground Truth:
       The 710 x 365 daily matrix is normalized to zero-mean unit-variance, and its rank-10 truncated
       SVD (capturing 84.8% of nationwide variance) defines the ground-truth reference field X*.
    2. Sensing Challenges:
       - 40% missingness (p_obs = 0.60) simulating telemetry downtime and regional coverage gaps.
       - 5% gross outliers (+5.0 std) simulating sensor saturation and severe wildfire plumes.
    3. Precision & Routing Schemes:
       - FP64 reference baseline.
       - FP8 E4M3 and FP4 E2M1 storage formats.
       - Static vs Dynamic vs Column-Balanced Quantization Error allocation.

Usage:
    python3 run_climate.py              # Full 5-seed statistical campaign (~25s)
    python3 run_climate.py --seeds 1    # Fast single-seed check (~5s)
====================================================================================================
"""

from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US EPA Continental PM2.5 Grid Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random Monte Carlo seeds (1 to 5, default: 5)")
    args = parser.parse_args()

    candidates = [
        SCRIPT_DIR / "data" / "epa_pm25_processed.npz",
        REPO_ROOT / "data" / "epa_pm25_processed.npz",
    ]
    data_file = None
    for cand in candidates:
        if cand.exists():
            data_file = cand
            break

    if data_file is None:
        raise FileNotFoundError(
            f"EPA PM2.5 dataset not found. Please ensure epa_pm25_processed.npz is in "
            f"{SCRIPT_DIR / 'data'} or {REPO_ROOT / 'data'}."
        )

    with np.load(data_file) as loaded:
        m_raw = loaded["matrix"]  # 710 monitoring stations x 365 days

    m, n = m_raw.shape
    pm_mean = float(np.mean(m_raw))
    pm_std = float(np.std(m_raw))
    m_norm = (m_raw - pm_mean) / pm_std

    rank = 10
    u, s_vals, vt = np.linalg.svd(m_norm, full_matrices=False)
    captured_energy = float(np.sum(s_vals[:rank] ** 2) / np.sum(s_vals ** 2))
    condition_num = float(s_vals[0] / s_vals[rank - 1])

    print("=" * 85)
    print("  CLIMATE & ENVIRONMENTAL BENCHMARK: US EPA PM2.5 CONTINENTAL GRID (2023)")
    print(f"  Domain: {m} stations x {n} days ({m * n:,} entries) | Explaining {captured_energy * 100:.1f}% Variance")
    print("=" * 85)

    clean_matrix = (u[:, :rank] * s_vals[:rank]) @ vt[:rank, :]

    all_seeds = [42, 101, 202, 303, 404]
    seeds = all_seeds[:max(1, min(args.seeds, len(all_seeds)))]

    p_obs = 0.60
    p_corrupt = 0.05
    config = SolverConfig(rank=rank, outer_iterations=20, inner_iterations=2)

    test_cases = [
        ("fp8_e4m3", 0.10, "random",      False, False, "FP8 E4M3 + 10% Random"),
        ("fp8_e4m3", 0.10, "quant_error", False, True,  "FP8 E4M3 + 10% Quant-Error (Dynamic)"),
        ("fp8_e4m3", 0.30, "quant_error", False, True,  "FP8 E4M3 + 30% Quant-Error (Dynamic)"),
        ("fp4_e2m1", 0.10, "random",      False, False, "FP4 E2M1 + 10% Random"),
        ("fp4_e2m1", 0.10, "influence",   False, False, "FP4 E2M1 + 10% Residual-Influence"),
        ("fp4_e2m1", 0.30, "quant_error", True,  True,  "FP4 E2M1 + 30% Col-Balanced Quant-Error"),
    ]

    records: list[dict[str, object]] = []
    t_start = time.perf_counter()

    for s_idx, seed in enumerate(seeds):
        t0_seed = time.perf_counter()
        rng = np.random.default_rng(seed)
        obs_mask = rng.random(clean_matrix.shape) < p_obs
        corrupt_mask = obs_mask & (rng.random(clean_matrix.shape) < p_corrupt)

        noisy_matrix = clean_matrix.copy()
        noisy_matrix += 0.03 * rng.standard_normal(clean_matrix.shape)
        noisy_matrix[corrupt_mask] += 5.0 * rng.standard_normal(np.count_nonzero(corrupt_mask))

        rows, cols = np.nonzero(obs_mask)
        prob = Problem(
            clean=clean_matrix,
            observed=obs_mask,
            rows=rows,
            cols=cols,
            values=noisy_matrix[rows, cols].astype(np.float64),
            clean_condition_number=condition_num,
            observation_probability=p_obs,
            oversampling_ratio=rows.size / (rank * (m + n - rank)),
        )

        # Baseline FP64 solve
        res_ref = solve(prob, config, storage_format="fp64", seed=seed)
        ref_err = res_ref.clean_relative_error

        print(f"\n[Seed {seed} ({s_idx+1}/{len(seeds)})] FP64 Reference Error: {ref_err:.5f}")
        for fmt, beta, pol, dyn, bal, label in test_cases:
            res = solve(
                prob, config, storage_format=fmt,  # type: ignore[arg-type]
                protected_fraction=beta, policy=pol,  # type: ignore[arg-type]
                dynamic=dyn, column_balanced=bal, seed=seed,
            )
            penalty = res.clean_relative_error / ref_err
            savings = (1.0 - (res.active_bits_per_observation / 32.0)) * 100.0
            print(f"  {label:<40} | Err: {res.clean_relative_error:.5f} | Penalty: {penalty:.3f}x | Savings: {savings:.1f}%")

            records.append({
                "seed": seed,
                "format": fmt,
                "beta": beta,
                "policy": pol,
                "dynamic": dyn,
                "balanced": bal,
                "error": res.clean_relative_error,
                "penalty": penalty,
                "bits_per_obs": res.active_bits_per_observation,
            })

    total_time = time.perf_counter() - t_start
    print("=" * 85)
    print(f"Campaign completed in {total_time:.2f}s across {len(seeds)} seeds.")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "climate_campaign_results.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Summary CSV written to: {out_file}\n")


if __name__ == "__main__":
    main()
