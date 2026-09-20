#!/usr/bin/env python3
"""
====================================================================================================
Synthetic Matrix Completion Benchmark Runner
====================================================================================================

Evaluates mixed-precision storage and influence-driven precision routing on synthetic matrices
under controlled ground-truth conditions:
    - Rank-r ground-truth X* = U V^T with prescribed condition number kappa.
    - Missing-at-random observation mask Omega (default: 70% observed).
    - Dense Gaussian background noise (default: 2% of signal standard deviation).
    - Heavy-tailed gross corruptions simulating severe outliers (default: 10% outliers at 8x scale).

Usage:
    python3 run_synthetic.py --smoke                   # Fast smoke test (<1s)
    python3 run_synthetic.py                           # Standard multi-seed evaluation
    python3 run_synthetic.py --output results/out.csv  # Save tabular CSV results
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

from src.mixed_precision_mc import (
    FormatName,
    PolicyName,
    SolverConfig,
    generate_problem,
    solve,
    storage_bytes_per_observation,
)


def parse_comma_list(value: str) -> list[str]:
    """Parses a comma-separated list of strings into stripped items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic Robust Matrix Completion Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true", help="Execute rapid smoke test (<1s)")
    parser.add_argument("--m", type=int, default=128, help="Matrix row dimension (default: 128)")
    parser.add_argument("--n", type=int, default=128, help="Matrix column dimension (default: 128)")
    parser.add_argument("--rank", type=int, default=8, help="Low-rank factor dimension r (default: 8)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--iterations", type=int, default=30, help="Outer IRLS iterations (default: 30)")
    parser.add_argument("--inner-iterations", type=int, default=3, help="Inner Riemannian steps (default: 3)")
    parser.add_argument("--observation-probability", type=float, default=0.70, help="Observation density (default: 0.70)")
    parser.add_argument("--corruption-fraction", type=float, default=0.10, help="Gross outlier fraction (default: 0.10)")
    parser.add_argument("--formats", type=parse_comma_list, default=parse_comma_list("fp8_e4m3,fp4_e2m1"),
                        help="Storage formats to evaluate (comma-separated)")
    parser.add_argument("--policies", type=parse_comma_list, default=parse_comma_list("random,quant_error,influence"),
                        help="Precision routing policies to evaluate (comma-separated)")
    parser.add_argument("--budgets", type=parse_comma_list, default=parse_comma_list("0.1,0.3"),
                        help="Protected FP32 budget fractions beta (comma-separated)")
    parser.add_argument("--output", type=str, default="", help="Path to write CSV results")

    args = parser.parse_args()

    if args.smoke:
        args.m = 32
        args.n = 32
        args.rank = 4
        args.seeds = 1
        args.iterations = 5
        args.inner_iterations = 2
        args.formats = ["fp8_e4m3", "fp4_e2m1"]
        args.policies = ["quant_error", "influence"]
        args.budgets = ["0.2"]

    print("=" * 80)
    print("  SYNTHETIC MATRIX COMPLETION BENCHMARK")
    print(f"  Dimensions: {args.m} x {args.n}, Rank: {args.rank}, Seeds: {args.seeds}")
    print(f"  Formats: {args.formats}, Policies: {args.policies}, Budgets: {args.budgets}")
    print("=" * 80)

    config = SolverConfig(
        rank=args.rank,
        outer_iterations=args.iterations,
        inner_iterations=args.inner_iterations,
    )

    budgets = [float(b) for b in args.budgets]
    records: list[dict[str, object]] = []

    header = f"{'Format':<10} | {'Policy':<12} | {'Budget':<7} | {'Mean RelErr':<12} | {'Avg Bits/Obs':<12} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))

    for fmt in args.formats:
        for pol in args.policies:
            for budget in budgets:
                errors: list[float] = []
                bits_list: list[float] = []
                times: list[float] = []

                for seed in range(args.seeds):
                    prob = generate_problem(
                        m=args.m,
                        n=args.n,
                        rank=args.rank,
                        observation_probability=args.observation_probability,
                        corruption_fraction=args.corruption_fraction,
                        seed=seed,
                    )

                    t0 = time.perf_counter()
                    res = solve(
                        prob,
                        config=config,
                        storage_format=fmt,  # type: ignore[arg-type]
                        protected_fraction=budget,
                        policy=pol,  # type: ignore[arg-type]
                        dynamic=False,
                        seed=seed,
                    )
                    elapsed = time.perf_counter() - t0

                    errors.append(res.clean_relative_error)
                    bits_list.append(res.active_bits_per_observation)
                    times.append(elapsed)

                mean_err = float(np.mean(errors))
                mean_bits = float(np.mean(bits_list))
                mean_time = float(np.mean(times))

                print(f"{fmt:<10} | {pol:<12} | {budget:<7.2f} | {mean_err:<12.5f} | {mean_bits:<12.2f} | {mean_time:<8.3f}")

                records.append({
                    "format": fmt,
                    "policy": pol,
                    "budget": budget,
                    "relative_error": mean_err,
                    "bits_per_obs": mean_bits,
                    "runtime_sec": mean_time,
                })

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
