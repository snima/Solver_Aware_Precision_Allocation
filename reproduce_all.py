#!/usr/bin/env python3
"""
====================================================================================================
Master Reproducibility Runner: All Paper Tables & Figures in One Command
====================================================================================================

Paper:
  "Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion"
  Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, José R. Herrero
  (CMMSE 2026 / Journal of Computational and Applied Mathematics)

Usage:
  python3 reproduce_all.py                       # Run standard real-world traffic benchmark (~10s)
  python3 reproduce_all.py --benchmark pems      # Run Caltrans PeMS08 highway network
  python3 reproduce_all.py --benchmark climate   # Run Continental US EPA PM2.5 monitoring grid
  python3 reproduce_all.py --benchmark synthetic # Run synthetic low-rank smoke test (<1s)
  python3 reproduce_all.py --benchmark all       # Run all benchmarks sequentially
====================================================================================================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_synthetic() -> None:
    print("\n" + "=" * 80)
    print("  [1/3] EXECUTING BENCHMARK: Synthetic Matrix Completion Grid")
    print("=" * 80)
    script = PKG_DIR / "benchmarks" / "synthetic" / "run_synthetic.py"
    subprocess.run([PYTHON, str(script), "--smoke"], check=True)


def run_pems(mode: str = "24h") -> None:
    print("\n" + "=" * 80)
    print(f"  [2/3] EXECUTING BENCHMARK: Caltrans PeMS08 Highway Traffic [Mode: {mode.upper()}]")
    print("=" * 80)
    script = PKG_DIR / "benchmarks" / "pems_traffic" / "run_pems.py"
    subprocess.run([PYTHON, str(script), "--mode", mode], check=True)


def run_climate(seeds: int = 1) -> None:
    print("\n" + "=" * 80)
    print(f"  [3/3] EXECUTING BENCHMARK: Continental US EPA PM2.5 Grid (2023) [{seeds} Seed(s)]")
    print("=" * 80)
    script = PKG_DIR / "benchmarks" / "climate_epa" / "run_climate.py"
    subprocess.run([PYTHON, str(script), "--seeds", str(seeds)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Unified Reproducibility Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="pems",
        choices=["pems", "climate", "synthetic", "all"],
        help="Benchmark suite to execute (default: pems [Table 6])",
    )
    parser.add_argument(
        "--pems-mode",
        type=str,
        default="24h",
        choices=["24h", "2weeks", "full", "speed"],
        help="PeMS08 traffic duration mode (default: 24h)",
    )
    parser.add_argument(
        "--climate-seeds",
        type=int,
        default=1,
        help="Number of random seeds for climate benchmark (default: 1 for fast check)",
    )

    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.benchmark == "synthetic":
        run_synthetic()
    elif args.benchmark == "pems":
        run_pems(args.pems_mode)
    elif args.benchmark == "climate":
        run_climate(args.climate_seeds)
    elif args.benchmark == "all":
        run_synthetic()
        run_pems(args.pems_mode)
        run_climate(args.climate_seeds)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL REQUESTED BENCHMARKS FINISHED IN {total_time:.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
