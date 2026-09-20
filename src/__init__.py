"""
Mixed-Precision Robust Matrix Completion on the Stiefel Manifold.

Core numerical solver, quantizers, and Riemannian optimization algorithms.
"""

from .mixed_precision_mc import (
    Problem,
    SolverConfig,
    SolveResult,
    generate_problem,
    quantize,
    solve,
    active_storage_bits_per_observation,
    storage_bytes_per_observation,
)

__all__ = [
    "Problem",
    "SolverConfig",
    "SolveResult",
    "generate_problem",
    "quantize",
    "solve",
    "active_storage_bits_per_observation",
    "storage_bytes_per_observation",
]
