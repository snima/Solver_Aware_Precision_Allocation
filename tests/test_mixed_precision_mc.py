"""
====================================================================================================
Unit Tests: Mixed-Precision Robust Matrix Completion Engine
====================================================================================================

Tests cover:
  1. Quantizer accuracy, finite representations, dynamic ranges, and monotonicity.
  2. Convergence of the Stiefel Riemannian IRLS solver on synthetic ground-truth instances.
  3. Orthogonality preservation on the Stiefel manifold: U^T U = I_r.
  4. Physical bit-rate storage accounting (including routing masks and shared block scales).
  5. Determinism across seeds and parameter sweeps.
"""

import unittest
import numpy as np

from src.mixed_precision_mc import (
    Problem,
    SolverConfig,
    active_storage_bits_per_observation,
    generate_problem,
    quantize,
    solve,
)


class QuantizerTestSuite(unittest.TestCase):
    """Verifies behavior and boundary conditions of emulated floating-point and integer formats."""

    def test_fp32_matches_numpy(self):
        """FP32 emulation must be bit-exact with IEEE single-precision float."""
        values = np.array([-1.1, 0.0, 0.3, 10.0], dtype=np.float64)
        quantized = quantize(values, "fp32")
        np.testing.assert_array_equal(quantized, values.astype(np.float32))

    def test_coarser_integer_has_no_lower_error(self):
        """Coarser bit-width formats (INT4) must produce >= quantization error than finer ones (INT8)."""
        values = np.linspace(-1.0, 1.0, 32)
        int8_err = np.linalg.norm(values - quantize(values, "int8"))
        int4_err = np.linalg.norm(values - quantize(values, "int4"))
        self.assertGreaterEqual(int4_err, int8_err)

    def test_quantizers_preserve_shape_and_finiteness(self):
        """All supported hardware formats must preserve array dimensions and return finite floats."""
        values = np.linspace(-3.0, 3.0, 70).reshape(7, 10)
        formats = ("tf32", "bf16", "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4")
        for fmt in formats:
            result = quantize(values, fmt)
            self.assertEqual(result.shape, values.shape, f"Failed for format {fmt}")
            self.assertTrue(np.all(np.isfinite(result)), f"Non-finite values in {fmt}")


class SolverTestSuite(unittest.TestCase):
    """Verifies the numerical convergence, manifold constraints, and stability of the IRLS solver."""

    def test_small_problem_runs_and_is_deterministic(self):
        """Identical problem instances and seeds must yield identical numerical results."""
        problem = generate_problem(m=24, n=20, rank=3, seed=7)
        config = SolverConfig(rank=3, outer_iterations=3, inner_iterations=1)
        res1 = solve(problem, config, "int8", 0.2, "influence", True, seed=9)
        res2 = solve(problem, config, "int8", 0.2, "influence", True, seed=9)
        self.assertTrue(np.isfinite(res1.clean_relative_error))
        self.assertEqual(res1.clean_relative_error, res2.clean_relative_error)
        self.assertEqual(res1.estimate.shape, problem.clean.shape)

    def test_reference_solver_recovers_low_rank_matrix(self):
        """Uncompressed FP64 reference solver must accurately reconstruct low-rank ground truth."""
        problem = generate_problem(m=40, n=40, rank=4, seed=2)
        config = SolverConfig(rank=4, outer_iterations=40, inner_iterations=3)
        result = solve(problem, config, seed=2)
        self.assertLess(result.clean_relative_error, 0.10, "Solver failed to reconstruct low-rank matrix")

    def test_stiefel_orthogonality_preserved(self):
        """The left factor U must satisfy U^T U = I_r within strict numerical tolerances."""
        problem = generate_problem(m=32, n=32, rank=4, seed=42)
        config = SolverConfig(rank=4, outer_iterations=10, inner_iterations=3)
        result = solve(problem, config, seed=42)
        eye = np.eye(config.rank)
        gram = result.u.T @ result.u
        np.testing.assert_allclose(gram, eye, atol=1e-12, err_msg="U is not strictly orthonormal on St(m, r)")

    def test_conditioning_and_oversampling_are_controlled(self):
        """Problem generator must accurately instantiate prescribed condition numbers and oversampling."""
        problem = generate_problem(
            m=80, n=40, rank=4, condition_number=1e3,
            oversampling_ratio=3.0, seed=3,
        )
        self.assertAlmostEqual(problem.clean_condition_number, 1e3, places=7)
        self.assertAlmostEqual(problem.oversampling_ratio, 3.0, delta=0.15)

    def test_lazy_block_routing_records_costs(self):
        """Dynamic block-level routing must accurately track churn and active storage overheads."""
        problem = generate_problem(m=24, n=20, rank=3, seed=4)
        config = SolverConfig(rank=3, outer_iterations=5, inner_iterations=1)
        result = solve(
            problem, config, "fp8_e4m3", 0.3, "influence",
            reroute_every=2, routing_granularity=16, seed=4,
        )
        self.assertEqual(result.reroute_count, 3)
        self.assertLessEqual(result.protected_fraction, 0.3)
        self.assertGreater(result.active_bits_per_observation, 8.0)
        self.assertGreaterEqual(result.mean_mask_churn, 0.0)


class StorageAccountingTestSuite(unittest.TestCase):
    """Verifies that physical memory models correctly tally bit-rates with masks and scales."""

    def test_active_bits_include_mask_and_scale(self):
        """FP8 block-scaled format with block_size=32 and scale_bits=16 must account for overhead."""
        protected = np.zeros(64, dtype=bool)
        # 64 entries: 64 * 8 (payload) + 64 (mask) + 2 blocks * 16 (scale) = 512 + 64 + 32 = 608 bits -> 9.5 bits/obs
        self.assertEqual(
            active_storage_bits_per_observation("fp8_e4m3", protected, 32),
            9.5,
        )
        # If 100% protected in FP32: 64 * 32 (payload) + 64 (mask) = 2048 + 64 = 2112 bits -> 33.0 bits/obs
        protected.fill(True)
        self.assertEqual(
            active_storage_bits_per_observation("fp8_e4m3", protected, 32),
            33.0,
        )


if __name__ == "__main__":
    unittest.main()
