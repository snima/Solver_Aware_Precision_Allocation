"""
====================================================================================================
Influence-Routed Mixed-Precision Robust Matrix Completion on the Stiefel Manifold
====================================================================================================

Authors:
    Nima Sahraneshinsamani (sahrans@uji.es) - Universitat Jaume I, Spain
    José I. Aliaga         (aliaga@uji.es)  - Universitat Jaume I, Spain
    Sandra Catalán         (catalans@uji.es)- Universitat Jaume I, Spain
    José R. Herrero        (josepr@ac.upc.edu) - Universitat Politècnica de Catalunya, Spain

Paper:
    "Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion"
    (Preprint / Under review).

----------------------------------------------------------------------------------------------------
Theoretical Summary & Intuition:
----------------------------------------------------------------------------------------------------
In large-scale matrix recovery, physical memory bandwidth is often the primary system bottleneck.
Modern hardware accelerators (such as NVIDIA Blackwell and AMD Instinct) feature native 8-bit (FP8)
and 4-bit (FP4) formats. However, naively truncating all continuous sensor observations to 4-bit
causes catastrophic error explosions (up to 60x error increase on sensor grids).

This library decouples two critical layers:
  1. Observation Storage Precision (Compressed into FP8/FP4/INT8 with a small FP32 protected pool).
  2. Solver Arithmetic Precision (Kept in standard FP64 or FP32).

Mathematical Formulation:
    min_{U in St(m, r), V in R^{n x r}}  sum_{(i,j) in Omega} rho( (U V^T)_{ij} - Y_{ij} ) + (lambda / 2) ||V||_F^2

Key Properties:
  - Stiefel Manifold St(m, r): We constrain U^T U = I_r. This eliminates the scale gauge ambiguity
    X = (U S)(V S^{-T})^T, strictly prevents factor magnitude drift across iterations, and ensures
    bounded row leverage scores ||u_i||_2 <= 1.
  - Cauchy Redescending M-Estimation: rho(z) = (c^2 / 2) * ln(1 + z^2 / c^2). The influence function
    psi(z) = z / (1 + z^2 / c^2) approaches zero as |z| -> inf. Outliers receive weight w -> 0,
    preventing gross sensor failures from wasting the scarce FP32 precision budget.
  - Solver-Aware Precision Routing:
      * Quantization-Error-Aware: Minimizes the first-order bound on Riemannian gradient perturbation;
        dominant for static storage across all formats.
      * Residual-Influence-Aware: Prioritizes entries with large gradients; provides superior stability
        for dynamic reallocation under ultra-coarse 4-bit formats.
      * Column-Balanced Quotas: Prevents column rank starvation in physical spatial sensor networks.
====================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np


# ==================================================================================================
# 1. Type Definitions, Data Classes & Hardware Format Specifications
# ==================================================================================================

FormatName = Literal[
    "fp64", "fp32", "tf32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2",
    "fp6_e3m2", "fp6_e2m3", "fp4_e2m1", "int8", "int4",
]

PolicyName = Literal["influence", "data", "quant_error", "weight", "random"]


@dataclass(frozen=True)
class Problem:
    """Represents a partially observed matrix completion problem instance.
    
    Attributes:
        clean: Ground-truth complete low-rank matrix X* in R^{m x n}.
        observed: Boolean mask in {0, 1}^{m x n} indicating observed entry locations (Omega).
        rows: 1D array of row indices for observed entries.
        cols: 1D array of column indices for observed entries.
        values: 1D array of observed (and potentially corrupted) entry values Y_Omega.
        clean_condition_number: Condition number kappa = sigma_1 / sigma_r of the ground-truth matrix.
        observation_probability: Empirical observation density |Omega| / (m * n).
        oversampling_ratio: Degrees-of-freedom oversampling ratio |Omega| / (r * (m + n - r)).
    """
    clean: np.ndarray
    observed: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    clean_condition_number: float
    observation_probability: float
    oversampling_ratio: float


@dataclass(frozen=True)
class SolverConfig:
    """Hyperparameters and numerical tolerances for the Riemannian IRLS solver.
    
    Attributes:
        rank: Target low-rank dimension r.
        outer_iterations: Number of outer IRLS iterations (updating weights, V, and routing).
        inner_iterations: Number of inner Riemannian gradient steps on U per outer iteration.
        cauchy_c: Tuning constant for Cauchy M-estimator (c = 2.385 yields 95% efficiency for Gaussian).
        anneal_iterations: Number of iterations over which Cauchy scale c is annealed from 3*c to c.
        scale_floor: Lower numerical bound on the estimated residual dispersion sigma (MAD).
        ridge: Tikhonov regularization parameter lambda for the V least-squares update.
        randomized_svd_threshold: Matrix dimension threshold above which randomized SVD is used.
        randomized_svd_oversampling: Extra dimensions sampled during randomized range finding.
        randomized_svd_power_iterations: Number of power iterations for subspace quality in SVD.
    """
    rank: int = 8
    outer_iterations: int = 60
    inner_iterations: int = 5
    cauchy_c: float = 2.385
    anneal_iterations: int = 40
    scale_floor: float = 1e-10
    ridge: float = 1e-8
    randomized_svd_threshold: int = 256
    randomized_svd_oversampling: int = 8
    randomized_svd_power_iterations: int = 2


@dataclass(frozen=True)
class SolveResult:
    """Results, convergence diagnostics, and timing breakdown returned by solve().
    
    Attributes:
        estimate: Reconstructed low-rank matrix X_hat = U V^T in R^{m x n}.
        clean_relative_error: Normalized recovery error ||X_hat - X*||_F / ||X*||_F.
        objective_history: Tuple of Cauchy objective values evaluated at each outer iteration.
        protected_fraction: Actual empirical fraction of observations protected in FP32 pool.
        u: Final orthogonal factor matrix U in St(m, r).
        v: Final coefficient factor matrix V in R^{n x r}.
        active_bits_per_observation: Physical bit cost per entry (payload + mask + scale overhead).
        routing_score_seconds: Wall-clock time spent computing influence/quantization scores.
        routing_selection_seconds: Wall-clock time spent selecting the protected set (partitioning).
        total_seconds: Total execution time of the solve() call.
        reroute_count: Total number of dynamic routing updates executed during iterations.
        mean_mask_churn: Average fraction of bits flipped between successive routing masks.
    """
    estimate: np.ndarray
    clean_relative_error: float
    objective_history: tuple[float, ...]
    protected_fraction: float
    u: np.ndarray
    v: np.ndarray
    active_bits_per_observation: float
    routing_score_seconds: float
    routing_selection_seconds: float
    total_seconds: float
    reroute_count: int
    mean_mask_churn: float


# Hardware format specifications: (exponent_bits, mantissa_bits)
FLOAT_LAYOUTS: dict[str, tuple[int, int]] = {
    "tf32": (8, 10),
    "bf16": (8, 7),
    "fp8_e4m3": (4, 3),
    "fp8_e5m2": (5, 2),
    "fp6_e3m2": (3, 2),
    "fp6_e2m3": (2, 3),
    "fp4_e2m1": (2, 1),
}

# Total physical bits allocated per data element
FORMAT_BITS: dict[str, int] = {
    "fp64": 64, "fp32": 32, "tf32": 19, "fp16": 16, "bf16": 16,
    "fp8_e4m3": 8, "fp8_e5m2": 8, "fp6_e3m2": 6, "fp6_e2m3": 6,
    "fp4_e2m1": 4, "int8": 8, "int4": 4,
}

# Formats utilizing block-scaled microscaling (MX / NVFP) with shared exponents
BLOCK_SCALED_FORMATS: set[str] = {
    "fp8_e4m3", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3", "fp4_e2m1",
    "int8", "int4",
}


# ==================================================================================================
# 2. Synthetic Problem Generator
# ==================================================================================================

def generate_problem(
    m: int = 128,
    n: int = 128,
    rank: int = 8,
    observation_probability: float = 0.70,
    noise_fraction: float = 0.02,
    corruption_fraction: float = 0.10,
    corruption_scale: float = 8.0,
    condition_number: float | None = None,
    oversampling_ratio: float | None = None,
    seed: int = 0,
) -> Problem:
    """Generates a synthetic low-rank matrix completion problem with controlled outliers.
    
    Constructs X* = U V^T where:
      1. U in St(m, r) is drawn uniformly from the Haar measure (via QR decomposition).
      2. V has geometrically spaced singular values matching condition number kappa = sigma_1 / sigma_r.
      3. A random subset Omega is observed, corrupted by dense Gaussian noise and sparse gross outliers.
    
    Args:
        m: Number of rows.
        n: Number of columns.
        rank: Ground-truth low rank r.
        observation_probability: Probability of observing each entry.
        noise_fraction: Standard deviation of dense Gaussian noise relative to data std.
        corruption_fraction: Fraction of observed entries corrupted by gross outliers.
        corruption_scale: Amplitude of sparse outliers relative to data std.
        condition_number: Desired condition number sigma_1 / sigma_r (if None, standard normal).
        oversampling_ratio: If given, overrides observation_probability based on matrix degrees-of-freedom.
        seed: Random seed for reproducibility.
        
    Returns:
        A Problem instance containing ground truth and observed noisy entries.
    """
    if rank > min(m, n):
        raise ValueError(f"rank ({rank}) must not exceed matrix dimensions ({m}x{n})")
    if condition_number is not None and condition_number < 1.0:
        raise ValueError("condition_number must be >= 1.0")

    rng = np.random.default_rng(seed)
    
    # 1. Sample orthogonal basis U in St(m, r)
    u, _ = np.linalg.qr(rng.standard_normal((m, rank)), mode="reduced")
    
    # 2. Sample coefficient factor V with prescribed condition number
    if condition_number is None:
        v = rng.standard_normal((n, rank))
    else:
        v_basis, _ = np.linalg.qr(rng.standard_normal((n, rank)), mode="reduced")
        singular = np.geomspace(1.0, 1.0 / condition_number, rank)
        singular *= np.sqrt(n * rank / np.dot(singular, singular))
        v = v_basis * singular

    # 3. Form clean ground truth X* = U V^T
    clean = u @ v.T
    singular = np.linalg.svd(v, compute_uv=False)
    realized_condition = float(singular[0] / singular[-1])
    data_scale = float(np.std(clean))

    # 4. Dense Gaussian noise: e_ij ~ N(0, (noise_fraction * std)^2)
    noisy = clean + noise_fraction * data_scale * rng.standard_normal((m, n))

    # 5. Missing-at-random observation mask Omega
    degrees_of_freedom = rank * (m + n - rank)
    if oversampling_ratio is not None:
        observation_probability = min(1.0, oversampling_ratio * degrees_of_freedom / (m * n))
    observed = rng.random((m, n)) < observation_probability

    # 6. Heavy-tailed corruptions: gross sparse outliers (simulating sensor spikes)
    corrupt = observed & (rng.random((m, n)) < corruption_fraction)
    noisy[corrupt] += corruption_scale * data_scale * rng.standard_normal(np.count_nonzero(corrupt))

    rows, cols = np.nonzero(observed)
    realized_oversampling = rows.size / degrees_of_freedom

    return Problem(
        clean=clean,
        observed=observed,
        rows=rows,
        cols=cols,
        values=noisy[rows, cols].astype(np.float64),
        clean_condition_number=realized_condition,
        observation_probability=float(np.mean(observed)),
        oversampling_ratio=realized_oversampling,
    )


# ==================================================================================================
# 3. Hardware Storage Emulation & Microscaled Quantization
# ==================================================================================================

def _round_binary(values: np.ndarray, exponent_bits: int, mantissa_bits: int) -> np.ndarray:
    """Rounds values to a finite binary floating-point representation.
    
    Emulates bit-level IEEE floating-point arithmetic including normal numbers,
    subnormal numbers, exponent bias, and finite overflow saturation.
    """
    x = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(x)
    finite = np.isfinite(x) & (x != 0)
    if not np.any(finite):
        return result

    ax = np.abs(x[finite])
    sign = np.sign(x[finite])
    bias = 2 ** (exponent_bits - 1) - 1
    min_exp = 1 - bias
    max_exp = (2**exponent_bits - 2) - bias
    min_subnormal = 2.0 ** (min_exp - mantissa_bits)
    max_finite = (2.0 - 2.0 ** (-mantissa_bits)) * 2.0**max_exp

    exponent = np.floor(np.log2(ax))
    normal = exponent >= min_exp
    rounded = np.empty_like(ax)

    # Normal numbers
    if np.any(normal):
        e = np.minimum(exponent[normal], max_exp)
        step = np.exp2(e - mantissa_bits)
        rounded[normal] = np.rint(ax[normal] / step) * step

    # Subnormal numbers
    if np.any(~normal):
        rounded[~normal] = np.rint(ax[~normal] / min_subnormal) * min_subnormal

    result[finite] = sign * np.minimum(rounded, max_finite)
    return result


def _block_scaled_float(
    values: np.ndarray,
    exponent_bits: int,
    mantissa_bits: int,
    block_size: int = 32,
) -> np.ndarray:
    """Emulates Open Compute Project (OCP) / NVFP microscaling (MX) quantization.
    
    Divides the input vector into contiguous blocks (default: 32 elements). For each block:
      1. Computes the peak absolute magnitude.
      2. Determines a shared power-of-two scale factor: scale = 2^{ceil(log2(peak / max_finite))}.
      3. Normalizes each element and rounds to the target low-bit float representation.
      4. Re-scales back to the physical dynamic range.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    result = np.empty_like(flat)
    bias = 2 ** (exponent_bits - 1) - 1
    max_exp = (2**exponent_bits - 2) - bias
    max_finite = (2.0 - 2.0 ** (-mantissa_bits)) * 2.0**max_exp

    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        peak = float(np.max(np.abs(block), initial=0.0))
        scale = 1.0 if peak == 0 else 2.0 ** np.ceil(np.log2(peak / max_finite))
        result[start : start + block.size] = (
            _round_binary(block / scale, exponent_bits, mantissa_bits) * scale
        )

    return result.reshape(np.shape(values))


def _block_scaled_integer(values: np.ndarray, bits: int, block_size: int = 32) -> np.ndarray:
    """Emulates block-scaled integer quantization (int8, int4) with symmetric dynamic range."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    result = np.empty_like(flat)
    qmax = 2 ** (bits - 1) - 1

    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        peak = float(np.max(np.abs(block), initial=0.0))
        scale = peak / qmax if peak else 1.0
        result[start : start + block.size] = np.clip(np.rint(block / scale), -qmax, qmax) * scale

    return result.reshape(np.shape(values))


def quantize(values: np.ndarray, format_name: FormatName, block_size: int = 32) -> np.ndarray:
    """Quantizes floating-point values to the discrete grid of a target hardware storage format.
    
    Supports FP64, FP32, TF32, FP16, BF16, microscaled FP8 (E4M3, E5M2), FP6, FP4 (E2M1), INT8, INT4.
    """
    x = np.asarray(values, dtype=np.float64)
    if format_name == "fp64":
        return x.copy()
    if format_name == "fp32":
        return x.astype(np.float32).astype(np.float64)
    if format_name == "fp16":
        return x.astype(np.float16).astype(np.float64)
    if format_name in ("int8", "int4"):
        return _block_scaled_integer(x, int(format_name[3:]), block_size)

    try:
        exponent_bits, mantissa_bits = FLOAT_LAYOUTS[format_name]
    except KeyError as exc:
        raise ValueError(f"Unknown storage format: {format_name}") from exc

    if format_name in ("tf32", "bf16"):
        return _round_binary(x, exponent_bits, mantissa_bits)

    return _block_scaled_float(x, exponent_bits, mantissa_bits, block_size)


def active_storage_bits_per_observation(
    format_name: FormatName,
    protected: np.ndarray,
    quantizer_block_size: int = 32,
    scale_bits: int = 16,
) -> float:
    """Calculates the exact physical bit-rate per observation for dual-pool routed storage.
    
    Total Physical Storage Accounting:
      - Protected Pool: 32 bits per entry (FP32).
      - Low-Precision Pool: FORMAT_BITS[format] bits per entry (e.g., 4 bits for FP4).
      - Routing Mask: Exactly 1 bit per observation for hardware address redirection.
      - Microscaling Scales: scale_bits (16 bits) shared per block of 32 low-precision entries.
    """
    count = protected.size
    if count == 0:
        return 0.0

    low = ~protected
    scale_count = 0
    if format_name in BLOCK_SCALED_FORMATS and np.any(low):
        padded = np.pad(low, (0, (-count) % quantizer_block_size))
        scale_count = int(np.count_nonzero(padded.reshape(-1, quantizer_block_size).any(axis=1)))

    total = (
        32 * int(np.count_nonzero(protected))
        + FORMAT_BITS[format_name] * int(np.count_nonzero(low))
        + count                      # 1-bit routing mask per observation
        + scale_bits * scale_count   # Shared block exponent scales
    )
    return total / count


def storage_bytes_per_observation(format_name: FormatName, protected_fraction: float) -> float:
    """Computes idealized payload bytes per observation excluding metadata overhead."""
    return (
        protected_fraction * 32.0
        + (1.0 - protected_fraction) * FORMAT_BITS[format_name]
    ) / 8.0


# ==================================================================================================
# 4. Robust Cauchy M-Estimation & Scale Annealing
# ==================================================================================================

def _weights(residual: np.ndarray, c: float, scale_floor: float) -> tuple[np.ndarray, float]:
    """Computes robust IRLS weights using the redescending Cauchy loss function.
    
    1. Scale Estimation (MAD):
       sigma = 1.4826 * median(|residual|)
       The constant 1.4826 ensures Fisher-consistent estimation of standard deviation for Gaussians:
           1 / (sqrt(2) * erfinv(0.5)) = 1.4826022...
    
    2. Cauchy Weight Function:
       For loss rho(r) = (c^2 sigma^2 / 2) * ln(1 + (r / (c sigma))^2),
       the influence function is psi(r) = rho'(r) = r / (1 + (r / (c sigma))^2).
       The corresponding IRLS weighting function is:
           w(r) = psi(r) / r = 1 / (1 + (r / (c sigma))^2)
       
       Notice that as |r| -> inf, w(r) -> 0 at rate O(1/r^2), completely nullifying severe outliers.
    """
    # MAD scale estimator
    sigma = max(1.4826 * float(np.median(np.abs(residual))), scale_floor)
    scaled = residual / (c * sigma)
    # Cauchy redescending weights in (0, 1]
    weights = 1.0 / (1.0 + scaled * scaled)
    return weights, sigma


# ==================================================================================================
# 5. Stiefel Manifold Optimization (Differential Geometry)
# ==================================================================================================

def _retract(u: np.ndarray) -> np.ndarray:
    """Projects an ambient matrix back onto the Stiefel manifold St(m, r) via Polar Retraction.
    
    Given U in R^{m x r} with rank r, its polar factor is:
        R_U = U (U^T U)^{-1/2}
    
    Computed stably via thin SVD: if U = L Sigma R^T, then:
        R_U = L R^T
    
    Verification of Stiefel condition:
        (L R^T)^T (L R^T) = R L^T L R^T = R I_r R^T = I_r
    
    Reference:
        Cambier & Absil, 'Robust low-rank matrix completion on the Stiefel manifold',
        SIAM J. Sci. Comput. 38(5), 2016.
    """
    left, _, right_t = np.linalg.svd(u, full_matrices=False)
    return left @ right_t


def _truncated_svd(
    matrix: np.ndarray,
    rank: int,
    config: SolverConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes a rank-r truncated SVD, using randomized range finding for large matrices."""
    if max(matrix.shape) <= config.randomized_svd_threshold:
        left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        return left[:, :rank], singular[:rank], right_t[:rank]

    # Halko et al. (2011) Randomized SVD with power iterations
    sample_rank = min(rank + config.randomized_svd_oversampling, min(matrix.shape))
    omega = rng.standard_normal((matrix.shape[1], sample_rank))
    basis, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(config.randomized_svd_power_iterations):
        right_basis, _ = np.linalg.qr(matrix.T @ basis, mode="reduced")
        basis, _ = np.linalg.qr(matrix @ right_basis, mode="reduced")
    small = basis.T @ matrix
    small_left, singular, right_t = np.linalg.svd(small, full_matrices=False)
    return basis @ small_left[:, :rank], singular[:rank], right_t[:rank]


def _initial_factors(
    problem: Problem,
    config: SolverConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Initializes factors (U, V) via outlier-clipped spectral initialization.
    
    Observations are clipped at 1.5 * MAD from the median to ensure extreme corruptions
    do not distort the initial subspace.
    """
    matrix = np.zeros_like(problem.clean)
    median = float(np.median(problem.values))
    mad = 1.4826 * float(np.median(np.abs(problem.values - median)))
    limit = max(1.5 * mad, np.finfo(float).eps)
    clipped = np.clip(problem.values, median - limit, median + limit)
    matrix[problem.rows, problem.cols] = clipped
    # Scale adjustment for observation density
    matrix *= matrix.size / max(problem.values.size, 1)

    left, singular, right_t = _truncated_svd(matrix, config.rank, config, rng)
    u = left[:, :config.rank]
    v = right_t[:config.rank, :].T * singular[:config.rank]
    return u, v


# ==================================================================================================
# 6. Factor Updates (Alternating Riemannian IRLS)
# ==================================================================================================

def _column_groups(cols: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """Pre-computes sorted observation indices grouped by column for fast V-step solves."""
    order = np.argsort(cols, kind="stable")
    counts = np.bincount(cols, minlength=n)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return tuple(order[offsets[j] : offsets[j + 1]] for j in range(n))


def _update_v(
    u: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n: int,
    ridge: float,
    column_groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Updates coefficient factor V in R^{n x r} by solving weighted normal equations per column.
    
    For column j:
        (U_{Omega_j}^T W_j U_{Omega_j} + lambda * I_r) v_j = U_{Omega_j}^T (W_j y_{Omega_j})
    """
    rank = u.shape[1]
    v = np.zeros((n, rank), dtype=np.float64)
    eye = np.eye(rank)

    for column, selected in enumerate(column_groups):
        if selected.size == 0:
            continue
        design = u[rows[selected]]       # Rows of U observed in this column
        w = weights[selected]            # Cauchy weights for this column's entries
        normal = design.T @ (w[:, None] * design) + ridge * eye
        rhs = design.T @ (w * y[selected])
        v[column] = np.linalg.solve(normal, rhs)

    return v


# ==================================================================================================
# 7. Influence Scoring & Precision Routing Policies
# ==================================================================================================

def _routing_scores(
    policy: PolicyName,
    residual: np.ndarray,
    values: np.ndarray,
    low_values: np.ndarray,
    weights: np.ndarray,
    v_rows: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Computes routing importance scores for each observation under the chosen policy.
    
    Policies:
      - influence:   Score_ij = w_ij * |r_ij| * ||v_j||_2
                     First-order gradient norm sensitivity ||nabla_U L_ij||_2.
      - quant_error: Score_ij = w_ij * |Y_ij - Q(Y_ij)| * ||v_j||_2
                     First-order Taylor approximation of loss change under quantization error.
      - data:        Score_ij = w_ij * |Y_ij| * ||v_j||_2 (data magnitude heuristic).
      - weight:      Score_ij = w_ij (pure Cauchy inlier confidence).
      - random:      Uniform random selection (unbiased baseline).
    """
    leverage = np.linalg.norm(v_rows, axis=1)  # Statistical leverage of column j

    if policy == "influence":
        return weights * np.abs(residual) * leverage
    if policy == "quant_error":
        return weights * np.abs(values - low_values) * leverage
    if policy == "data":
        return weights * np.abs(values) * leverage
    if policy == "weight":
        return weights
    if policy == "random":
        return rng.random(values.size)

    raise ValueError(f"Unknown routing policy: {policy}")


def _select_protected(
    scores: np.ndarray,
    protected_count: int,
    routing_granularity: int = 1,
) -> np.ndarray:
    """Standard global selection: protects the entries with highest routing scores."""
    protected = np.zeros(scores.size, dtype=bool)
    if protected_count <= 0:
        return protected
    if protected_count >= scores.size:
        protected.fill(True)
        return protected

    if routing_granularity == 1:
        indices = np.argpartition(scores, -protected_count)[-protected_count:]
        protected[indices] = True
        return protected

    # Block-granularity selection
    starts = np.arange(0, scores.size, routing_granularity)
    block_count = protected_count // routing_granularity
    if block_count == 0:
        return protected
    block_scores = np.add.reduceat(scores, starts)
    chosen = np.argpartition(block_scores, -block_count)[-block_count:]
    for block in chosen:
        start = starts[block]
        protected[start : min(start + routing_granularity, scores.size)] = True
    return protected


def _select_protected_balanced(
    scores: np.ndarray,
    protected_count: int,
    cols: np.ndarray,
    n_cols: int,
    min_column_fraction: float = 0.05,
) -> np.ndarray:
    """Column-balanced precision routing: guarantees each column a minimum high-precision share.
    
    Under ultra-low-bit formats (FP4), columns with few protected entries can suffer catastrophic
    loss of numerical rank, causing singular normal equations in the V-update.
    
    Two-Phase Allocation:
      Phase 1 (Fairness Quota): For each column j, protect its top entries up to:
          quota_j = ceil(min_column_fraction * |Omega_j|)
      Phase 2 (Global Optimality): Distribute the remaining budget to the entries with
          the highest scores globally across the entire matrix.
    """
    protected = np.zeros(scores.size, dtype=bool)
    if protected_count <= 0:
        return protected
    if protected_count >= scores.size:
        protected.fill(True)
        return protected

    budget_left = protected_count

    # Phase 1: Per-column fairness quota
    for j in range(n_cols):
        col_mask = cols == j
        col_indices = np.where(col_mask)[0]
        if col_indices.size == 0:
            continue
        quota = min(int(np.ceil(min_column_fraction * col_indices.size)), budget_left)
        if quota <= 0:
            continue
        col_scores = scores[col_indices]
        if quota >= col_indices.size:
            protected[col_indices] = True
            budget_left -= col_indices.size
        else:
            top_in_col = np.argpartition(col_scores, -quota)[-quota:]
            protected[col_indices[top_in_col]] = True
            budget_left -= quota
        if budget_left <= 0:
            break

    # Phase 2: Distribute remaining budget globally by score
    if budget_left > 0:
        remaining = ~protected
        remaining_indices = np.where(remaining)[0]
        if remaining_indices.size > 0:
            remaining_scores = scores[remaining_indices]
            take = min(budget_left, remaining_indices.size)
            top_global = np.argpartition(remaining_scores, -take)[-take:]
            protected[remaining_indices[top_global]] = True

    return protected


# ==================================================================================================
# 8. Master Alternating Riemannian IRLS Solver Loop
# ==================================================================================================

def solve(
    problem: Problem,
    config: SolverConfig = SolverConfig(),
    storage_format: FormatName = "fp64",
    protected_fraction: float = 0.0,
    policy: PolicyName = "influence",
    dynamic: bool = False,
    reroute_every: int | None = None,
    routing_granularity: int = 1,
    arithmetic_format: FormatName = "fp64",
    block_size: int = 32,
    seed: int = 0,
    column_balanced: bool = False,
    min_column_fraction: float = 0.05,
) -> SolveResult:
    """Solves robust matrix completion with precision-routed storage and Riemannian optimization.
    
    Workflow:
      1. Spectral initialization of U on St(m, r) and V in R^{n x r}.
      2. Quantize unprotected observations to storage_format.
      3. Outer IRLS Loop:
           a. Anneal Cauchy scale c and compute robust weights w_ij.
           b. (Re-)evaluate precision routing (if dynamic or iteration 0).
           c. Solve regularized weighted least-squares for V (column-by-column).
           d. Inner Riemannian loop on U: compute Euclidean gradient, project to Stiefel
              tangent space, perform Armijo line search along polar retraction path.
      4. Return SolveResult with error metrics, active bit accounting, and timing.
    
    Args:
        problem: Partially observed problem instance.
        config: Solver tolerances, ranks, and iteration limits.
        storage_format: Coarse storage format for unprotected entries (e.g. 'fp4_e2m1', 'fp8_e4m3').
        protected_fraction: Budget beta in [0, 1] of observations allocated to the FP32 pool.
        policy: Routing score policy ('influence', 'quant_error', 'random').
        dynamic: If True, periodically re-evaluates precision routing as factor estimates converge.
        reroute_every: Iteration frequency for dynamic re-routing (default: 1 if dynamic else 0).
        routing_granularity: Element or block granularity for routing decisions.
        arithmetic_format: Precision used for gradient computation and retraction arithmetic.
        block_size: Element count per microscaled quantization block.
        seed: Random seed for initialization and tie-breaking.
        column_balanced: If True, activates column-balanced fairness quota allocation.
        min_column_fraction: Minimum fraction of entries protected per column under column balancing.
        
    Returns:
        SolveResult containing the recovered matrix, relative error, objective history, and timing.
    """
    if not 0.0 <= protected_fraction <= 1.0:
        raise ValueError("protected_fraction must be in [0, 1]")
    if routing_granularity < 1:
        raise ValueError("routing_granularity must be positive")
    if reroute_every is None:
        reroute_every = 1 if dynamic else 0
    if reroute_every < 0:
        raise ValueError("reroute_every must be nonnegative")

    started = perf_counter()
    rng = np.random.default_rng(seed)

    # 1. Initialize factors: U in St(m, r) and V in R^{n x r}
    u, v = _initial_factors(problem, config, rng)

    # 2. Pre-quantize all observed entries to the coarse format
    low_values = quantize(problem.values, storage_format, block_size)

    # 3. Setup precision routing budget
    protected_count = int(np.floor(protected_fraction * problem.values.size))
    protected = np.zeros(problem.values.size, dtype=bool)
    if protected_count >= problem.values.size:
        protected.fill(True)

    column_groups = _column_groups(problem.cols, problem.clean.shape[1])
    history: list[float] = []
    score_seconds = 0.0
    selection_seconds = 0.0
    reroute_count = 0
    churn: list[float] = []
    previous_protected: np.ndarray | None = None

    # =========================================================================
    # Outer IRLS Loop
    # =========================================================================
    for iteration in range(config.outer_iterations):
        # 1. Current full residual: r_ij = u_i^T v_j - Y_ij
        full_residual = np.sum(u[problem.rows] * v[problem.cols], axis=1) - problem.values

        # 2. Cauchy scale annealing: cools from 3*c down to c
        anneal = min(iteration / max(config.anneal_iterations, 1), 1.0)
        current_c = config.cauchy_c * (3.0 - 2.0 * anneal)
        full_weights, _ = _weights(full_residual, current_c, config.scale_floor)

        # 3. Precision Re-Routing Check
        should_reroute = (
            0 < protected_count < problem.values.size
            and (iteration == 0 or (reroute_every > 0 and iteration % reroute_every == 0))
        )
        if should_reroute:
            score_started = perf_counter()
            scores = _routing_scores(
                policy, full_residual, problem.values, low_values, full_weights,
                v[problem.cols], rng,
            )
            score_seconds += perf_counter() - score_started

            selection_started = perf_counter()
            if column_balanced:
                new_protected = _select_protected_balanced(
                    scores, protected_count, problem.cols,
                    problem.clean.shape[1], min_column_fraction,
                )
            else:
                new_protected = _select_protected(scores, protected_count, routing_granularity)
            selection_seconds += perf_counter() - selection_started

            if previous_protected is not None:
                churn.append(float(np.mean(new_protected != previous_protected)))
            protected = new_protected
            previous_protected = protected.copy()
            reroute_count += 1

        # 4. Form routed observation vector: high-precision where protected, low-precision elsewhere
        routed_values = np.where(protected, problem.values, low_values)
        residual = np.sum(u[problem.rows] * v[problem.cols], axis=1) - routed_values
        weights, sigma = _weights(residual, current_c, config.scale_floor)

        # 5. V-Step: Closed-form regularized weighted least squares per column
        v = _update_v(
            u, problem.rows, problem.cols, routed_values, weights,
            problem.clean.shape[1], config.ridge, column_groups,
        )
        if arithmetic_format != "fp64":
            v = quantize(v, arithmetic_format, block_size)

        # =====================================================================
        # Inner Riemannian Optimization on U in St(m, r)
        # =====================================================================
        for _ in range(config.inner_iterations):
            residual = np.sum(u[problem.rows] * v[problem.cols], axis=1) - routed_values
            if arithmetic_format != "fp64":
                residual = quantize(residual, arithmetic_format, block_size)

            # Euclidean gradient: G = sum_{(i,j) in Omega} 2 * w_ij * r_ij * (e_i v_j^T)
            gradient = np.zeros_like(u)
            np.add.at(gradient, problem.rows, 2.0 * (weights * residual)[:, None] * v[problem.cols])

            # Orthogonal projection onto Stiefel tangent space:
            # P_{T_U}(G) = G - U ((U^T G + G^T U) / 2)
            tangent = gradient - u @ ((u.T @ gradient + gradient.T @ u) / 2.0)
            if arithmetic_format != "fp64":
                tangent = quantize(tangent, arithmetic_format, block_size)

            # Conservative Lipschitz step bound
            lipschitz = max(2.0 * np.linalg.norm(v, 2) ** 2, 1e-12)
            step = 1.0 / lipschitz
            base_loss = float(np.dot(weights, residual * residual))
            descent = float(np.sum(tangent * tangent))

            # Armijo backtracking line search along the retraction path
            for _ in range(16):
                candidate = _retract(u - step * tangent)
                if arithmetic_format != "fp64":
                    candidate = _retract(quantize(candidate, arithmetic_format, block_size))
                candidate_residual = (
                    np.sum(candidate[problem.rows] * v[problem.cols], axis=1)
                    - routed_values
                )
                candidate_loss = float(np.dot(weights, candidate_residual * candidate_residual))

                # Armijo sufficient decrease condition: f(new) <= f(old) - c_1 * step * ||tangent||^2
                if candidate_loss <= base_loss - 1e-4 * step * descent:
                    u = candidate
                    break
                step *= 0.5

        # Record objective history: Cauchy negative log-likelihood
        final_residual = np.sum(u[problem.rows] * v[problem.cols], axis=1) - routed_values
        history.append(float(np.mean(np.log1p((final_residual / (current_c * sigma)) ** 2))))

    # 6. Final reconstruction: X_hat = U V^T
    estimate = u @ v.T
    error = float(np.linalg.norm(estimate - problem.clean) / np.linalg.norm(problem.clean))
    actual_fraction = float(np.mean(protected))

    return SolveResult(
        estimate=estimate,
        clean_relative_error=error,
        objective_history=tuple(history),
        protected_fraction=actual_fraction,
        u=u,
        v=v,
        active_bits_per_observation=active_storage_bits_per_observation(storage_format, protected, block_size),
        routing_score_seconds=score_seconds,
        routing_selection_seconds=selection_seconds,
        total_seconds=perf_counter() - started,
        reroute_count=reroute_count,
        mean_mask_churn=float(np.mean(churn)) if churn else 0.0,
    )
