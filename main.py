"""Section 7 conditional spectral-gap benchmarks for AQC Max-Cut paths.

The benchmark studies

    H(s) = (1-s) H_I + s H_P,

in the even-parity sector.  Sparse Ritz/residual lower bounds are conditional on
correct eigenvalue indexing.  Analytic endpoint bounds and the continuation
use the shift-invariant gap Lipschitz bound

    K_gap_cert = upward_binary64(W_upper + Lambda_I),
    Lambda_I = 2 floor(n_v / 2),

where ``W_upper`` is the least binary64 value not below the exact dyadic sum
of the archived binary64 edge weights.  Thus
``-Lambda_I I <= H_P-H_I <= W_exact I <= W_upper I``.
"""

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from fractions import Fraction

import numpy as np
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh


BENCHMARK_NS = (10, 12)
BENCHMARK_SEEDS = tuple(range(20))
DEFAULT_ETA = 0.9
DEFAULT_H_FLOOR = 1e-6
DEFAULT_SOLVER_TOL = 1e-10
_UNRESOLVED_PROBE_INITIAL = 1e-3
_UNRESOLVED_PROBE_MAX = 5e-2
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
_RUNTIME_PROVENANCE = None


# ----------------------------- [7.1] instances -----------------------------

def _exact_binary64_sum(values):
    """Sum finite binary64 values as exact dyadic rationals."""
    total = Fraction(0)
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("edge weights must be finite binary64 values")
        total += Fraction.from_float(value)
    return total


def _least_binary64_not_below(exact_value):
    """Return the least finite binary64 value greater than or equal to a rational."""
    exact_value = Fraction(exact_value)
    candidate = float(exact_value)
    if not math.isfinite(candidate):
        raise OverflowError("exact bound is outside the finite binary64 range")
    if Fraction.from_float(candidate) < exact_value:
        candidate = math.nextafter(candidate, math.inf)
    if not math.isfinite(candidate):
        raise OverflowError("upward-rounded bound is not finite")
    predecessor = math.nextafter(candidate, -math.inf)
    if Fraction.from_float(candidate) < exact_value:
        raise ArithmeticError("failed to round the rational bound upward")
    if math.isfinite(predecessor) and Fraction.from_float(predecessor) >= exact_value:
        raise ArithmeticError("computed binary64 upper bound is not minimal")
    return candidate


def _analytic_weight_bounds(edges, Lambda_I):
    weights = [float(edge[2]) for edge in edges]
    W_model = float(sum(weights))
    if not math.isfinite(W_model):
        raise OverflowError("nominal edge-weight sum is not finite")
    W_exact = _exact_binary64_sum(weights)
    W_upper = _least_binary64_not_below(W_exact)
    W_upper_exact = Fraction.from_float(W_upper)
    K_gap_exact = W_exact + int(Lambda_I)
    K_gap_from_W_upper_exact = W_upper_exact + int(Lambda_I)
    K_gap_cert = _least_binary64_not_below(K_gap_from_W_upper_exact)
    K_gap_cert_exact = Fraction.from_float(K_gap_cert)
    if K_gap_cert_exact < K_gap_exact:
        raise ArithmeticError("K_gap_cert does not upper-bound W_exact + Lambda_I")
    return {
        "W_model": W_model,
        "W_exact": W_exact,
        "W_upper": W_upper,
        "W_upper_exact": W_upper_exact,
        "K_gap_exact": K_gap_exact,
        "K_gap_from_W_upper_exact": K_gap_from_W_upper_exact,
        "K_gap_cert": K_gap_cert,
        "K_gap_cert_exact": K_gap_cert_exact,
    }


def _fraction_metadata(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def er_graph(n, p, rng):
    return [
        (i, j, rng.uniform(0.5, 1.5))
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < p
    ]


def is_connected(n, edges):
    """Return whether the undirected graph induced by ``edges`` is connected."""
    if n == 0:
        return True
    adjacency = [[] for _ in range(n)]
    for u, v, *_ in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = {0}
    frontier = [0]
    while frontier:
        vertex = frontier.pop()
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == n


def pinned_cost_vector(N, edges):
    """Return uncut-edge costs with vertex 0 pinned to spin +1."""
    x = np.arange(1 << N, dtype=np.int64)
    c = np.zeros(1 << N, dtype=np.float64)
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1.0
        u, v = min(u, v), max(u, v)
        if u == 0:
            c += w * (((x >> (v - 1)) & 1) == 0)
        else:
            c += w * (
                ((x >> (u - 1)) & 1) == ((x >> (v - 1)) & 1)
            )
    return c


def uncut_of_assignment(edges, assignment):
    """Evaluate one cut in polynomial time; vertex 0 is fixed by the caller."""
    return sum(
        edge[2] * (assignment[edge[0]] == assignment[edge[1]])
        if len(edge) > 2
        else (assignment[edge[0]] == assignment[edge[1]])
        for edge in edges
    )


def greedy_cut(n, edges, rng, restarts=30):
    """Run deterministic-with-respect-to-``rng`` 1-exchange local search."""
    adjacency = [[] for _ in range(n)]
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1.0
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))

    best = None
    for _ in range(restarts):
        assignment = rng.integers(0, 2, n).astype(bool)
        assignment[0] = True
        moved = True
        while moved:
            moved = False
            for vertex in range(1, n):
                same_weight = sum(
                    weight
                    for neighbor, weight in adjacency[vertex]
                    if assignment[neighbor] == assignment[vertex]
                )
                different_weight = sum(
                    weight
                    for neighbor, weight in adjacency[vertex]
                    if assignment[neighbor] != assignment[vertex]
                )
                if same_weight > different_weight:
                    assignment[vertex] = ~assignment[vertex]
                    moved = True
        value = uncut_of_assignment(edges, assignment)
        if best is None or value < best[0]:
            best = (value, assignment.copy())
    if best is None:
        raise ValueError("restarts must be positive")
    return best


def spectral_maxcut_ub(n, edges):
    """Return the spectral Max-Cut upper bound n*lambda_max(L)/4."""
    laplacian = np.zeros((n, n))
    for edge in edges:
        u, v = edge[0], edge[1]
        weight = edge[2] if len(edge) > 2 else 1.0
        laplacian[u, u] += weight
        laplacian[v, v] += weight
        laplacian[u, v] -= weight
        laplacian[v, u] -= weight
    W = (
        sum(edge[2] for edge in edges)
        if edges and len(edges[0]) > 2
        else float(len(edges))
    )
    return min(W, n * np.linalg.eigvalsh(laplacian)[-1] / 4.0)


# ------------------- matvec engine + conditional Ritz bounds ---------------

class _CountingLinearOperator(LinearOperator):
    def __init__(self, matrix, owner):
        self._matrix = matrix
        self._owner = owner
        super().__init__(dtype=np.dtype(np.float64), shape=matrix.shape)

    def _matvec(self, x):
        self._owner.matvecs += 1
        return self._matrix.dot(x)


class PathOperator:
    def __init__(self, N, c):
        self.N = N
        self.d = 1 << N
        self.c = c
        self.matvecs = 0

        import scipy.sparse as sp
        from maxcut_gap_benchmark import driver_matrix

        self.HI_sparse = driver_matrix(N + 1, sector=True).tocsr()
        self.HP_sparse = sp.diags(c, format="csr")

    def H(self, s):
        matrix = (1.0 - s) * self.HI_sparse + s * self.HP_sparse
        return _CountingLinearOperator(matrix, self)

    def D(self):
        matrix = self.HP_sparse - self.HI_sparse
        return _CountingLinearOperator(matrix, self)


def _deterministic_start(size, salt=0):
    seed = np.random.SeedSequence([0x5A17, int(size), int(salt)])
    return np.random.default_rng(seed).standard_normal(size)


def lowest_two(Hop, v0=None, tol=DEFAULT_SOLVER_TOL):
    """Return two Ritz pairs and residual norms.

    The lower bound ``theta1-r1-theta0`` is conditional on ``theta1`` having
    the correct eigenvalue index.  ``theta0`` is a Rayleigh--Ritz upper bound
    in exact arithmetic; all returned values remain floating-point numerical
    outputs.
    """
    dimension = Hop.shape[0]
    if dimension <= 2:
        raise ValueError("lowest_two requires an operator dimension above 2")
    if v0 is None:
        v0 = _deterministic_start(dimension)
    try:
        values, vectors = eigsh(
            Hop,
            k=2,
            which="SA",
            v0=v0,
            tol=tol,  # pyright: ignore[reportArgumentType]
            ncv=min(dimension, 50),
        )
    except ArpackError as exc:
        if "Starting vector is zero" not in str(exc):
            raise
        values, vectors = eigsh(
            Hop,
            k=2,
            which="SA",
            v0=_deterministic_start(dimension, salt=7919),
            tol=tol,  # pyright: ignore[reportArgumentType]
            ncv=min(dimension, 50),
        )
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    residuals = np.asarray(
        [
            np.linalg.norm(Hop @ vectors[:, j] - values[j] * vectors[:, j])
            for j in range(2)
        ]
    )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(residuals)):
        raise FloatingPointError("non-finite Ritz output")
    return values, vectors, residuals


def spectral_diameter_diagnostics(pathop, tol=DEFAULT_SOLVER_TOL):
    """Estimate the extreme eigenvalues and spectral diameter of D.

    These are ordinary floating-point eigensolver diagnostics, not exact
    values and not substitutes for ``K_gap_cert``.
    """
    operator = pathop.D()
    dimension = operator.shape[0]
    lambda_max_est = float(
        eigsh(
            operator,
            k=1,
            which="LA",
            v0=_deterministic_start(dimension, 1),
            tol=tol,  # pyright: ignore[reportArgumentType]
            return_eigenvectors=False,
        )[0]
    )
    lambda_min_est = float(
        eigsh(
            operator,
            k=1,
            which="SA",
            v0=_deterministic_start(dimension, 2),
            tol=tol,  # pyright: ignore[reportArgumentType]
            return_eigenvectors=False,
        )[0]
    )
    return {
        "lambda_min_est": lambda_min_est,
        "lambda_max_est": lambda_max_est,
        "spectral_diameter_est": lambda_max_est - lambda_min_est,
        "operator_norm_est": max(abs(lambda_min_est), abs(lambda_max_est)),
    }


def path_lipschitz(pathop, tol=DEFAULT_SOLVER_TOL):
    """Compatibility API returning a floating-point operator-norm estimate."""
    return spectral_diameter_diagnostics(pathop, tol)["operator_norm_est"]


# ------------------------- [7.2] endpoint bounds ---------------------------

def _validate_K_gap(K_gap):
    K_gap = float(K_gap)
    if not np.isfinite(K_gap) or K_gap < 0.0:
        raise ValueError("K_gap must be finite and nonnegative")
    return K_gap


def weyl_endpoint_cert(s, gap0, gap1, K_gap):
    """Evaluate the shift-invariant endpoint Weyl envelope."""
    K_gap = _validate_K_gap(K_gap)
    return max(gap0 - s * K_gap, gap1 - (1.0 - s) * K_gap)


def psd_floor_cert(s, E1I, E1P, ceilings):
    """Evaluate the PSD-floor endpoint bound for the supplied ceilings."""
    return max((1.0 - s) * E1I, s * E1P) - min(
        (1.0 - s) * a + s * b for a, b in ceilings
    )


# ------------- [7.3] Algorithm 1: conditional continuation ----------------

def _validate_sweep_inputs(K_gap_cert, eta, h_floor):
    K_gap_cert = _validate_K_gap(K_gap_cert)
    eta = float(eta)
    h_floor = float(h_floor)
    if not np.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must be finite and lie strictly between 0 and 1")
    if not np.isfinite(h_floor) or h_floor <= 0.0:
        raise ValueError("h_floor must be finite and positive")
    return K_gap_cert, eta, h_floor


def _merge_windows(windows):
    merged = []
    for start, end in windows:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def conditional_sweep(
    pathop,
    K_gap_cert,
    eta=DEFAULT_ETA,
    h_floor=DEFAULT_H_FLOOR,
    tol=DEFAULT_SOLVER_TOL,
    max_anchors=None,
):
    """Run the conditional adaptive continuation sweep.

    A normal proposal is ``eta*delta_lo/K_gap_cert``.  If it falls below
    ``h_floor``, repeated floor triggers use mark-and-probe advancement: the
    planned probe starts at ``max(h_floor, 1e-3)``, doubles after consecutive
    triggers, and is capped at ``max(h_floor, 0.05)``.  The full skipped region
    is recorded as unresolved whenever its actual-length bound is nonpositive.
    Thus probing can skip redundant 1e-6 solves without silently claiming
    coverage; the positive minimum advancement also guarantees termination.

    The return shape matches the historical sweep API:
    ``(records, unresolved_windows, conditional_intervals)``.
    """
    K_gap_cert, eta, h_floor = _validate_sweep_inputs(
        K_gap_cert, eta, h_floor
    )
    if max_anchors is not None:
        if not isinstance(max_anchors, (int, np.integer)) or max_anchors <= 0:
            raise ValueError("max_anchors must be a positive integer or None")

    s = 0.0
    warm_start = None
    floor_streak = 0
    records = []
    windows = []
    intervals = []

    while s < 1.0:
        if max_anchors is not None and len(records) >= max_anchors:
            windows.append((s, 1.0))
            break

        values, vectors, residuals = lowest_two(
            pathop.H(s), v0=warm_start, tol=tol
        )
        delta_lo = float(
            max(0.0, (values[1] - residuals[1]) - values[0])
        )
        records.append((s, float(values[0]), float(values[1]), delta_lo))
        remaining = 1.0 - s

        if K_gap_cert == 0.0:
            h_proposed = remaining
            h_floor_overridden = remaining
            h_probe_planned = remaining
            floor_override_applied = False
            floor_streak = 0
            step_mode = "constant_gap"
        else:
            h_proposed = eta * (delta_lo / K_gap_cert)
            if not np.isfinite(h_proposed):
                h_proposed = np.finfo(np.float64).max
            floor_override_applied = h_proposed < h_floor
            h_floor_overridden = (
                h_floor if floor_override_applied else h_proposed
            )
            if floor_override_applied:
                probe_initial = max(h_floor, _UNRESOLVED_PROBE_INITIAL)
                probe_cap = max(h_floor, _UNRESOLVED_PROBE_MAX)
                growth = 2.0 ** min(floor_streak, 30)
                h_probe_planned = min(probe_cap, probe_initial * growth)
                floor_streak += 1
                step_mode = "floor_mark_and_probe"
            else:
                h_probe_planned = h_proposed
                floor_streak = 0
                step_mode = "adaptive"

        endpoint_clip_applied = h_probe_planned > remaining
        if h_probe_planned >= remaining:
            s_end = 1.0
        else:
            s_end = s + h_probe_planned
            if s_end <= s:
                s_end = float(np.nextafter(s, 1.0))
        h_actual = s_end - s
        B_k = float(delta_lo - K_gap_cert * h_actual)
        is_conditionally_resolved = B_k > 0.0

        if is_conditionally_resolved:
            if step_mode == "constant_gap":
                resolution_status = "constant_gap_conditionally_resolved"
            elif floor_override_applied:
                resolution_status = "floor_override_conditionally_resolved"
            else:
                resolution_status = "conditionally_resolved"
            reason = ""
        else:
            if step_mode == "constant_gap":
                resolution_status = "constant_gap_unresolved"
                reason = "nonpositive_constant_gap_anchor_bound"
            elif floor_override_applied:
                resolution_status = "unresolved_floor_probe"
                reason = "floor_triggered_probe_has_nonpositive_actual_bound"
            else:
                resolution_status = "unresolved_nonpositive_actual_bound"
                reason = "nonpositive_actual_interval_bound"
            windows.append((s, s_end))

        intervals.append(
            {
                "s_start": float(s),
                "s_end": float(s_end),
                "delta_lo": delta_lo,
                "theta0": float(values[0]),
                "theta1": float(values[1]),
                "r0": float(residuals[0]),
                "r1": float(residuals[1]),
                "K_gap_cert": K_gap_cert,
                "B_k": B_k,
                "h_proposed": float(h_proposed),
                "h_floor_overridden": float(h_floor_overridden),
                "h_probe_planned": float(h_probe_planned),
                "h_actual": float(h_actual),
                "h_step": float(h_actual),
                "floor_override_applied": bool(floor_override_applied),
                "endpoint_clip_applied": bool(endpoint_clip_applied),
                "step_mode": step_mode,
                "is_conditionally_resolved": bool(is_conditionally_resolved),
                "resolution_status": resolution_status,
                "reason": reason,
                "residuals": [float(residuals[0]), float(residuals[1])],
            }
        )
        warm_start = vectors[:, 0]
        s = s_end

    return records, _merge_windows(windows), intervals


def certified_sweep(
    pathop,
    K_gap_cert,
    eta=DEFAULT_ETA,
    h_floor=DEFAULT_H_FLOOR,
    tol=DEFAULT_SOLVER_TOL,
    max_anchors=None,
):
    """Backward-compatible function name for :func:`conditional_sweep`."""
    return conditional_sweep(
        pathop,
        K_gap_cert,
        eta=eta,
        h_floor=h_floor,
        tol=tol,
        max_anchors=max_anchors,
    )


def uniform_grid_sweep(
    pathop, K_gap, delta_target, s_grid, tol=DEFAULT_SOLVER_TOL
):
    """Evaluate the fixed-grid workload using ``K_gap`` directly."""
    K_gap = _validate_K_gap(K_gap)
    delta_target = float(delta_target)
    if not np.isfinite(delta_target) or delta_target <= 0.0:
        raise ValueError("delta_target must be finite and positive")

    if K_gap == 0.0:
        s_anchors = np.asarray([0.0])
    else:
        n_uniform = int(np.ceil(K_gap / delta_target)) + 1
        s_anchors = np.linspace(0.0, 1.0, n_uniform)

    anchor_gaps = []
    warm_start = None
    for s in s_anchors:
        values, vectors, residuals = lowest_two(
            pathop.H(float(s)), v0=warm_start, tol=tol
        )
        anchor_gaps.append(
            float(max(0.0, (values[1] - residuals[1]) - values[0]))
        )
        warm_start = vectors[:, 0]

    s_grid = np.asarray(s_grid, dtype=float)
    envelope = np.full_like(s_grid, -np.inf)
    for s_anchor, gap_lo in zip(s_anchors, anchor_gaps):
        envelope = np.maximum(
            envelope, gap_lo - K_gap * np.abs(s_grid - s_anchor)
        )
    return (
        int(len(s_anchors)),
        float(np.mean(envelope >= delta_target)),
        float(np.mean(envelope > 0.0)),
        float(np.mean(envelope <= 0.0)),
    )


def hybrid_profile(records, K_gap, sgrid):
    """Return the conditional anchor envelope with slope ``K_gap``."""
    K_gap = _validate_K_gap(K_gap)
    sgrid = np.asarray(sgrid, dtype=float)
    output = np.full_like(sgrid, -np.inf)
    for s_anchor, _, _, gap_lo in records:
        output = np.maximum(
            output, gap_lo - K_gap * np.abs(sgrid - s_anchor)
        )
    return output


def heuristic_solve_integral(K_gap, eta, sampled_gaps, s_grid):
    """Evaluate (K_gap/eta) times the sampled integral of 1/gap."""
    K_gap = _validate_K_gap(K_gap)
    eta = float(eta)
    if not np.isfinite(eta) or not 0.0 < eta < 1.0:
        raise ValueError("eta must be finite and lie strictly between 0 and 1")
    sampled_gaps = np.asarray(sampled_gaps, dtype=float)
    s_grid = np.asarray(s_grid, dtype=float)
    if sampled_gaps.shape != s_grid.shape:
        raise ValueError("sampled_gaps and s_grid must have matching shapes")
    if np.any(~np.isfinite(sampled_gaps)) or np.any(sampled_gaps <= 0.0):
        raise ValueError("sampled_gaps must be finite and strictly positive")
    if K_gap == 0.0:
        return 0.0
    return float(
        (K_gap / eta) * np.trapezoid(1.0 / sampled_gaps, s_grid)
    )


# --------------------- [7.4] sampled dense reference -----------------------

def sampled_dense_gap_curve(N, c, sgrid):
    """Compute a dense floating-point sampled reference gap curve."""
    from maxcut_gap_benchmark import driver_matrix

    driver = driver_matrix(N + 1, sector=True).toarray()
    gaps = []
    for s in sgrid:
        values = np.linalg.eigvalsh(
            (1.0 - s) * driver + s * np.diag(c)
        )
        gaps.append(values[1] - values[0])
    return np.asarray(gaps)


def dense_gap_curve(N, c, sgrid):
    """Compatibility alias for :func:`sampled_dense_gap_curve`."""
    return sampled_dense_gap_curve(N, c, sgrid)


# --------------------------- output provenance -----------------------------

def _canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(filename):
    try:
        with open(os.path.join(_PROJECT_DIR, filename), "rb") as handle:
            return _sha256_bytes(handle.read())
    except OSError:
        return None


def _runtime_provenance():
    global _RUNTIME_PROVENANCE
    if _RUNTIME_PROVENANCE is not None:
        return _RUNTIME_PROVENANCE

    try:
        repository_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        repository_head = None

    _RUNTIME_PROVENANCE = {
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "repository": {
            "head": repository_head,
            "source_sha256": {
                "main.py": _source_sha256("main.py"),
                "maxcut_gap_benchmark.py": _source_sha256(
                    "maxcut_gap_benchmark.py"
                ),
            },
        },
    }
    return _RUNTIME_PROVENANCE


def _atomic_write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    with open(temporary_path, "wb") as handle:
        handle.write(payload)
    os.replace(temporary_path, path)


def _write_json(path, payload):
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _atomic_write_bytes(path, encoded)


def _graph_record(
    N,
    n_vertices,
    p,
    seed,
    attempts,
    rng_name,
    edges,
    Lambda_I,
    weight_bounds,
):
    edge_payload = [
        [int(u), int(v), float(weight)] for u, v, weight in edges
    ]
    W_model = weight_bounds["W_model"]
    W_exact = weight_bounds["W_exact"]
    W_upper = weight_bounds["W_upper"]
    W_upper_exact = weight_bounds["W_upper_exact"]
    K_gap_exact = weight_bounds["K_gap_exact"]
    K_gap_from_W_upper_exact = weight_bounds[
        "K_gap_from_W_upper_exact"
    ]
    K_gap_cert = weight_bounds["K_gap_cert"]
    K_gap_cert_exact = weight_bounds["K_gap_cert_exact"]
    edge_payload_sha256 = _sha256_bytes(
        _canonical_json_bytes(edge_payload)
    )
    record = {
        "schema_version": 2,
        "record_type": "generated_weighted_graph_instance",
        "row_key": {"N": int(N), "seed": int(seed)},
        "N_effective_qubits": int(N),
        "n_vertices": int(n_vertices),
        "num_edges": int(len(edges)),
        "graph_model": {
            "name": "Erdos-Renyi",
            "p": float(p),
            "p_hex": float(p).hex(),
            "edge_weight_distribution": "independent uniform [0.5, 1.5)",
            "acceptance_conditions": [
                "connected",
                "nondegenerate pinned optimum",
            ],
        },
        "generation": {
            "seed": int(seed),
            "rng_bit_generator": rng_name,
            "attempts": int(attempts),
        },
        "edge_list": edge_payload,
        "edge_weight_hex": [float(weight).hex() for _, _, weight in edges],
        "edge_weight_semantics": (
            "archived binary64 values interpreted as exact dyadic rationals"
        ),
        "edge_payload_sha256": edge_payload_sha256,
        "edge_weights_sum": W_model,
        "edge_weights_sum_hex": W_model.hex(),
        "edge_weights_sum_semantics": "compatibility alias of W_model",
        "W_model": W_model,
        "W_model_hex": W_model.hex(),
        "W_exact": _fraction_metadata(W_exact),
        "W_exact_numerator": W_exact.numerator,
        "W_exact_denominator": W_exact.denominator,
        "W_upper": W_upper,
        "W_upper_hex": W_upper.hex(),
        "W_upper_exact": _fraction_metadata(W_upper_exact),
        "W_upper_numerator": W_upper_exact.numerator,
        "W_upper_denominator": W_upper_exact.denominator,
        "Lambda_I": int(Lambda_I),
        "K_gap_exact": _fraction_metadata(K_gap_exact),
        "K_gap_exact_numerator": K_gap_exact.numerator,
        "K_gap_exact_denominator": K_gap_exact.denominator,
        "K_gap_from_W_upper_exact": _fraction_metadata(
            K_gap_from_W_upper_exact
        ),
        "K_gap_cert": K_gap_cert,
        "K_gap_cert_hex": K_gap_cert.hex(),
        "K_gap_cert_exact": _fraction_metadata(K_gap_cert_exact),
        "K_gap_cert_numerator": K_gap_cert_exact.numerator,
        "K_gap_cert_denominator": K_gap_cert_exact.denominator,
        "analytic_inequality": (
            "-Lambda_I I <= D <= W_exact I <= W_upper I and "
            "diameter(D) <= K_gap_cert"
        ),
        "rounding_policy": (
            "W_upper is the least binary64 >= W_exact; K_gap_cert is the "
            "least binary64 >= exact_binary64(W_upper)+Lambda_I"
        ),
        "provenance": _runtime_provenance(),
        "graph_record_hash_scope": (
            "SHA-256 of canonical JSON for every record field except "
            "graph_record_sha256"
        ),
    }
    record["graph_record_sha256"] = _sha256_bytes(
        _canonical_json_bytes(record)
    )
    return record


def _verify_graph_record(record):
    expected = record.get("graph_record_sha256")
    hash_input = {
        key: value
        for key, value in record.items()
        if key != "graph_record_sha256"
    }
    actual = _sha256_bytes(_canonical_json_bytes(hash_input))
    if expected != actual:
        raise ValueError("graph record hash mismatch")


# --------------------------- [7.5] experiment driver -----------------------

def _interval_statistics(intervals, windows, delta_target):
    coverage = float(sum(interval["h_actual"] for interval in intervals))
    unresolved_width = float(sum(end - start for start, end in windows))
    target_width = float(
        sum(
            interval["h_actual"]
            for interval in intervals
            if interval["is_conditionally_resolved"]
            and interval["B_k"] >= delta_target
        )
    )
    positive_width = float(
        sum(
            interval["h_actual"]
            for interval in intervals
            if interval["is_conditionally_resolved"]
        )
    )
    complete = abs(coverage - 1.0) <= 1e-12
    raw_global = (
        float(min(interval["B_k"] for interval in intervals))
        if complete and intervals
        else None
    )
    resolved_bounds = [
        interval["B_k"]
        for interval in intervals
        if interval["is_conditionally_resolved"]
    ]
    return {
        "interval_coverage": coverage,
        "target_fraction": target_width,
        "positive_fraction": positive_width,
        "unresolved_fraction": unresolved_width,
        "global_raw": raw_global,
        "global_nonnegative": (
            max(0.0, raw_global) if raw_global is not None else None
        ),
        "resolved_min": (
            float(min(resolved_bounds)) if resolved_bounds else None
        ),
        "global_resolved": bool(
            complete
            and intervals
            and all(
                interval["is_conditionally_resolved"]
                for interval in intervals
            )
        ),
    }


def _sampled_tightness(reference_gap, conditional_profile, s_grid, global_lb):
    lower_profile = np.maximum(0.0, conditional_profile)
    reference_index = int(np.argmin(reference_gap))
    ratios = lower_profile / reference_gap
    slack = reference_gap - lower_profile
    reference_min = float(reference_gap[reference_index])
    return {
        "sampled_reference_gap_min_est": reference_min,
        "sampled_reference_s_min_est": float(s_grid[reference_index]),
        "sampled_conditional_lb_at_reference_min": float(
            lower_profile[reference_index]
        ),
        "sampled_tightness_ratio_at_reference_min": float(
            ratios[reference_index]
        ),
        "sampled_tightness_ratio_mean": float(np.mean(ratios)),
        "sampled_tightness_ratio_min": float(np.min(ratios)),
        "sampled_tightness_ratio_max": float(np.max(ratios)),
        "sampled_additive_slack_mean": float(np.mean(slack)),
        "sampled_additive_slack_max": float(np.max(slack)),
        "global_lb_to_sampled_reference_min_ratio": float(
            global_lb / reference_min
        ),
    }


def run_instance(
    N,
    p,
    seed,
    delta_target=0.25,
    validate=None,
    eta=DEFAULT_ETA,
    h_floor=DEFAULT_H_FLOOR,
):
    """Run one deterministic graph instance and return the historical tuple."""
    _, eta, h_floor = _validate_sweep_inputs(0.0, eta, h_floor)
    delta_target = float(delta_target)
    if not np.isfinite(delta_target) or delta_target <= 0.0:
        raise ValueError("delta_target must be finite and positive")

    start_time = time.time()
    rng = np.random.default_rng(seed)
    rng_name = type(rng.bit_generator).__name__
    n_vertices = N + 1
    attempts = 0
    while True:
        attempts += 1
        edges = er_graph(n_vertices, p, rng)
        m = len(edges)
        costs = pinned_cost_vector(N, edges)
        two_lowest = np.partition(costs, 1)[:2]
        E0P, E1P = float(two_lowest[0]), float(two_lowest[1])
        if is_connected(n_vertices, edges) and E1P > E0P:
            break

    Lambda_I = int(2 * (n_vertices // 2))
    weight_bounds = _analytic_weight_bounds(edges, Lambda_I)
    W_model = weight_bounds["W_model"]
    W_exact = weight_bounds["W_exact"]
    W_upper = weight_bounds["W_upper"]
    W_upper_exact = weight_bounds["W_upper_exact"]
    K_gap_exact = weight_bounds["K_gap_exact"]
    K_gap_from_W_upper_exact = weight_bounds[
        "K_gap_from_W_upper_exact"
    ]
    K_gap_cert = weight_bounds["K_gap_cert"]
    K_gap_cert_exact = weight_bounds["K_gap_cert_exact"]
    _validate_K_gap(K_gap_cert)
    D_norm_bound = float(max(W_upper, Lambda_I))
    D_norm_exact = max(W_exact, Fraction(Lambda_I))
    if Fraction.from_float(D_norm_bound) < D_norm_exact:
        raise ArithmeticError("D_norm_bound is not an upward-safe bound")

    pathop = PathOperator(N, costs)
    pathop.matvecs = 0
    diameter_diagnostics = spectral_diameter_diagnostics(pathop)
    diagnostic_matvecs = int(pathop.matvecs)

    x_star_cost = E0P
    ceiling_oracle = [
        (0.0, W_upper / 2.0),
        (n_vertices / 2.0, x_star_cost),
    ]
    E0P_lower_bound = max(
        0.0, W_model - spectral_maxcut_ub(n_vertices, edges)
    )
    heuristic_value, _ = greedy_cut(n_vertices, edges, rng)
    ceiling_poly = [
        (0.0, W_upper / 2.0),
        (n_vertices / 2.0, float(heuristic_value)),
    ]
    E1P_poly = E0P_lower_bound

    s_grid = np.linspace(0.0, 1.0, 201)
    endpoint_bounds = {
        "weyl": np.asarray(
            [
                weyl_endpoint_cert(
                    s, 2.0, E1P - E0P, K_gap_cert
                )
                for s in s_grid
            ]
        ),
        "floor": np.asarray(
            [
                psd_floor_cert(s, 2.0, E1P, ceiling_oracle)
                for s in s_grid
            ]
        ),
        "floor_poly": np.asarray(
            [
                psd_floor_cert(s, 2.0, E1P_poly, ceiling_poly)
                for s in s_grid
            ]
        ),
    }

    pathop.matvecs = 0
    records, windows, intervals = conditional_sweep(
        pathop,
        K_gap_cert,
        eta=eta,
        h_floor=h_floor,
    )
    conditional_profile = hybrid_profile(records, K_gap_cert, s_grid)
    interval_stats = _interval_statistics(
        intervals, windows, delta_target
    )
    (
        n_uniform,
        uniform_target_fraction,
        uniform_positive_fraction,
        uniform_unresolved_fraction,
    ) = uniform_grid_sweep(
        pathop, K_gap_cert, delta_target, s_grid
    )

    anchor_gap_estimates = np.asarray(
        [theta1 - theta0 for _, theta0, theta1, _ in records]
    )
    anchor_min_index = int(np.argmin(anchor_gap_estimates))
    production_wall = time.time() - start_time
    diameter_estimate = diameter_diagnostics["spectral_diameter_est"]
    fraction_weyl = float(np.mean(endpoint_bounds["weyl"] > 0.0))
    fraction_floor = float(np.mean(endpoint_bounds["floor"] > 0.0))
    fraction_floor_poly = float(
        np.mean(endpoint_bounds["floor_poly"] > 0.0)
    )

    row = {
        "schema_version": 4,
        "N": int(N),
        "n_vertices": int(n_vertices),
        "m": int(m),
        "seed": int(seed),
        "connected": True,
        "W": W_upper,
        "W_semantics": "compatibility alias of W_upper",
        "W_model": W_model,
        "W_model_hex": W_model.hex(),
        "W_exact_numerator": W_exact.numerator,
        "W_exact_denominator": W_exact.denominator,
        "W_upper": W_upper,
        "W_upper_hex": W_upper.hex(),
        "W_upper_numerator": W_upper_exact.numerator,
        "W_upper_denominator": W_upper_exact.denominator,
        "Lambda_I": Lambda_I,
        "K_gap_exact_numerator": K_gap_exact.numerator,
        "K_gap_exact_denominator": K_gap_exact.denominator,
        "K_gap_from_W_upper_exact_numerator": (
            K_gap_from_W_upper_exact.numerator
        ),
        "K_gap_from_W_upper_exact_denominator": (
            K_gap_from_W_upper_exact.denominator
        ),
        "K_gap_cert": K_gap_cert,
        "K_gap_cert_hex": K_gap_cert.hex(),
        "K_gap_cert_numerator": K_gap_cert_exact.numerator,
        "K_gap_cert_denominator": K_gap_cert_exact.denominator,
        "K_gap_cert_derivation": (
            "least binary64 >= exact_binary64(W_upper) + Lambda_I"
        ),
        "D_norm_bound": D_norm_bound,
        "D_norm_exact_numerator": D_norm_exact.numerator,
        "D_norm_exact_denominator": D_norm_exact.denominator,
        "D_norm_bound_derivation": "max(W_upper, Lambda_I)",
        "D_lambda_min_est": diameter_diagnostics["lambda_min_est"],
        "D_lambda_max_est": diameter_diagnostics["lambda_max_est"],
        "D_spectral_diameter_est": diameter_estimate,
        "D_norm_est": diameter_diagnostics["operator_norm_est"],
        "K_gap_cert_over_diameter_est": (
            K_gap_cert / diameter_estimate
            if diameter_estimate > 0.0
            else None
        ),
        "diagnostic_matvecs": diagnostic_matvecs,
        "E0P_model": E0P,
        "E1P_model": E1P,
        "delta_target": delta_target,
        "eta": eta,
        "h_floor": h_floor,
        "unresolved_probe_strategy": "exponential_mark_and_probe",
        "unresolved_probe_initial": _UNRESOLVED_PROBE_INITIAL,
        "unresolved_probe_max": _UNRESOLVED_PROBE_MAX,
        "dmin_est": float(anchor_gap_estimates[anchor_min_index]),
        "s_star": float(records[anchor_min_index][0]),
        "anchor_ritz_gap_min_est": float(
            anchor_gap_estimates[anchor_min_index]
        ),
        "anchor_ritz_gap_s_min_est": float(records[anchor_min_index][0]),
        "frac_weyl": fraction_weyl,
        "frac_floor": fraction_floor,
        "frac_floor_poly": fraction_floor_poly,
        "weyl0_frac_pos": fraction_weyl,
        "psd_frac_pos": fraction_floor,
        "solves_hybrid": len(records),
        "conditional_oracle_calls": len(records),
        "weylA_oracle_calls": len(records),
        "solves_uniform": n_uniform,
        "uniform_solves": n_uniform,
        "target_conditionally_resolved_frac": interval_stats[
            "target_fraction"
        ],
        "positive_conditionally_resolved_frac": interval_stats[
            "positive_fraction"
        ],
        "unresolved_frac": interval_stats["unresolved_fraction"],
        "num_windows": len(windows),
        "total_window_width": interval_stats["unresolved_fraction"],
        "conditional_interval_coverage": interval_stats[
            "interval_coverage"
        ],
        "global_conditional_gap_lb_raw": interval_stats["global_raw"],
        "global_conditional_gap_lb": interval_stats[
            "global_nonnegative"
        ],
        "resolved_interval_conditional_gap_lb_min": interval_stats[
            "resolved_min"
        ],
        "global_conditionally_resolved": interval_stats[
            "global_resolved"
        ],
        "conditional_envelope_sampled_min_raw": float(
            np.min(conditional_profile)
        ),
        "conditional_envelope_sampled_min_nonnegative": float(
            max(0.0, np.min(conditional_profile))
        ),
        "uniform_sampled_target_frac": uniform_target_fraction,
        "uniform_sampled_positive_frac": uniform_positive_fraction,
        "uniform_sampled_unresolved_frac": uniform_unresolved_fraction,
        "uniform_target_frac": uniform_target_fraction,
        "uniform_positive_frac": uniform_positive_fraction,
        "uniform_uncert_frac": uniform_unresolved_fraction,
        "matvecs": int(pathop.matvecs),
        "dense_consistency_checked": False,
        "validated": False,
        "sampled_reference_gap_min_est": None,
        "sampled_reference_s_min_est": None,
        "sampled_conditional_lb_at_reference_min": None,
        "sampled_tightness_ratio_at_reference_min": None,
        "sampled_tightness_ratio_mean": None,
        "sampled_tightness_ratio_min": None,
        "sampled_tightness_ratio_max": None,
        "sampled_additive_slack_mean": None,
        "sampled_additive_slack_max": None,
        "global_lb_to_sampled_reference_min_ratio": None,
        "heuristic_integral_sampled_est": None,
        "N_heuristic": None,
        "wall": production_wall,
        "t_weylA": production_wall,
        "wall_validation": 0.0,
        "conditional_log_path": None,
    }

    graph_record = _graph_record(
        N,
        n_vertices,
        p,
        seed,
        attempts,
        rng_name,
        edges,
        Lambda_I,
        weight_bounds,
    )
    graph_relative_path = f"results/graph_N{N}_seed{seed}.json"
    _write_json(
        os.path.join(_PROJECT_DIR, graph_relative_path), graph_record
    )
    source_hashes = graph_record["provenance"]["repository"][
        "source_sha256"
    ]
    row.update(
        {
            "graph_schema_version": graph_record["schema_version"],
            "graph_model": graph_record["graph_model"]["name"],
            "graph_p": float(p),
            "graph_generation_seed": int(seed),
            "graph_generation_attempts": int(attempts),
            "rng_bit_generator": rng_name,
            "edge_payload_sha256": graph_record["edge_payload_sha256"],
            "graph_record_sha256": graph_record[
                "graph_record_sha256"
            ],
            "graph_record_path": graph_relative_path,
            "repository_head": graph_record["provenance"]["repository"][
                "head"
            ],
            "source_main_sha256": source_hashes["main.py"],
            "source_driver_sha256": source_hashes[
                "maxcut_gap_benchmark.py"
            ],
        }
    )

    if validate and N <= 12:
        validation_start = time.time()
        reference_gap = sampled_dense_gap_curve(N, costs, s_grid)
        for name, bound in endpoint_bounds.items():
            assert np.all(bound <= reference_gap + 1e-8), (
                f"endpoint bound {name} exceeds sampled reference"
            )
        assert np.all(
            conditional_profile <= reference_gap + 1e-6
        ), "conditional profile exceeds sampled reference"

        row["dense_consistency_checked"] = True
        row["validated"] = True
        tightness = _sampled_tightness(
            reference_gap,
            conditional_profile,
            s_grid,
            row["global_conditional_gap_lb"],
        )
        row.update(tightness)
        heuristic_integral = heuristic_solve_integral(
            K_gap_cert, eta, reference_gap, s_grid
        )
        row["heuristic_integral_sampled_est"] = heuristic_integral
        row["N_heuristic"] = heuristic_integral

        conditional_log_relative_path = (
            f"results/conditional_log_N{N}_seed{seed}.json"
        )
        conditional_log = {
            "schema_version": 4,
            "record_level": "Level-2 conditional numerical diagnostic",
            "symmetry_sector": "even-parity",
            "N_effective_qubits": int(N),
            "n_vertices": int(n_vertices),
            "num_edges": int(m),
            "graph_record_path": graph_relative_path,
            "graph_record_sha256": graph_record["graph_record_sha256"],
            "edge_payload_sha256": graph_record["edge_payload_sha256"],
            "edge_weight_semantics": (
                "archived binary64 values interpreted as exact dyadic rationals"
            ),
            "W_model": W_model,
            "W_model_hex": W_model.hex(),
            "W_exact": _fraction_metadata(W_exact),
            "W_exact_numerator": W_exact.numerator,
            "W_exact_denominator": W_exact.denominator,
            "W_upper": W_upper,
            "W_upper_hex": W_upper.hex(),
            "W_upper_exact": _fraction_metadata(W_upper_exact),
            "W_upper_numerator": W_upper_exact.numerator,
            "W_upper_denominator": W_upper_exact.denominator,
            "Lambda_I": Lambda_I,
            "K_gap_exact": _fraction_metadata(K_gap_exact),
            "K_gap_exact_numerator": K_gap_exact.numerator,
            "K_gap_exact_denominator": K_gap_exact.denominator,
            "K_gap_from_W_upper_exact": _fraction_metadata(
                K_gap_from_W_upper_exact
            ),
            "K_gap_cert": K_gap_cert,
            "K_gap_cert_hex": K_gap_cert.hex(),
            "K_gap_cert_exact": _fraction_metadata(K_gap_cert_exact),
            "K_gap_cert_numerator": K_gap_cert_exact.numerator,
            "K_gap_cert_denominator": K_gap_cert_exact.denominator,
            "K_gap_cert_derivation": (
                "-Lambda_I I <= D <= W_exact I <= W_upper I; "
                "K_gap_cert is the least binary64 not below exact_binary64("
                "W_upper)+Lambda_I"
            ),
            "D_norm_bound": D_norm_bound,
            "D_norm_exact": _fraction_metadata(D_norm_exact),
            "D_norm_bound_derivation": (
                "max(W_upper, Lambda_I); no further rounded arithmetic"
            ),
            "D_floating_point_diagnostics": diameter_diagnostics,
            "hamiltonian_path": (
                "H(s)=(1-s)H_I+sH_P in the even-parity sector"
            ),
            "eta": eta,
            "h_floor": h_floor,
            "unresolved_probe_strategy": {
                "name": "exponential_mark_and_probe",
                "initial": _UNRESOLVED_PROBE_INITIAL,
                "growth_factor": 2.0,
                "maximum": _UNRESOLVED_PROBE_MAX,
                "coverage_rule": (
                    "every skipped interval is unresolved unless B_k from "
                    "its actual length is positive"
                ),
            },
            "solver_tolerance": DEFAULT_SOLVER_TOL,
            "solver": {
                "library": "scipy.sparse.linalg.eigsh",
                "k": 2,
                "which": "SA",
                "ncv": "min(dimension, 50)",
                "maxiter": "SciPy default",
                "initial_vector": "deterministic local RNG vector",
                "warm_start": "previous ground-state Ritz vector",
                "all_calls_converged": True,
            },
            "eigenvalue_indexing_check": (
                "conditional; not independently verified"
            ),
            "global_conditional_gap_lb_raw": row[
                "global_conditional_gap_lb_raw"
            ],
            "global_conditional_gap_lb": row[
                "global_conditional_gap_lb"
            ],
            "resolved_interval_conditional_gap_lb_min": row[
                "resolved_interval_conditional_gap_lb_min"
            ],
            "conditionally_resolved_fraction": row[
                "positive_conditionally_resolved_frac"
            ],
            "unresolved_fraction": row["unresolved_frac"],
            "num_anchor_points": len(records),
            "sampled_reference_tightness": tightness,
            "heuristic_integral_sampled_est": heuristic_integral,
            "software": graph_record["provenance"]["software"],
            "repository": graph_record["provenance"]["repository"],
            "conditional_intervals": intervals,
            "unresolved_windows": [
                {
                    "s_start": float(start),
                    "s_end": float(end),
                    "reason": "one or more nonpositive actual interval bounds",
                }
                for start, end in windows
            ],
        }
        _write_json(
            os.path.join(_PROJECT_DIR, conditional_log_relative_path),
            conditional_log,
        )
        row["conditional_log_path"] = conditional_log_relative_path
        row["wall_validation"] = time.time() - validation_start

    return row, (s_grid, endpoint_bounds, conditional_profile, records)


def run_instance_parallel(args):
    N, seed = args
    row, _ = run_instance(
        N,
        p=0.5,
        seed=seed,
        validate=(N == 10 and seed == 0),
    )
    return row


def _write_graph_archive(rows):
    records = []
    for row in sorted(rows, key=lambda item: (item["N"], item["seed"])):
        path = os.path.join(_PROJECT_DIR, row["graph_record_path"])
        with open(path, "r", encoding="ascii") as handle:
            record = json.load(handle)
        _verify_graph_record(record)
        if record["graph_record_sha256"] != row["graph_record_sha256"]:
            raise ValueError("CSV row and graph record hashes disagree")
        records.append(record)

    payload = b"".join(
        _canonical_json_bytes(record) + b"\n" for record in records
    )
    relative_path = "results/graph_instances.jsonl"
    _atomic_write_bytes(os.path.join(_PROJECT_DIR, relative_path), payload)
    return {
        "path": relative_path,
        "sha256": _sha256_bytes(payload),
        "records": len(records),
    }


def _write_csv(rows, relative_path="section7_results.csv"):
    path = os.path.join(_PROJECT_DIR, relative_path)
    temporary_path = f"{path}.{os.getpid()}.tmp"
    with open(
        temporary_path, "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)
    with open(path, "rb") as handle:
        return _sha256_bytes(handle.read())


def _write_benchmark_outputs(rows):
    expected_keys = {
        (N, seed) for N in BENCHMARK_NS for seed in BENCHMARK_SEEDS
    }
    actual_keys = {(row["N"], row["seed"]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("benchmark output must contain exactly the 40 N=10/12 rows")

    rows.sort(key=lambda row: (row["N"], row["seed"]))
    archive = _write_graph_archive(rows)
    for row in rows:
        row["graph_archive_path"] = archive["path"]
        row["graph_archive_sha256"] = archive["sha256"]
        row["graph_archive_records"] = archive["records"]
    csv_sha256 = _write_csv(rows)

    manifest = {
        "schema_version": 1,
        "result_schema_version": rows[0]["schema_version"],
        "N_values": list(BENCHMARK_NS),
        "seeds": list(BENCHMARK_SEEDS),
        "rows": len(rows),
        "csv": {
            "path": "section7_results.csv",
            "sha256": csv_sha256,
        },
        "graph_archive": archive,
        "source_sha256": {
            "main.py": rows[0]["source_main_sha256"],
            "maxcut_gap_benchmark.py": rows[0][
                "source_driver_sha256"
            ],
        },
    }
    manifest_relative_path = "results/section7_run_manifest.json"
    _write_json(
        os.path.join(_PROJECT_DIR, manifest_relative_path), manifest
    )
    return archive, manifest_relative_path, csv_sha256


if __name__ == "__main__":
    import multiprocessing

    benchmark_rows = []
    for benchmark_N in BENCHMARK_NS:
        print(f"Running N={benchmark_N} with 20 seeds in parallel...")
        with multiprocessing.Pool(processes=5) as pool:
            group_rows = pool.map(
                run_instance_parallel,
                [(benchmark_N, seed) for seed in BENCHMARK_SEEDS],
            )
        benchmark_rows.extend(group_rows)
        for benchmark_row in group_rows:
            print(benchmark_row)

    graph_archive, manifest_path, result_csv_sha256 = (
        _write_benchmark_outputs(benchmark_rows)
    )
    print(
        f"Wrote {graph_archive['records']} graph records to "
        f"{graph_archive['path']} (sha256={graph_archive['sha256']})"
    )
    print(
        f"Wrote section7_results.csv (sha256={result_csv_sha256}) and "
        f"{manifest_path}"
    )
