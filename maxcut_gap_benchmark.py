#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maxcut_gap_benchmark.py
=======================
Computational study for Section 7 of:
    "A Matrix Analysis Approach to Spectral Gap Estimation in
     Adiabatic Quantum Computing"

Compares, on random Max-Cut instances, the paper's certificates against the
exact instantaneous spectral gap Delta(s) of
    H(s) = (1-s) H_I + s H_P ,     s in [0,1],
with
    H_I = sum_i (I - X_i)/2                      (PSD driver, E0=0, gs |+...+>)
    H_P = sum_{(i,j) in E} w_ij (I + Z_i Z_j)/2  (PSD "anti-cut": ground state
                                                  = maximum cut, E0 = W - MaxCut)
Methods implemented
-------------------
Paper certificates (rigorous lower bounds):
  [W-Lip]   Weyl-Lipschitz propagation from certified anchors
            (Prop. 5.4 + Remark 5.5). Two anchors are FREE and exact:
            s=0 (driver spectrum analytic) and s=1 (H_P diagonal).
            An *adaptive* variant inserts extra Lanczos-certified anchors at
            the argmin of the current envelope until the whole path is
            certified positive (or a budget is reached).
  [PSD]     Componentwise-floor + Rayleigh-Ritz ceiling certificate
            (Prop. 5.6): needs only E1(H_I), E1(H_P), E0(H_I), <psi_I|H_P|psi_I>.
  [Horn]    T^n_1 Horn spectral-profile certificate (Prop. 6.4), using the
            full endpoint spectra (analytic for H_I, diagonal for H_P).

Literature comparators (estimates, not certificates):
  [Krylov]  m-step Lanczos/Krylov subspace ("classical QKSD emulation",
            cf. Parrish-McMahon; Stair et al.; Cortes-Gray PRA 105, 022417):
            Ritz-value gap theta_1 - theta_0 on each grid point.
  [2-level] Diabatic two-level (perturbative-crossing / anti-crossing) model
            spanned by the endpoint ground states; also returns the predicted
            crossing location s_x. The location metric mirrors the aim of
            Werner-Garcia-Saez-Estarellas (PRR 5, 043236 (2023)), who bound
            the LOCATION of the minimal gap via graph quantities.
  [Exact]   Grid exact diagonalization / sparse Lanczos = ground truth
            (standard practice in the annealing literature).

Symmetry convention
-------------------
Both endpoint Hamiltonians commute with the global flip P = prod_i X_i, and
every cut is doubly degenerate, so the full-space gap closes trivially at s=1.
By default all quantities (exact gap AND certificates) are computed in the
EVEN-parity sector containing |+>^N, of dimension 2^(N-1), where:
  spec(H_I)|_even = { k even : multiplicity C(N,k) },
  spec(H_P)|_even = { anti-cut(x) : x a representative of a pair {x, ~x} }.
Use --full-space to disable the reduction.

Outputs
-------
  outdir/results.csv           one row per instance x method with metrics
  outdir/instance_<id>.png     gap curve + all bounds/estimates
  outdir/summary_scatter.png   certified/estimated min-gap vs true min-gap
  outdir/summary_runtime.png   mean wall-time per method
  outdir/scaling.png           (with --scan-N) runtime scaling

Dependencies: numpy, scipy, matplotlib. (networkx optional: regular graphs.)
"""

import argparse
import csv
import math
import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DENSE_CUTOFF = 600          # below this dimension, use dense eigensolvers
EIGSH_TOL = 1e-10

# ----------------------------------------------------------------------
# 1. Instance generation (Max-Cut)
# ----------------------------------------------------------------------
def _connected(N, edges):
    """Union-find connectivity check."""
    parent = list(range(N))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (i, j, _w) in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(v) for v in range(N)}) == 1

def random_maxcut_instance(N, kind="erdos", p=0.5, weighted=True, rng=None):
    """
    Return list of edges (i, j, w). Weighted (generic) instances have a
    unique max cut almost surely => nondegenerate sector gap at s=1.
    Resamples until connected (disconnected graphs carry extra Z2 symmetries).
    """
    rng = rng or np.random.default_rng()
    for _ in range(2000):
        edges = []
        if kind == "erdos":
            for i in range(N):
                for j in range(i + 1, N):
                    if rng.random() < p:
                        edges.append([i, j])
        elif kind == "ring+chords":
            edges = [[i, (i + 1) % N] for i in range(N)]
            for _ in range(N // 2):
                i, j = rng.choice(N, size=2, replace=False)
                if [min(i, j), max(i, j)] not in [e[:2] for e in edges]:
                    edges.append([min(i, j), max(i, j)])
        elif kind == "regular":
            try:
                import networkx as nx
                G = nx.random_regular_graph(3, N, seed=int(rng.integers(2**31)))
                edges = [[min(u, v), max(u, v)] for u, v in G.edges()]
            except ImportError:
                raise SystemExit("kind='regular' requires networkx")
        elif kind == "complete":
            edges = [[i, j] for i in range(N) for j in range(i + 1, N)]
        else:
            raise ValueError(f"unknown graph kind {kind!r}")
        if not edges:
            continue
        if weighted:
            w = rng.uniform(0.5, 1.5, size=len(edges))
        else:
            w = np.ones(len(edges))
        full = [(i, j, float(wij)) for (i, j), wij in zip(edges, w)]
        if _connected(N, full):
            return full
    raise RuntimeError("could not generate a connected instance")

# ----------------------------------------------------------------------
# 2. Hamiltonian construction (even-parity sector by default)
# ----------------------------------------------------------------------
def anticut_diagonal(N, edges, sector=True):
    """
    Diagonal of H_P = sum w_ij (I + Z_i Z_j)/2 in the computational basis:
    value(x) = total weight of UNCUT edges = W - cut(x).
    sector=True: only pair representatives x in [0, 2^(N-1)).
    """
    dim = 1 << (N - 1) if sector else 1 << N
    x = np.arange(dim, dtype=np.int64)
    vals = np.zeros(dim)
    for (i, j, w) in edges:
        same = 1 - ((x >> i ^ x >> j) & 1)     # 1 if bits i, j agree (uncut)
        vals += w * same
    return vals

def driver_matrix(N, sector=True):
    """
    Sparse H_I = (N/2) I - (1/2) sum_i X_i, restricted to the even-parity
    sector via representatives x in [0, 2^(N-1)) with rep(y) = min(y, ~y).
    """
    if sector:
        dim = 1 << (N - 1)
        mask = (1 << N) - 1
        x = np.arange(dim, dtype=np.int64)
        rows, cols, data = [], [], []
        for i in range(N):
            y = x ^ (1 << i)
            rep = np.minimum(y, mask ^ y)      # complement identification
            rows.append(rep)
            cols.append(x)
            data.append(np.full(dim, -0.5))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        H = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
        H = H + sp.identity(dim, format="csr") * (N / 2.0)
        return H
    dim = 1 << N
    x = np.arange(dim, dtype=np.int64)
    rows, cols, data = [], [], []
    for i in range(N):
        y = x ^ (1 << i)
        rows.append(y)
        cols.append(x)
        data.append(np.full(dim, -0.5))
    H = sp.coo_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(dim, dim)).tocsr()
    return H + sp.identity(dim, format="csr") * (N / 2.0)

def driver_spectrum(N, sector=True):
    """Analytic spectrum of H_I: level k (k excitations), multiplicity C(N,k);
    even sector keeps even k only."""
    ks = range(0, N + 1, 2) if sector else range(0, N + 1)
    return np.concatenate([np.full(math.comb(N, k), float(k)) for k in ks])

def path_hamiltonian(s, HI, DP):
    return ((1.0 - s) * HI + s * sp.diags(DP)).tocsr()

# ----------------------------------------------------------------------
# 3. Ground truth: exact gap curve (dense or warm-started Lanczos)
# ----------------------------------------------------------------------
def lowest_eigs(H, k=6, v0=None):
    """Return (sorted lowest-k eigenvalues, ground vector or None)."""
    n = H.shape[0]
    if n <= DENSE_CUTOFF:
        w = np.linalg.eigvalsh(H.toarray())
        return w[:min(k, n)], None
    k = min(k, n - 2)
    w, V = spla.eigsh(H, k=k, which="SA", v0=v0, tol=EIGSH_TOL, maxiter=10000)
    idx = np.argsort(w)
    return w[idx], V[:, idx[0]]

def exact_gap_curve(s_grid, HI, DP):
    E0, E1 = np.zeros_like(s_grid), np.zeros_like(s_grid)
    v0 = None
    for a, s in enumerate(s_grid):
        w, v0 = lowest_eigs(path_hamiltonian(s, HI, DP), k=6, v0=v0)
        E0[a], E1[a] = w[0], w[1]
    return E0, E1

# ----------------------------------------------------------------------
# 4. Paper certificate 1: Weyl-Lipschitz propagation (Prop. 5.4 / Rem. 5.5)
# ----------------------------------------------------------------------
def weyl_lipschitz_constant(HI, DP, exact=True):
    """
    L = || H_P - H_I ||_2 (in the working space).
    exact=False returns the fully analytic bound ||H_P|| + ||H_I||
    """
    if not exact:
        return float(np.max(DP) + np.max(driver_like_norm(HI)))
    M = (sp.diags(DP) - HI).tocsr()
    n = M.shape[0]
    if n <= DENSE_CUTOFF:
        w = np.linalg.eigvalsh(M.toarray())
        return float(max(abs(w[0]), abs(w[-1])))
    hi = spla.eigsh(M, k=1, which="LA", return_eigenvectors=False,
                    tol=1e-8, maxiter=10000)[0]
    lo = spla.eigsh(M, k=1, which="SA", return_eigenvectors=False,
                    tol=1e-8, maxiter=10000)[0]
    return float(max(abs(hi), abs(lo)))

def driver_like_norm(HI):
    n = HI.shape[0]
    N = int(round(math.log2(n))) + 1
    return np.array([N if N % 2 == 0 else N - 1], dtype=float)

def weyl_envelope(s_grid, anchors, L):
    """Eq. (9): Delta(s) >= max_over_anchors { Delta(s0) - 2 L |s - s0| }."""
    env = np.full_like(s_grid, -np.inf)
    for (s0, g0) in anchors:
        env = np.maximum(env, g0 - 2.0 * L * np.abs(s_grid - s0))
    return env

def adaptive_weyl(s_grid, anchors, L, gap_oracle, max_anchors=25, min_gap_tol=1e-9):
    """
    Insert Lanczos-certified anchors at the argmin of the current envelope
    until the envelope is positive everywhere or the budget is exhausted.
    """
    anchors = list(anchors)
    calls = 0
    while True:
        env = weyl_envelope(s_grid, anchors, L)
        j = int(np.argmin(env))
        if env[j] > 0:
            return env, anchors, True, calls
        if len(anchors) >= max_anchors:
            return env, anchors, False, calls
        s_new = float(s_grid[j])
        if any(abs(s_new - s0) < 1e-12 for s0, _ in anchors):
            return env, anchors, False, calls        
        g_new = gap_oracle(s_new)
        calls += 1
        if g_new <= min_gap_tol:
            return env, anchors, False, calls        
        anchors.append((s_new, g_new))

# ----------------------------------------------------------------------
# 5. Paper certificate 2: PSD componentwise floor (Prop. 5.6)
# ----------------------------------------------------------------------
def psd_floor_curve(s_grid, E1_I, E1_P, E0_I, mean_P):
    """
    Eq. (62):
    Delta(s) >= max{(1-s) E1(H_I), s E1(H_P)} - (1-s) E0(H_I) - s <psi_I|H_P|psi_I>.
    """
    s = s_grid
    return (np.maximum((1 - s) * E1_I, s * E1_P) - (1 - s) * E0_I - s * mean_P)

# ----------------------------------------------------------------------
# 6. Paper certificate 3: T^n_1 Horn spectral-profile bound (Prop. 6.4)
# ----------------------------------------------------------------------
def horn_t1_curve(s_grid, specI, specP):
    """
    Eq. (18) with A = (1-s) H_I, B = s H_P and spectra in NON-INCREASING order:
       Delta >= max{a_n + b_{n-1}, a_{n-1} + b_n} - min_i {a_i + b_{n+1-i}}.
    """
    aI = np.sort(specI)[::-1].astype(float)
    aP = np.sort(specP)[::-1].astype(float)
    aP_rev = aP[::-1]                       
    out = np.empty_like(s_grid)
    for a, s in enumerate(s_grid):
        A_last, A_last2 = (1 - s) * aI[-1], (1 - s) * aI[-2]
        B_last, B_last2 = s * aP[-1], s * aP[-2]
        L1 = max(A_last + B_last2, A_last2 + B_last)
        U0 = np.min((1 - s) * aI + s * aP_rev)
        out[a] = L1 - U0
    return out

# ----------------------------------------------------------------------
# 7. Literature comparator A: truncated Krylov (classical QKSD emulation)
# ----------------------------------------------------------------------
def lanczos_ritz(H, v0, m):
    """m-step Lanczos with full reorthogonalization; returns Ritz values."""
    n = H.shape[0]
    Q = np.zeros((m, n))
    alph, beta = [], []
    q = v0 / np.linalg.norm(v0)
    Q[0] = q
    q_prev, b = np.zeros(n), 0.0
    for j in range(m):
        w = H @ q - b * q_prev
        a = float(q @ w)
        w -= a * q
        w -= Q[:j + 1].T @ (Q[:j + 1] @ w)   
        alph.append(a)
        b = float(np.linalg.norm(w))
        if b < 1e-13 or j == m - 1:
            break
        beta.append(b)
        q_prev, q = q, w / b
        Q[j + 1] = q
    T = np.diag(alph)
    if beta:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    return np.linalg.eigvalsh(T)

def krylov_gap_curve(s_grid, HI, DP, m=40, rng=None):
    """
    Ritz gap theta_1 - theta_0 from the Krylov space K_m(H(s), |+>).
    """
    rng = rng or np.random.default_rng(0)
    n = HI.shape[0]
    v0 = np.full(n, 1.0 / math.sqrt(n))
    v0 = v0 + 1e-8 * rng.standard_normal(n)
    gaps = np.empty_like(s_grid)
    for a, s in enumerate(s_grid):
        th = lanczos_ritz(path_hamiltonian(s, HI, DP), v0, m)
        gaps[a] = th[1] - th[0] if len(th) > 1 else np.nan
    return gaps

# ----------------------------------------------------------------------
# 8. Literature comparator B: two-level diabatic-crossing model
# ----------------------------------------------------------------------
def two_level_model(s_grid, N, HI, DP, W):
    """
    Effective 2x2 generalized eigenproblem in span{ |u> = |+>^N , |b> }.
    """
    n = HI.shape[0]
    E0P = float(np.min(DP))
    ground = np.flatnonzero(np.isclose(DP, E0P, atol=1e-12))
    g = len(ground)
    b_vec = np.zeros(n)
    b_vec[ground] = 1.0 / math.sqrt(g)
    Hbb_I = float(b_vec @ (HI @ b_vec))
    sigma = math.sqrt(g / n)                       
    Hub_P = sigma * E0P                            
    S = np.array([[1.0, sigma], [sigma, 1.0]])
    ew, ev = np.linalg.eigh(S)
    S_isqrt = ev @ np.diag(ew ** -0.5) @ ev.T
    gaps = np.empty_like(s_grid)
    for a, s in enumerate(s_grid):
        M = np.array([[s * W / 2.0,                    s * Hub_P],
                      [s * Hub_P, (1 - s) * Hbb_I + s * E0P]])
        w = np.linalg.eigvalsh(S_isqrt @ M @ S_isqrt)
        gaps[a] = w[1] - w[0]
    denom = Hbb_I + W / 2.0 - E0P
    s_x = Hbb_I / denom if denom > 0 else np.nan
    return gaps, s_x

# ----------------------------------------------------------------------
# 9. Benchmark driver
# ----------------------------------------------------------------------
def run_instance(inst_id, N, edges, s_grid, sector=True, krylov_m=40,
                 max_anchors=25, rng=None, outdir=None, make_plot=True):
    rng = rng or np.random.default_rng()
    res = {"instance": inst_id, "N": N, "edges": len(edges),
           "sector": "even" if sector else "full"}
    W = sum(w for _, _, w in edges)
    DP = anticut_diagonal(N, edges, sector)
    HI = driver_matrix(N, sector)
    specI = driver_spectrum(N, sector)
    dim = HI.shape[0]
    res["dim"] = dim
    res["maxcut"] = W - float(np.min(DP))
    
    # ---------- Ground Truth ----------
    t0 = time.perf_counter()
    E0, E1 = exact_gap_curve(s_grid, HI, DP)
    gap = E1 - E0
    t_exact = time.perf_counter() - t0
    j_min = int(np.argmin(gap))
    res.update(true_dmin=float(gap[j_min]), true_smin=float(s_grid[j_min]), t_exact=t_exact)
    
    DP_sorted = np.sort(DP)
    gap_s1 = float(DP_sorted[1] - DP_sorted[0])   
    E1_I = float(np.sort(specI)[1])               
    gap_s0 = E1_I                                 
    
    # ---------- [W-Lip] Weyl-Lipschitz ----------
    t0 = time.perf_counter()
    L = weyl_lipschitz_constant(HI, DP, exact=True)
    endpoint_anchors = [(0.0, gap_s0), (1.0, gap_s1)]
    env0 = weyl_envelope(s_grid, endpoint_anchors, L)   
    t_weyl0 = time.perf_counter() - t0
    
    def oracle(s):
        w, _ = lowest_eigs(path_hamiltonian(s, HI, DP), k=6)
        return float(w[1] - w[0])
        
    t0 = time.perf_counter()
    envA, anchors, certified, n_calls = adaptive_weyl(
        s_grid, endpoint_anchors, L, oracle, max_anchors=max_anchors)
    t_weylA = time.perf_counter() - t0 + t_weyl0
    res.update(weyl_L=L, weyl0_min=float(env0.min()), weyl0_frac_pos=float(np.mean(env0 > 0)), t_weyl0=t_weyl0,
               weylA_min=float(envA.min()), weylA_certified=bool(certified), weylA_anchors=len(anchors),
               weylA_oracle_calls=n_calls, t_weylA=t_weylA)
    
    # ---------- [PSD] Prop. 5.6 ----------
    t0 = time.perf_counter()
    psd = psd_floor_curve(s_grid, E1_I=E1_I, E1_P=float(DP_sorted[1]), E0_I=0.0, mean_P=W / 2.0)
    t_psd = time.perf_counter() - t0
    res.update(psd_min=float(psd.min()), psd_frac_pos=float(np.mean(psd > 0)), t_psd=t_psd)
    
    # ---------- [Horn] Prop. 6.4 ----------
    t0 = time.perf_counter()
    horn = horn_t1_curve(s_grid, specI, DP)
    t_horn = time.perf_counter() - t0
    res.update(horn_min=float(horn.min()), horn_frac_pos=float(np.mean(horn > 0)), t_horn=t_horn)
    
    # ---------- [Krylov] ----------
    t0 = time.perf_counter()
    kry = krylov_gap_curve(s_grid, HI, DP, m=krylov_m, rng=rng)
    t_kry = time.perf_counter() - t0
    jk = int(np.nanargmin(kry))
    res.update(krylov_dmin=float(kry[jk]), krylov_smin=float(s_grid[jk]),
               krylov_rel_err=float(abs(kry[jk] - gap[j_min]) / max(gap[j_min], 1e-15)),
               krylov_loc_err=float(abs(s_grid[jk] - s_grid[j_min])), t_krylov=t_kry)
    
    # ---------- [2-level] ----------
    t0 = time.perf_counter()
    tl, s_x = two_level_model(s_grid, N, HI, DP, W)
    t_tl = time.perf_counter() - t0
    jt = int(np.argmin(tl))
    res.update(twolvl_dmin=float(tl[jt]), twolvl_smin=float(s_grid[jt]), twolvl_scross=float(s_x),
               twolvl_rel_err=float(abs(tl[jt] - gap[j_min]) / max(gap[j_min], 1e-15)),
               twolvl_loc_err=float(abs(s_grid[jt] - s_grid[j_min])), t_twolvl=t_tl)
    
    # ---------- Visual Generation ----------
    if make_plot and outdir:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.plot(s_grid, gap, "k-", lw=2, label=r"exact $\Delta(s)$")
        ax.plot(s_grid, envA, "-", label=f"Weyl adaptive ({len(anchors)} anchors)")
        ax.plot(s_grid, env0, "--", label="Weyl endpoints only")
        ax.plot(s_grid, psd, "-.", label="PSD floor (Prop. 5.6)")
        ax.plot(s_grid, horn, ":", label=r"$T_1^n$ Horn (Prop. 6.4)")
        ax.plot(s_grid, kry, ".", ms=3, alpha=0.6, label=f"Krylov m={krylov_m}")
        ax.plot(s_grid, tl, "-", alpha=0.6, label="two-level model")
        for (s0, g0) in anchors:
            ax.plot([s0], [g0], "rv", ms=7)
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_xlabel("s"); ax.set_ylabel("gap / bound")
        ax.set_title(f"instance {inst_id}: N={N}, |E|={len(edges)}, dim={dim}")
        ax.set_ylim(-0.2, 1.1 * gap.max())
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"instance_{inst_id}.png"), dpi=160)
        plt.close(fig)
    return res

def summarize(rows, outdir):
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)
            
    fig, ax = plt.subplots(figsize=(6.5, 6))
    truth = np.array([r["true_dmin"] for r in rows])
    for key, lab, mk in [("weylA_min", "Weyl adaptive", "o"), ("psd_min", "PSD floor", "s"),
                         ("horn_min", "Horn $T_1^n$", "d"), ("krylov_dmin", "Krylov", "x"),
                         ("twolvl_dmin", "two-level", "+")]:
        vals = np.array([r[key] for r in rows])
        ax.plot(truth, vals, mk, label=lab, alpha=0.75)
    lim = [0, 1.15 * truth.max()]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"true $\Delta_{\min}$")
    ax.set_ylabel("certified bound / estimate (min over s)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary_scatter.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    tkeys = [("t_exact", "exact grid"), ("t_weylA", "Weyl adaptive"), ("t_weyl0", "Weyl endpoints"),
             ("t_psd", "PSD floor"), ("t_horn", "Horn"), ("t_krylov", "Krylov"), ("t_twolvl", "two-level")]
    means = [np.mean([r[k] for r in rows]) for k, _ in tkeys]
    ax.bar([lab for _, lab in tkeys], means)
    ax.set_yscale("log"); ax.set_ylabel("mean wall time [s]")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary_runtime.png"), dpi=160)
    plt.close(fig)

# ----------------------------------------------------------------------
# 10. Self-test on the paper's worked examples (Sections 5.3, 5.4, 6.3)
# ----------------------------------------------------------------------
def selftest():
    s = np.linspace(0, 1, 1001)
    weyl = 2.0 - 2.0 * math.sqrt(2.0) * s
    assert np.isclose(weyl[500], 2 - math.sqrt(2), atol=1e-9)       
    horn = horn_t1_curve(s, np.array([1.0, -1.0]), np.array([1.0, -1.0]))
    assert np.allclose(horn, 2.0 * np.abs(1.0 - 2.0 * s), atol=1e-12)
    psd = psd_floor_curve(s, E1_I=2.0, E1_P=2.0, E0_I=0.0, mean_P=1.0)
    assert np.isclose(psd[500], 0.5, atol=1e-12)
    
    edges = [(0, 1, 1.0)]
    DP = anticut_diagonal(2, edges, sector=True)
    HI = driver_matrix(2, sector=True)
    E0, E1 = exact_gap_curve(s, HI, DP)
    horn2 = horn_t1_curve(s, driver_spectrum(2, True), DP)
    assert np.all(horn2 <= (E1 - E0) + 1e-9)
    print("self-test passed: reproduces Secs. 5.3 / 5.4 / 6.3 and sanity checks.")

# ----------------------------------------------------------------------
# 11. CLI Execution Entrypoint
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=10, help="number of qubits")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--graph", default="erdos", choices=["erdos", "regular", "ring+chords", "complete"])
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--unweighted", action="store_true")
    ap.add_argument("--grid", type=int, default=201)
    ap.add_argument("--full-space", action="store_true")
    ap.add_argument("--krylov-m", type=int, default=40)
    ap.add_argument("--max-anchors", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--scan-N", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    os.makedirs(args.outdir, exist_ok=True)
    master = np.random.default_rng(args.seed)
    s_grid = np.linspace(0.0, 1.0, args.grid)
    sector = not args.full_space

    if args.scan_N:
        Ns = [int(x) for x in args.scan_N.split(",")]
        rows = []
        for N in Ns:
            rng = np.random.default_rng(master.integers(2**63))
            edges = random_maxcut_instance(N, args.graph, args.p, not args.unweighted, rng)
            print(f"[scan] N={N} ...")
            rows.append(run_instance(f"N{N}", N, edges, s_grid, sector, args.krylov_m, args.max_anchors, rng, args.outdir))
        summarize(rows, args.outdir)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        for k, lab in [("t_exact", "exact grid"), ("t_weylA", "Weyl adaptive"), ("t_horn", "Horn"), ("t_krylov", "Krylov")]:
            ax.semilogy(Ns, [r[k] for r in rows], "o-", label=lab)
        ax.set_xlabel("N"); ax.set_ylabel("wall time [s]"); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "scaling.png"), dpi=160)
        return

    rows = []
    for i in range(args.instances):
        rng = np.random.default_rng(master.integers(2**63))
        edges = random_maxcut_instance(args.N, args.graph, args.p, not args.unweighted, rng)
        print(f"instance {i}: N={args.N}, |E|={len(edges)} ...")
        r = run_instance(i, args.N, edges, s_grid, sector, args.krylov_m, args.max_anchors, rng, args.outdir)
        rows.append(r)
    summarize(rows, args.outdir)
    print(f"\nwrote {os.path.join(args.outdir, 'results.csv')} and plots.")

if __name__ == "__main__":
    main()
