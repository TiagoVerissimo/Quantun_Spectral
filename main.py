"""

Section 7 experiments: certified spectral-gap benchmarks for AQC Max-Cut paths.

  H(s) = (1-s) H_I + s H_P

  H_I  = sum_i (I - X_i)/2          PSD, spec {0..N}, binomial multiplicities

  H_P  = diag(c), c[x] = # uncut edges

Phases of the code generation

  [7.1] instance generation        [7.2] certificates (oracle + poly inputs)

  [7.3] Algorithm 1 hybrid sweep   [7.4] baselines + validation

  [7.5] experiment driver -> CSV / figures

"""

import csv
import hashlib
import json
import os
import platform
import subprocess
import time

import numpy as np


from scipy.sparse.linalg import LinearOperator, eigsh



# ----------------------------- [7.1] instances -----------------------------

def er_graph(n, p, rng):

    return [(i, j, rng.uniform(0.5, 1.5)) for i in range(n) for j in range(i + 1, n) if rng.random() < p]



def pinned_cost_vector(N, edges):

    """c[x] = uncut edges; qubit q <-> vertex q+1; bit=0 <-> spin +1 (= pinned v0)."""

    x = np.arange(1 << N, dtype=np.int64)

    c = np.zeros(1 << N, dtype=np.float64)

    for edge in edges:

        u, v = edge[0], edge[1]

        w = edge[2] if len(edge) > 2 else 1.0

        u, v = min(u, v), max(u, v)

        if u == 0:

            c += w * (((x >> (v - 1)) & 1) == 0)

        else:

            c += w * (((x >> (u - 1)) & 1) == ((x >> (v - 1)) & 1))

    return c



def uncut_of_assignment(edges, a):     # poly-time cost of one cut, a[0]=True fixed

    return sum(edge[2] * (a[edge[0]] == a[edge[1]]) if len(edge) > 2 else (a[edge[0]] == a[edge[1]]) for edge in edges)



def greedy_cut(n, edges, rng, restarts=30):

    """1-exchange local search; returns (uncut_value, assignment). Poly time."""

    adj = [[] for _ in range(n)]

    for edge in edges:

        u, v = edge[0], edge[1]

        w = edge[2] if len(edge) > 2 else 1.0

        adj[u].append((v, w))

        adj[v].append((u, w))

    best = None

    for _ in range(restarts):

        a = rng.integers(0, 2, n).astype(bool); a[0] = True

        moved = True

        while moved:

            moved = False

            for v in range(1, n):

                same_wt = sum(w for (u, w) in adj[v] if a[u] == a[v])

                diff_wt = sum(w for (u, w) in adj[v] if a[u] != a[v])

                if same_wt > diff_wt:

                    a[v] = ~a[v]

                    moved = True

        val = uncut_of_assignment(edges, a)

        if best is None or val < best[0]:

            best = (val, a.copy())

    return best



def spectral_maxcut_ub(n, edges):

    """maxcut <= n*lam_max(Laplacian)/4  ->  poly-time LOWER bound on E0(H_P)."""

    L = np.zeros((n, n))

    for edge in edges:

        u, v = edge[0], edge[1]

        w = edge[2] if len(edge) > 2 else 1.0

        L[u, u] += w; L[v, v] += w; L[u, v] -= w; L[v, u] -= w

    W = sum(edge[2] for edge in edges) if len(edges) > 0 and len(edges[0]) > 2 else float(len(edges))

    return min(W, n * np.linalg.eigvalsh(L)[-1] / 4.0)



# ------------------- matvec engine + Lanczos with enclosures ----------------

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

        H_s = (1.0 - s) * self.HI_sparse + s * self.HP_sparse

        def mv(v):

            self.matvecs += 1

            return H_s.dot(v)

        return LinearOperator((self.d, self.d), matvec=mv, dtype=np.float64)



    def D(self):

        D_op = self.HP_sparse - self.HI_sparse

        def mv(v):

            self.matvecs += 1

            return D_op.dot(v)

        return LinearOperator((self.d, self.d), matvec=mv, dtype=np.float64)



def lowest_two(Hop, v0=None, tol=1e-10):

    """Ritz pair (theta0, theta1) + residual norms.

    CAVEAT (state in paper): E1 >= theta1 - r1 assumes that theta1 has the

    correct eigenvalue index. The Ritz value theta0 is itself a Rayleigh--Ritz

    upper bound on E0, so max(0, theta1 - r1 - theta0) is the conditional

    Level-2 anchor lower bound used by the continuation sweep."""

    vals, vecs = eigsh(Hop, k=2, which='SA', v0=v0, tol=tol,

                       ncv=min(Hop.shape[0] - 1, 50))

    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]

    res = np.array([np.linalg.norm(Hop @ vecs[:, j] - vals[j] * vecs[:, j])

                    for j in range(2)])

    return vals, vecs, res



def path_lipschitz(pathop):

    D = pathop.D()

    hi = eigsh(D, k=1, which='LA', tol=1e-8, return_eigenvectors=False)[0]

    lo = eigsh(D, k=1, which='SA', tol=1e-8, return_eigenvectors=False)[0]

    return max(abs(hi), abs(lo))



# ------------------------- [7.2] the three certificates ---------------------

def weyl_endpoint_cert(s, gap0, gap1, L):                       # Prop 5.4

    return max(gap0 - 2 * s * L, gap1 - 2 * (1 - s) * L)



def psd_floor_cert(s, E1I, E1P, ceilings):                      # Prop 5.6 + dictionary

    """ceilings: list of (a,b) with E0(s) <= (1-s)a + s b from trial states."""

    return max((1 - s) * E1I, s * E1P) - min((1 - s) * a + s * b for a, b in ceilings)







# ---------------- [7.3] Algorithm 1: hybrid certified continuation ----------

def certified_sweep(pathop, L_cert, eta=0.9, h_floor=1e-6, tol=1e-10):

    s, v0 = 0.0, None

    records, windows = [], []

    step_details = []

    while s < 1.0 - 1e-12:

        vals, vecs, res = lowest_two(pathop.H(s), v0=v0, tol=tol)

        gap_lo = max(0.0, (vals[1] - res[1]) - vals[0]) # conditional Level-2 bound

        records.append((s, vals[0], vals[1], gap_lo))

        if gap_lo > 0:

            h = eta * gap_lo / (2.0 * L_cert)

        else:

            h = 0.0

        is_certified = True

        reason = ""

        if h < h_floor:

            h = h_floor

            is_certified = False

            reason = "step_below_floor"

            windows.append((s, min(1.0, s + h)))

        step_details.append({

            's_start': s,

            'h_step': h,

            'delta_lo': gap_lo,

            'theta0': float(vals[0]),

            'theta1': float(vals[1]),

            'r0': float(res[0]),

            'r1': float(res[1]),

            'is_certified': is_certified,

            'reason': reason,

            'residuals': [float(res[0]), float(res[1])]

        })

        v0, s = vecs[:, 0], min(1.0, s + h)             # warm start next solve

    # merge adjacent uncertified windows

    merged = []

    for w in windows:

        if merged and abs(w[0] - merged[-1][1]) < 1e-12:

            merged[-1] = (merged[-1][0], w[1])

        else:

            merged.append(w)

    # build formal certificates list

    certificates = []

    for detail in step_details:

        s_start = detail['s_start']

        h_step = detail['h_step']

        s_end = min(1.0, s_start + h_step)

        delta_lo = detail['delta_lo']

        B_k = delta_lo - 2.0 * L_cert * h_step

        certificates.append({

            's_start': float(s_start),

            's_end': float(s_end),

            'delta_lo': float(delta_lo),

            'theta0': detail['theta0'],

            'theta1': detail['theta1'],

            'r0': detail['r0'],

            'r1': detail['r1'],

            'L_cert': float(L_cert),

            'B_k': float(B_k),

            'is_certified': detail['is_certified'],

            'reason': detail['reason'],

            'residuals': detail['residuals']

        })

    return records, merged, certificates



def uniform_grid_sweep(pathop, L_cert, delta_target, s_grid, tol=1e-10):

    h_uniform = delta_target / (2.0 * L_cert)

    n_uniform = int(np.ceil(1.0 / h_uniform)) + 1

    s_anchors = np.linspace(0.0, 1.0, n_uniform)

    anchor_gaps = []

    v0 = None

    for s in s_anchors:

        vals, vecs, res = lowest_two(pathop.H(s), v0=v0, tol=tol)

        gap_lo = max(0.0, (vals[1] - res[1]) - vals[0])

        anchor_gaps.append(gap_lo)

        v0 = vecs[:, 0]

    env = np.full_like(s_grid, -np.inf)

    for s0, glo in zip(s_anchors, anchor_gaps):

        env = np.maximum(env, glo - 2.0 * L_cert * np.abs(s_grid - s0))

    target_frac = np.mean(env >= delta_target)

    pos_frac = np.mean(env > 0)

    uncert_frac = np.mean(env <= 0)

    return n_uniform, target_frac, pos_frac, uncert_frac



def hybrid_profile(records, L, sgrid):

    out = np.full_like(sgrid, -np.inf)

    for (si, _, _, glo) in records:

        out = np.maximum(out, glo - 2 * L * np.abs(sgrid - si))

    return out



# --------------------- [7.4] baselines + validation -------------------------

def dense_gap_curve(N, c, sgrid):

    from maxcut_gap_benchmark import driver_matrix

    HI = driver_matrix(N + 1, sector=True).toarray()

    gaps = []

    for s in sgrid:

        w = np.linalg.eigvalsh((1 - s) * HI + s * np.diag(c))

        gaps.append(w[1] - w[0])

    return np.array(gaps)



# --------------------------- [7.5] experiment driver ------------------------

def run_instance(N, p, seed, delta_target=0.25, validate=None):

    t_start = time.time()

    rng = np.random.default_rng(seed)

    n = N + 1

    while True:

        edges = er_graph(n, p, rng); m = len(edges)

        c = pinned_cost_vector(N, edges)

        two = np.partition(c, 1)[:2]                        # oracle endpoint data

        E0P, E1P = float(two[0]), float(two[1])

        if E1P > E0P:

            break

    x_star_cost = E0P                                   # exact optimum (oracle mode)

    pathop = PathOperator(N, c)

    L = path_lipschitz(pathop)

    W = sum(edge[2] for edge in edges) if len(edges) > 0 and len(edges[0]) > 2 else float(m)

    L_cert = float(W + n)                               # L_cert = W + n_v



    # ceilings: |+>^N gives (0, W/2); optimal basis state gives (n_v/2, E0P)

    ceil_oracle = [(0.0, W / 2.0), ((N + 1) / 2.0, x_star_cost)]

    # poly-input mode

    E0P_lb = max(0.0, W - spectral_maxcut_ub(n, edges))

    heur_val, _ = greedy_cut(n, edges, rng)

    ceil_poly = [(0.0, W / 2.0), ((N + 1) / 2.0, float(heur_val))]

    # Continuous weights have no certified unit spacing above E0(H_P).
    # E1(H_P) >= E0(H_P) >= E0P_lb is weaker but valid without an
    # integer-spectrum separation promise.
    E1P_poly = E0P_lb

    assert E1P > E0P, "degenerate optimum: resample or report separately"

    sgrid = np.linspace(0, 1, 201)

    certs = {

        'weyl':  np.array([weyl_endpoint_cert(s, 2.0, E1P - E0P, L_cert) for s in sgrid]),

        'floor': np.array([psd_floor_cert(s, 2.0, E1P, ceil_oracle) for s in sgrid]),

        'floor_poly': np.array([psd_floor_cert(s, 2.0, E1P_poly, ceil_poly) for s in sgrid]),


    }

    pathop.matvecs = 0

    records, windows, certificates = certified_sweep(pathop, L_cert)

    hybrid = hybrid_profile(records, L_cert, sgrid)

    w_target = 0.0

    w_pos = 0.0

    w_uncert = 0.0

    for i in range(len(records)):

        s_curr = records[i][0]

        gap_lo = records[i][3]

        if i < len(records) - 1:

            h = records[i+1][0] - s_curr

        else:

            h = 1.0 - s_curr

        is_uncert = any(w[0] - 1e-12 <= s_curr < w[1] - 1e-12 for w in windows)

        if is_uncert:

            w_uncert += h

        else:

            interval_floor = gap_lo - 2.0 * L_cert * h

            if interval_floor >= delta_target:

                w_target += h

            else:

                w_pos += h

    n_uniform, uni_target_frac, uni_pos_frac, uni_uncert_frac = uniform_grid_sweep(pathop, L_cert, delta_target, sgrid)

    gap_est = np.array([t1 - t0 for (_, t0, t1, _) in records])

    t_prod = time.time() - t_start

    frac_weyl = float(np.mean(certs['weyl'] > 0))

    frac_floor = float(np.mean(certs['floor'] > 0))

    frac_floor_poly = float(np.mean(certs['floor_poly'] > 0))


    row = dict(N=N, m=m, seed=seed, L_exact=L, L_cert=L_cert,

               dmin_est=float(gap_est.min()),

               s_star=float(records[int(np.argmin(gap_est))][0]),

               frac_weyl=frac_weyl,

               frac_floor=frac_floor,

               frac_floor_poly=frac_floor_poly,


               weyl0_frac_pos=frac_weyl,

               psd_frac_pos=frac_floor,


               solves_hybrid=len(records),

               weylA_oracle_calls=len(records),

               solves_uniform=n_uniform,

               uniform_solves=n_uniform,

               target_certified_frac=w_target,

               positive_certified_frac=w_target + w_pos,

               uncertified_frac=w_uncert,

               num_windows=len(windows),

               total_window_width=w_uncert,

               uniform_target_frac=uni_target_frac,

               uniform_positive_frac=uni_pos_frac,

               uniform_uncert_frac=uni_uncert_frac,

               matvecs=pathop.matvecs,

               validated=False,

               wall=t_prod,

               t_weylA=t_prod,

               wall_validation=0.0,

               N_heuristic=0.0)

    if validate and N <= 12:                            # referee-proofing asserts

        t_val_start = time.time()

        true_gap = dense_gap_curve(N, c, sgrid)

        for k, v in certs.items():

            assert np.all(v <= true_gap + 1e-8), f"certificate {k} VIOLATED"

        assert np.all(hybrid <= true_gap + 1e-6), "hybrid profile violated"

        row['validated'] = True

        # Compute cost model integral

        eta = 0.9

        N_heuristic = (2.0 * L_cert / eta) * float(np.trapz(1.0 / true_gap, sgrid))

        row['N_heuristic'] = N_heuristic

        # Save a self-identifying conditional Level-2 diagnostic record.
        edge_payload = [[int(u), int(v), float(w)] for (u, v, w) in edges]
        instance_bytes = json.dumps(
            edge_payload, separators=(',', ':'), ensure_ascii=True
        ).encode('ascii')
        try:
            repo_head = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                check=True
            ).stdout.strip()
            repo_dirty = bool(subprocess.run(
                ['git', 'status', '--porcelain'], capture_output=True, text=True,
                check=True
            ).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            repo_head, repo_dirty = None, None
        source_hashes = {}
        for source_name in ('main.py', 'maxcut_gap_benchmark.py'):
            try:
                with open(source_name, 'rb') as source_file:
                    source_hashes[source_name] = hashlib.sha256(
                        source_file.read()
                    ).hexdigest()
            except OSError:
                source_hashes[source_name] = None

        cert_log = {

            'schema_version': 2,

            'record_level': 'Level-2 conditional numerical diagnostic',

            'symmetry_sector': 'even-parity',

            'N_effective_qubits': int(N),

            'n_vertices': int(n),

            'num_edges': int(m),

            'graph_model': {'name': 'Erdos-Renyi', 'p': float(p), 'seed': int(seed)},

            'edge_list': edge_payload,

            'instance_sha256': hashlib.sha256(instance_bytes).hexdigest(),

            'edge_weights_sum': float(W),

            'L_cert': float(L_cert),

            'L_cert_derivation': 'sum(edge_weights) + n_vertices',

            'hamiltonian_path': 'H(s)=(1-s)H_I+sH_P in the even-parity sector',

            'eta': 0.9,

            'h_floor': 1e-6,

            'solver_tolerance': 1e-10,

            'solver': {
                'library': 'scipy.sparse.linalg.eigsh',
                'k': 2,
                'which': 'SA',
                'ncv': 'min(dimension-1, 50)',
                'maxiter': 'SciPy default',
                'warm_start': 'previous ground-state Ritz vector; none at s=0',
                'all_calls_converged': True
            },

            'eigenvalue_indexing_check': 'conditional; not independently verified',

            'minimum_certified_lower_bound': float(np.min([c['B_k'] for c in certificates if c['is_certified']]) if any(c['is_certified'] for c in certificates) else 0.0),

            'certified_fraction': float(w_target + w_pos),

            'unresolved_fraction': float(w_uncert),

            'num_anchor_points': int(len(records)),

            'N_heuristic': float(N_heuristic),

            'software': {
                'python': platform.python_version(),
                'numpy': np.__version__,
                'scipy': __import__('scipy').__version__
            },

            'repository': {
                'head': repo_head,
                'dirty_worktree': repo_dirty,
                'source_sha256': source_hashes
            },

            'certificates': certificates,

            'unresolved_windows': [

                {

                    's_start': float(w[0]),

                    's_end': float(w[1]),

                    'reason': 'step_below_floor'

                } for w in windows

            ]

        }

        os.makedirs('results', exist_ok=True)

        with open(f'results/certificate_log_N{N}_seed{seed}.json', 'w') as f:

            json.dump(cert_log, f, indent=2)

        row['wall_validation'] = time.time() - t_val_start

    return row, (sgrid, certs, hybrid, records)



def run_instance_parallel(args):

    N, seed = args

    row, _ = run_instance(N, p=0.5, seed=seed, validate=(N <= 10 and seed == 0))

    return row



if __name__ == "__main__":

    import multiprocessing

    rows = []

    first = True

    for N in [10, 12, 14]:                          # extend to 24 on a workstation

        print(f"Running N={N} with 20 seeds in parallel...")

        with multiprocessing.Pool(processes=5) as pool:

            results = pool.map(run_instance_parallel, [(N, seed) for seed in range(20)])



        for row in results:

            rows.append(row)

            print(row)



            with open("section7_results.csv", "a" if not first else "w", newline="") as f:

                w = csv.DictWriter(f, fieldnames=row.keys())

                if first:

                    w.writeheader()

                    first = False

                w.writerow(row)
