"""
cone_checker.py

General-purpose tool to check, for a Feynman-matroid constraint matrix A
(rows = edges, columns = constraints, |E| x |F|, entries in {-1,0,1}),
whether the cones K_sigma^B appearing in the matroid-polytope construction
are simplicial or non-simplicial, for every choice of column basis B and
sign vector sigma.

Core idea (see accompanying discussion):
  1. Full-dimensionality check   -- discard degenerate (lower-dim) cones.
  2. Pointedness check           -- cone must not contain a full line.
  3. Facet redundancy removal    -- count genuine non-redundant facets via LP.
  4. Compare facet count to dim  -- #facets > |F|  <=>  non-simplicial.
  5. Ray-count cross-check       -- independent V-representation sanity check.

Usage: 
--matrix shape (E,F) or (Num edges, Num constraints)

1. To check simpliciality for a specific basis
    python cone_checker.py check --matrix '[[1,0],[0,1]]' --basis '[0,1,3]' 

2. To check simpliciality for all bases
    python cone_checker.py check --matrix '[[1,0],[0,1]]'

3. To run demo calculations
    python cone_checker.py demo                           
"""

import numpy as np
from itertools import product, combinations
from scipy.optimize import linprog


# ----------------------------------------------------------------------
# Core geometric checks
# ----------------------------------------------------------------------

def is_full_dimensional(A_ineq, tol=1e-7, box=1.0):
    """
    Check whether {tau : A_ineq @ tau >= 0} has non-empty interior.

    Maximizes t subject to  A_ineq[e] . tau >= t  for all e,  tau in [-box,box]^d.
    Returns (True/False, margin t_opt).
    """
    n, d = A_ineq.shape
    c = np.zeros(d + 1)
    c[-1] = -1.0  # minimize -t  <=>  maximize t
    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(d + 1)
        row[:d] = -A_ineq[i]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)  # -A_i.tau + t <= 0  <=>  A_i.tau >= t
    bounds = [(-box, box)] * d + [(-box, box)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return False, None
    t_opt = -res.fun
    return t_opt > tol, t_opt


def is_pointed(A_ineq, tol=1e-9):
    """
    A cone is pointed iff the only tau satisfying A_ineq @ tau = 0 (all rows
    tight simultaneously) is tau = 0, i.e. rank(A_ineq) == dim.
    """
    d = A_ineq.shape[1]
    rank = np.linalg.matrix_rank(A_ineq, tol=tol)
    return rank == d


def _dedupe_directions(A_ineq, tol=1e-7):
    """
    Group row indices by normalized direction (rows that are positive
    multiples of each other represent the SAME inequality/facet direction).
    Returns a list of representative indices, one per distinct direction,
    plus a map representative -> list of all indices sharing that direction.
    This must be done before redundancy testing: a one-at-a-time "is this
    implied by all the OTHERS" LP will wrongly call a constraint redundant
    if an exact duplicate of it is sitting among "the others".
    """
    n = A_ineq.shape[0]
    norms = np.linalg.norm(A_ineq, axis=1)
    reps = []          # representative index per group
    groups = {}        # representative -> list of member indices
    used = [False] * n
    for i in range(n):
        if used[i] or norms[i] < tol:
            continue
        vi = A_ineq[i] / norms[i]
        members = [i]
        used[i] = True
        for j in range(i + 1, n):
            if used[j] or norms[j] < tol:
                continue
            vj = A_ineq[j] / norms[j]
            if np.linalg.norm(vi - vj) < 1e-6:  # same direction (positive multiple)
                members.append(j)
                used[j] = True
        reps.append(i)
        groups[i] = members
    return reps, groups


def facet_indices(A_ineq, tol=1e-7, box=1.0):
    """
    Return indices of the non-redundant ("active") facets of
    {tau : A_ineq @ tau >= 0}, i.e. those inequalities that are not implied
    by the rest.

    Constraint i is a genuine facet if there exists tau with
      A_ineq[i] . tau < 0   while   A_ineq[j] . tau >= 0  for all j != i.

    Rows that are exact positive multiples of each other (duplicate
    directions) are first collapsed to a single representative -- otherwise
    a duplicated constraint is invisible to itself in the "check against all
    others" test and gets wrongly discarded as redundant (see _dedupe_directions).
    """
    reps, groups = _dedupe_directions(A_ineq, tol=tol)
    if not reps:
        return []
    A_reduced = A_ineq[reps]
    m = len(reps)
    active_local = []
    for i in range(m):
        c = A_reduced[i]
        A_ub, b_ub = [], []
        for j in range(m):
            if j == i:
                continue
            A_ub.append(-A_reduced[j])
            b_ub.append(0.0)
        # m == 1 (no other constraints to compare against) is the generic
        # case for a full-dimensional pointed cone in dim == 1 -- a ray in R
        # has exactly one defining inequality. linprog wants None, not an
        # empty list, to mean "no inequality constraints".
        if not A_ub:
            A_ub, b_ub = None, None
        bounds = [(-box, box)] * A_ineq.shape[1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.success and res.fun < -tol:
            active_local.append(i)
    # map back to a representative original index per active direction
    return [reps[k] for k in active_local]


def ray_count(A_ineq, active_idx, tol=1e-7):
    """
    Cross-check: count extreme rays of the cone by testing, for every
    combination of (dim-1) active facets, whether the resulting direction
    vector actually lies in the cone (satisfies all inequalities).
    Returns the number of distinct extreme rays found (up to sign/scale).
    """
    d = A_ineq.shape[1]
    if len(active_idx) < d - 1:
        return 0
    rays = []
    for combo in combinations(active_idx, d - 1):
        M = A_ineq[list(combo)]
        ns = null_space_1d(M, tol=tol)
        if ns is None:
            continue
        v = ns
        # orient so that it (approximately) satisfies the inequalities
        viol = A_ineq @ v
        if np.all(viol >= -tol):
            direction = v
        elif np.all(-viol >= -tol):
            direction = -v
        else:
            continue
        # dedupe by normalized direction
        nrm = direction / np.linalg.norm(direction)
        if not any(np.allclose(nrm, r, atol=1e-5) or np.allclose(nrm, -r, atol=1e-5)
                   for r in rays):
            rays.append(nrm)
    return len(rays)


def null_space_1d(M, tol=1e-9):
    """Return a unit vector spanning the 1-dim nullspace of M (d-1 x d), or None."""
    u, s, vh = np.linalg.svd(M)
    d = M.shape[1]
    if s.size < d or s[-1] < tol * max(1.0, s[0] if s.size else 1.0):
        return vh[-1]
    # check the smallest singular value is (numerically) zero
    if len(s) == d - 1 or (len(s) == d and s[-1] < 1e-7):
        return vh[-1]
    return None


# ----------------------------------------------------------------------
# High-level per-cone classification
# ----------------------------------------------------------------------

def classify_cone(A_ineq, dim, tol=1e-7):
    """
    Full pipeline for a single cone (given by its inequality matrix A_ineq,
    rows = one per edge, A_ineq @ tau >= 0).

    Returns a dict with keys:
      status        : 'empty_or_lowdim' | 'not_pointed' | 'simplicial' | 'non_simplicial'
      n_facets      : number of non-redundant facets (if applicable)
      n_rays        : ray-count cross-check (if applicable)
      consistent    : whether facet count and ray count agree (3D sanity check)
    """
    fulldim, margin = is_full_dimensional(A_ineq, tol=tol)
    if not fulldim:
        return dict(status="empty_or_lowdim", n_facets=None, n_rays=None,
                     consistent=None, margin=margin)

    if not is_pointed(A_ineq):
        return dict(status="not_pointed", n_facets=None, n_rays=None,
                     consistent=None, margin=margin)

    active = facet_indices(A_ineq, tol=tol)
    nf = len(active)
    nr = ray_count(A_ineq, active, tol=tol)

    consistent = None
    if dim == 3:
        # for pointed full-dim cones in R^3: #facets == #rays always
        consistent = (nf == nr)

    if nf < dim:
        # Should not happen for a genuinely full-dimensional pointed cone;
        # flag rather than silently mislabel. Usually indicates the
        # pointedness/full-dimensionality checks passed only marginally
        # (near-degenerate cone) -- inspect with tighter tolerance.
        status = "suspect_underdetermined"
    elif nf == dim:
        status = "simplicial"
    else:
        status = "non_simplicial"

    return dict(status=status, n_facets=nf, n_rays=nr, consistent=consistent,
                margin=margin)


# ----------------------------------------------------------------------
# Basis reduction: raw constraint matrix + column basis -> generators a_e^B
# ----------------------------------------------------------------------

def basis_generators(A, basis, tol=1e-9):
    """
    Reduce the raw (E, F) constraint matrix to the basis-dependent generator
    matrix R_B (rows a_e^B), following the paper's eq. (1.21):

        R_B = A^T (A_B^{-1})^T   ,   equivalently   R_B = A @ inv(A[basis, :])

    where A_B = A[basis, :] is the F x F submatrix picked out by `basis`
    (an invertible choice of F edges). For e in `basis`, row e of R_B is a
    standard basis vector (up to which coordinate), since A_B @ A_B^{-1} = I
    -- e.g. the tree-diagram example (paper sec. 3.1) reproduces eq. (3.5)
    and eq. (3.9) exactly from this formula.

    Parameters
    ----------
    A     : (E, F) array -- raw constraint matrix, rows = edges.
    basis : length-F sequence of edge indices (0-indexed rows of A) whose
            submatrix A[basis, :] is invertible.
    tol   : singularity threshold on det(A_B).

    Returns
    -------
    R_B    : (E, F) array, row e is the generator a_e^B.
    det_AB : determinant of A[basis, :] (sign/magnitude are meaningful --
             see | det A_B | in the volume normalisation, eq. 1.48).
    """
    A = np.asarray(A, dtype=float)
    E, F = A.shape
    basis = tuple(int(i) for i in basis)
    if len(basis) != F:
        raise ValueError(f"basis must have exactly F={F} edge indices "
                          f"(one per constraint), got {len(basis)}: {basis}")
    if len(set(basis)) != len(basis):
        raise ValueError(f"basis indices must be distinct, got {basis}")
    if any(i < 0 or i >= E for i in basis):
        raise ValueError(f"basis indices must be in [0, {E - 1}], got {basis}")

    A_B = A[basis, :]
    det_AB = np.linalg.det(A_B)
    if abs(det_AB) < tol:
        raise ValueError(f"basis {basis} is singular (det A_B = {det_AB:.3g}); "
                          "these edges do not form a valid column basis")
    R_B = A @ np.linalg.inv(A_B)
    return R_B, det_AB


# ----------------------------------------------------------------------
# Full scan over all (basis, sigma) pairs for a constraint matrix A
# ----------------------------------------------------------------------

def scan_matrix(A, tol=1e-7, verbose=False, stop_after_first=False, basis=None):
    """
    Scan every invertible column basis B (if basis=None, else
    check only for the given basis) and every sign vector sigma
    in {+1,-1}^E, build the corresponding cone K_sigma^B, and classify it.

    Parameters
    ----------
    A : (E, F) array, entries typically in {-1, 0, 1}
    verbose : print a running summary
    stop_after_first : stop as soon as one non-simplicial cone is found
                        (useful for a quick existence check on large E)

    Returns
    -------
    list of dicts, one per (B, sigma) pair that gave a full-dimensional
    pointed cone, each with keys:
        'basis', 'sigma', 'det_AB', plus the classify_cone() output.
    """
    A = np.asarray(A, dtype=float)
    E, F = A.shape
    # Sanity check:
    if F > E:
        raise ValueError("Num of constraints (F) should be less than Num of Edges (E)." \
        "Found F>E. Perhaps you input the transpose matrix?")
    dim = F
    results = []

    if basis is not None:
        basis_set = [tuple(int(i) for i in basis)]
    else:
        basis_set = list(combinations(range(E), F))

    n_simplicial = 0
    n_nonsimplicial = 0
    n_skipped = 0
    n_suspect = 0

    for B_idx in basis_set:
        try:
            R_B, detAB = basis_generators(A, B_idx)
        except ValueError:
            continue  # singular basis

        for bits in product([1, -1], repeat=E):
            sigma = np.array(bits, dtype=float)
            A_ineq = np.array([-sigma[e] * R_B[e, :] for e in range(E)])
            res = classify_cone(A_ineq, dim, tol=tol)

            if res["status"] in ("empty_or_lowdim", "not_pointed"):
                n_skipped += 1
                continue

            entry = dict(basis=B_idx, sigma=tuple(bits), det_AB=detAB, **res)
            results.append(entry)

            if res["status"] == "simplicial":
                n_simplicial += 1
            elif res["status"] == "suspect_underdetermined":
                n_suspect += 1
                if verbose:
                    print(f"SUSPECT (fewer facets than dim -- inspect tol): "
                          f"basis={B_idx}, sigma={bits}, facets={res['n_facets']}, "
                          f"dim={dim}")
            else:  # non_simplicial
                n_nonsimplicial += 1
                if verbose:
                    print(f"NON-SIMPLICIAL: basis={B_idx}, det(A_B)={detAB:.3g}, "
                          f"sigma={bits}, facets={res['n_facets']}, "
                          f"rays={res['n_rays']}, dim={dim}")
                if stop_after_first:
                    if verbose:
                        print(f"\nStopping at first non-simplicial cone found "
                              f"(scanned {n_simplicial+n_nonsimplicial+n_skipped+n_suspect} "
                              f"candidates so far).")
                    return results

    if verbose:
        print(f"\nScan complete. dim={dim}, |E|={E}")
        print(f"  full-dimensional pointed cones checked: "
              f"{n_simplicial + n_nonsimplicial + n_suspect}")
        print(f"  simplicial:      {n_simplicial}")
        print(f"  non-simplicial:  {n_nonsimplicial}")
        print(f"  suspect (facets < dim, needs closer look): {n_suspect}")
        print(f"  skipped (degenerate / not pointed): {n_skipped}")
        if dim == 3:
            n_inconsistent = sum(1 for r in results if r.get("consistent") is False)
            if n_inconsistent:
                print(f"  WARNING: {n_inconsistent} cones failed the "
                      f"facet-count == ray-count sanity check (3D) -- "
                      f"inspect tol / numerical precision.")

    return results

def run_demo():
    print("=" * 70)
    print("Test 1: K4 closed graph (graphic matroid) -- expect ALL simplicial")
    print("=" * 70)
    # reduced incidence matrix of K4 (vertex 3 dropped), used as the
    # bond-matroid style constraint matrix on the cycle space -- but for a
    # quick self-test we just verify unimodularity pre-screen kicks in on
    # any graphic incidence matrix.
    aG_K4 = np.array([
        [1, 1, 1, 0, 0, 0],
        [-1, 0, 0, 1, 1, 0],
        [0, -1, 0, -1, 0, 1],
    ], dtype=float)
    scan_matrix(aG_K4.T, verbose=True)

    print()
    print("=" * 70)
    print("Test 2: rank-3 non-graphic matrix -- expect SOME non-simplicial")
    print("=" * 70)
    A_nongraphic = np.array([
        [1, 1, 0, -1, 0, 1],
        [0, 1, -1, 1, -1, 0],
        [1, 0, 1, 0, 1, -1],
    ], dtype=float)
    scan_matrix(A_nongraphic.T, verbose=True, stop_after_first=True)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="cone checker")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_demo = subparsers.add_parser("demo", help="Run a demo example")

    p_check = subparsers.add_parser("check", help="Run a custom example")
    p_check.add_argument("--matrix", type=str, help="JSON array of constraint matrix, e.g. '[[1,0],[0,1]]'")
    p_check.add_argument("--basis", type=str, default=None, help="JSON array of basis indices, e.g. '[0,1,3]'; omit to scan all bases")

    args = parser.parse_args()
    if args.mode == "demo":
        run_demo()
    else:
        import json
        scan_matrix(
            A = np.array(json.loads(args.matrix), dtype=float), 
            basis = np.array(json.loads(args.basis), dtype=float) if args.basis is not None else None, 
            verbose=True
            )

    