"""
Cone condition analysis: To evaluate the non-empty cones
 
Given the cone conditions

    K_sigma = { tau in R^d : sigma_e * (A[e] . tau) >= 0   for all edges e }

(one linear form per edge, encoded as a row of the matrix A), find every sign
vector sigma in {+1,-1}^|E| for which K_sigma is non-empty -- i.e. has a
genuine interior point, not just the origin or a lower-dimensional face.

Feed the generator vectors a_e^B (rows of A); returns the
full list of surviving sign vectors directly

Method
------
1. Detect edges whose linear forms are proportional (parallel hyperplanes).
   These are forced to share a single free sign (up to a known relative
   sign), which shrinks the search space from 2^|E| to 2^(#distinct
   directions) for free -- no LP needed for this step.
2. For each sign pattern on the reduced set of independent directions, run a
   small linear program (LP) that tests whether the corresponding cone has
   non-empty interior (i.e. is full-dimensional).
3. Expand the surviving reduced patterns back into full sign vectors on the
   original edge set.

Sign convention
----------------
The cone condition implemented here is  sigma_e * (A[e] . tau) >= 0.
If your cone conditions are written as  -sigma_e * (A[e] . tau) >= 0 (as in
the "K^B_sigma" convention of the companion notes), simply pass  -A  instead
of  A: the set of surviving sign vectors for  -A  is exactly the set for  A
with every sign flipped, and the *count* is identical either way.

Usage:

python cone_sign_analysis.py --matrix '[[1,0],[0,1]]'
"""

from itertools import product
import numpy as np
from scipy.optimize import linprog


def find_surviving_sign_vectors(A, tol=1e-9, verbose=False):
    """
    Parameters
    ----------
    A : array_like, shape (n_edges, n_constraints)
        Row e gives the coefficients of the linear form ell_e(tau) = A[e].tau
        appearing in the e-th cone condition sigma_e * ell_e(tau) >= 0.
    tol : float
        Numerical tolerance used both for detecting proportional rows and
        for the LP feasibility test.
    verbose : bool
        If True, print the parallel-class reduction and search-space size.

    Returns
    -------
    survivors : list of tuple(int)
        Each tuple is a sign vector (length n_edges, entries +-1) for which
        the cone K_sigma has non-empty interior.
    """
    A = np.asarray(A, dtype=float)
    A = A.T                         # transpose immediately to shape (n_constraints, n_edges) as in paper
    n_edges, d = A.shape

    canonical, rel_sign, reps = _group_parallel_rows(A, tol)
    n_groups = len(reps)

    if verbose:
        print(f"{n_edges} edges reduce to {n_groups} independent hyperplane "
              f"direction(s).")
        print(f"Search space: 2^{n_edges} = {2**n_edges}  ->  "
              f"2^{n_groups} = {2**n_groups} LP calls.")

    survivors = []
    for t in product([1, -1], repeat=n_groups):
        sigma = tuple(t[canonical[e]] * rel_sign[e] for e in range(n_edges))
        if _cone_has_interior(A, sigma, tol):
            survivors.append(sigma)

    return survivors


def _group_parallel_rows(A, tol):
    """
    Group rows of A that are proportional to each other (i.e. lie on the same
    line through the origin -- same hyperplane, possibly opposite normal
    direction).

    Returns
    -------
    canonical : list[int]      group index of each edge
    rel_sign  : list[+-1]      +1 if the edge's row points the same way as
                                the group's representative row, -1 if opposite
    reps      : list[np.ndarray]  representative row vector of each group
    """
    n_edges = A.shape[0]
    canonical = [-1] * n_edges
    rel_sign = [1] * n_edges
    reps = []

    for e in range(n_edges):
        v = A[e]
        if np.linalg.norm(v) < tol:
            raise ValueError(f"Row {e} of A is (numerically) the zero vector.")
        placed = False
        for g, rep in enumerate(reps):
            if _is_proportional(v, rep, tol):
                canonical[e] = g
                rel_sign[e] = 1 if np.dot(v, rep) > 0 else -1
                placed = True
                break
        if not placed:
            canonical[e] = len(reps)
            rel_sign[e] = 1
            reps.append(v)

    return canonical, rel_sign, reps


def _is_proportional(v, w, tol):
    """True iff v = c*w for some nonzero scalar c (same line through origin)."""
    M = np.vstack([v, w])
    s = np.linalg.svd(M, compute_uv=False)
    # rank(M) == 1  <=>  smallest singular value ~ 0
    return s[-1] < tol * max(1.0, s[0])


def _cone_has_interior(A, sigma, tol):
    """
    LP feasibility test: does

        { tau : sigma_e * (A[e].tau) >= 0  for all e }

    have non-empty interior?  Maximize a slack epsilon subject to

        sigma_e * (A[e].tau) - epsilon >= 0   for all e
        -1 <= tau_i <= 1                      (harmless box: the cone is
                                                scale invariant, so this
                                                never creates a false
                                                negative)
        0 <= epsilon <= 1

    and check whether the optimal epsilon is strictly positive.
    """
    n_edges, d = A.shape

    c = np.zeros(d + 1)
    c[-1] = -1.0  # minimize -epsilon  <=>  maximize epsilon

    A_ub = np.zeros((n_edges, d + 1))
    for e in range(n_edges):
        A_ub[e, :d] = -sigma[e] * A[e]
        A_ub[e, d] = 1.0
    b_ub = np.zeros(n_edges)

    bounds = [(-1, 1)] * d + [(0, 1)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return False
    return (-res.fun) > tol


def format_sign_vector(sigma):
    return "(" + ", ".join("+" if s > 0 else "-" for s in sigma) + ")"

def full_run(matrix):
    survivors = find_surviving_sign_vectors(matrix, verbose=True)
    print(f"\n{len(survivors)} surviving sign vectors:")
    for s in survivors:
        print("  ", format_sign_vector(s))

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="cone decomposition")
    parser.add_argument("--matrix", type=str, help="JSON array of constraint matrix, e.g. '[[1,0],[0,1]]'")
    args = parser.parse_args()
    full_run(np.array(json.loads(args.matrix), dtype=float))
    

