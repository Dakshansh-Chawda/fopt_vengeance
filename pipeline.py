"""
pipeline.py

Inputs
- the Feynman-matroid constraint matrix A
(rows = edges, columns = constraints: (E, F), and
- a chosen column basis B (a length-F list of edge indices)

  1. the basis-reduced generators  a_e^B = A @ inv(A[B, :])   (eq. 1.21),
     via cone_checker.basis_generators()
  2. the surviving sign vectors sigma (non-empty cones K_sigma^B), via
     cone_sign_analysis.find_surviving_sign_vectors()
  3. the corresponding Zonotope (vertices q^sigma, convex hull, polar dual),
     via zonotope_analysis.Zonotope
  4. per-surviving-cone simplicial/non-simplicial classification, restricted
     to B, via cone_checker.classify_cone()
  5. cone-decomposition / zonotope-hull / primal-dual plots, dispatched to
     whichever of full_generic_analysis / interactive_3d_plots /
     interactive_4d_plots supports the resulting dimension -- or, outside
     dim in {2, 3, 4} where no spatial plot exists in this repo, the
     dimension-agnostic text report (print_table / print_dual_table)
     instead.

Sign convention
----------------
cone_sign_analysis.find_surviving_sign_vectors(A) tests sigma_e*(A[e].tau)>=0,
while the paper's K_sigma^B (eq. 1.24) and cone_checker's classify_cone both
use the opposite sign: -sigma_e*(R_B tau)_e >= 0. Calling
find_surviving_sign_vectors(-R_B) (not R_B) is what makes the returned sigma
directly usable both in q^sigma = q0 + sum sigma_e d_e a_e^B (eq. 1.49, no
extra sign) and in cone_checker's own A_ineq = -sigma_e * R_B[e] convention
-- see cone_sign_analysis.py's own "Sign convention" docstring section for
the general form of this trick.

Two-stage entry points
-----------------------
compute_pipeline(A, basis, ...)   -- pure computation, no plotting/printing
plot_pipeline(result, ...)        -- dispatches plots by result.dim
run_pipeline(A, basis, ...)       -- compute + print + plot, one call
"""

from dataclasses import dataclass, field

import numpy as np

from cone_sign_analysis import find_surviving_sign_vectors
from cone_checker import basis_generators, classify_cone
from zonotope_analysis import Zonotope, print_table
from dual_analysis import print_dual_table


@dataclass
class PipelineResult:
    A: np.ndarray
    basis: tuple
    R_B: np.ndarray
    det_AB: float
    dim: int
    n_edges: int
    survivors: list
    orientations: list
    zonotope: Zonotope
    cone_results: list = field(repr=False)
    n_simplicial: int = 0
    n_nonsimplicial: int = 0
    n_suspect: int = 0
    n_anomalous: int = 0


# ---------------------------------------------------------------------------
# Stage 1: compute (basis reduction -> survivors -> zonotope -> simpliciality)
# ---------------------------------------------------------------------------

def compute_pipeline(A, basis, d=None, q0=None, tol=1e-9, verbose=True):
    """
    Parameters
    ----------
    A     : (n_edges, n_constraints) array -- RAW constraint matrix, same
            convention as cone_checker.scan_matrix (rows = edges).
    basis : length n_constraints sequence of edge indices forming an
            invertible column basis (as in cone_checker's --basis).
    d, q0 : forwarded to Zonotope (segment half-lengths / base point);
            default to all-ones / origin like Zonotope itself.
    tol   : shared LP / singularity tolerance for every stage.
    verbose : print the parallel-class reduction (from
            find_surviving_sign_vectors) and a final summary.

    Returns
    -------
    PipelineResult
    """
    A = np.asarray(A, dtype=float)
    R_B, det_AB = basis_generators(A, basis, tol=tol)
    n_edges, dim = R_B.shape

    survivors = find_surviving_sign_vectors(-R_B, tol=tol, verbose=verbose)
    orientations = [(str(i + 1), np.array(s, dtype=float))
                    for i, s in enumerate(survivors)]

    zonotope = Zonotope(R_B, d=d, q0=q0)

    cone_results = []
    n_simplicial = n_nonsimplicial = n_suspect = n_anomalous = 0
    for sigma in survivors:
        s = np.array(sigma, dtype=float)
        A_ineq = -s[:, None] * R_B
        res = classify_cone(A_ineq, dim, tol=tol)
        cone_results.append(dict(sigma=sigma, **res))
        status = res["status"]
        if status == "simplicial":
            n_simplicial += 1
        elif status == "non_simplicial":
            n_nonsimplicial += 1
        elif status == "suspect_underdetermined":
            n_suspect += 1
        else:
            # empty_or_lowdim / not_pointed: find_surviving_sign_vectors only
            # tests full-dimensionality (no pointedness check), so a survivor
            # rejected here means the two LP tolerances disagree at this
            # basis -- rare, but worth surfacing rather than silently
            # dropping it from the counts.
            n_anomalous += 1

    result = PipelineResult(
        A=A, basis=tuple(int(i) for i in basis), R_B=R_B, det_AB=det_AB,
        dim=dim, n_edges=n_edges, survivors=survivors,
        orientations=orientations, zonotope=zonotope,
        cone_results=cone_results, n_simplicial=n_simplicial,
        n_nonsimplicial=n_nonsimplicial, n_suspect=n_suspect,
        n_anomalous=n_anomalous)

    if verbose:
        _print_summary(result)
    return result


def _print_summary(r):
    print("Pipeline summary")
    print("-" * 46)
    print(f"  basis B         : {r.basis}  (det A_B = {r.det_AB:.6g})")
    print(f"  ambient dim     : {r.dim}")
    print(f"  surviving cones : {len(r.survivors)}")
    print(f"    simplicial      : {r.n_simplicial}")
    print(f"    non-simplicial  : {r.n_nonsimplicial}")
    if r.n_suspect:
        print(f"    suspect         : {r.n_suspect}  (facets < dim -- inspect tol)")
    if r.n_anomalous:
        print(f"    anomalous       : {r.n_anomalous}  (rejected by classify_cone "
              f"despite surviving the LP scan -- tolerance mismatch)")
    print()


# ---------------------------------------------------------------------------
# Stage 2: plot, dispatched by dimension
# ---------------------------------------------------------------------------

_SPATIAL_DIMS = (2, 3, 4)


def plot_pipeline(result, interactive=True, n_pts=None, show=True, titles=None):
    """
    Dispatch cone-decomposition / zonotope-hull / primal-dual plots
    according to result.dim:
        dim == 2          -> full_generic_analysis / zonotope_analysis / dual_analysis (matplotlib)
        dim == 3           -> interactive_3d_plots (Plotly) if interactive else full_generic_analysis
        dim == 4           -> interactive_4d_plots (Plotly; no matplotlib support exists for 4-D)
        dim not in {2,3,4} -> no spatial plot exists in this repo; prints
                              print_table()/print_dual_table() instead.

    Returns
    -------
    dict of the figures produced (empty for the text-only fallback).
    """
    dim = result.dim
    R_B, survivors = result.R_B, result.survivors
    z, orientations = result.zonotope, result.orientations
    cone_kwargs = {} if n_pts is None else {"n_pts": n_pts}
    figs = {}

    if dim == 2:
        from full_generic_analysis import plot_cone_decomposition
        from zonotope_analysis import plot as plot_zonotope
        from dual_analysis import plot_primal_dual
        figs["cones"] = plot_cone_decomposition(R_B, survivors, **cone_kwargs)
        figs["hull"] = plot_zonotope(z)
        figs["dual"] = plot_primal_dual(z, orientations=orientations, titles=titles)

    elif dim == 3 and interactive:
        from interactive_3d_plots import (plot_cone_decomposition_interactive,
                                           plot_3d_interactive,
                                           plot_primal_dual_3d_interactive)
        figs["cones"] = plot_cone_decomposition_interactive(R_B, survivors, **cone_kwargs)
        figs["hull"] = plot_3d_interactive(z, orientations=orientations)
        figs["dual"] = plot_primal_dual_3d_interactive(z, orientations=orientations, titles=titles)
        if show:
            for fig in figs.values():
                fig.show()

    elif dim == 3:
        from full_generic_analysis import (plot_cone_decomposition, plot_3d,
                                            plot_primal_dual_3d)
        figs["cones"] = plot_cone_decomposition(R_B, survivors, **cone_kwargs)
        figs["hull"] = plot_3d(z, orientations=orientations)
        figs["dual"] = plot_primal_dual_3d(z, orientations=orientations, titles=titles)

    elif dim == 4:
        from interactive_4d_plots import (plot_cone_decomposition_interactive_4d,
                                           plot_4d_interactive,
                                           plot_primal_dual_4d_interactive)
        figs["cones"] = plot_cone_decomposition_interactive_4d(R_B, survivors, **cone_kwargs)
        figs["hull"] = plot_4d_interactive(z, orientations=orientations)
        figs["dual"] = plot_primal_dual_4d_interactive(z, orientations=orientations, titles=titles)
        if show:
            for fig in figs.values():
                fig.show()

    else:
        print(f"No spatial plotting is available for dim == {dim} "
              f"(this repo only plots dim in {_SPATIAL_DIMS}); "
              "showing the dimension-agnostic vertex/dual tables instead.\n")
        print_table(z, orientations=orientations)
        try:
            print_dual_table(z, orientations=orientations)
        except ValueError as exc:
            print(f"  (polar dual unavailable: {exc})")

    return figs


# ---------------------------------------------------------------------------
# Convenience: compute + report + plot in one call
# ---------------------------------------------------------------------------

def run_pipeline(A, basis, d=None, q0=None, tol=1e-9, interactive=True,
                  n_pts=None, show=True, verbose=True, titles=None):
    """compute_pipeline() + print_table() + plot_pipeline(), in one call."""
    result = compute_pipeline(A, basis, d=d, q0=q0, tol=tol, verbose=verbose)
    print_table(result.zonotope, orientations=result.orientations)
    figs = plot_pipeline(result, interactive=interactive, n_pts=n_pts,
                          show=show, titles=titles)
    return result, figs


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="basis -> surviving cones -> zonotope -> simpliciality, in one shot")
    parser.add_argument("--matrix", type=str, required=True,
                         help="JSON array, raw constraint matrix, shape (n_edges, n_constraints)")
    parser.add_argument("--basis", type=str, required=True,
                         help="JSON array of edge indices forming a column basis, e.g. '[0,1,3]'")
    parser.add_argument("--d", type=str, default=None,
                         help="JSON array of segment half-lengths (default: all ones)")
    parser.add_argument("--q0", type=str, default=None,
                         help="JSON array base point (default: origin)")
    args = parser.parse_args()

    A_in = np.array(json.loads(args.matrix), dtype=float)
    basis_in = json.loads(args.basis)
    d_in = np.array(json.loads(args.d), dtype=float) if args.d else None
    q0_in = np.array(json.loads(args.q0), dtype=float) if args.q0 else None

    result = compute_pipeline(A_in, basis_in, d=d_in, q0=q0_in, verbose=True)
    print_table(result.zonotope, orientations=result.orientations)
