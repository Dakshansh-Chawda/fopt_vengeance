# ===========================================================================
# POLAR DUAL ANALYSIS
# ===========================================================================

from itertools import product
from zonotope_analysis import Zonotope

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def polar_dual(points, center=None, tol=1e-9, return_hull=False):
    """
    Polar dual of P = conv(points), valid in ANY dimension.

        P* = { y : v . y <= 1  for all v in P }

    Dual vertices are read off from the FACETS of P: a facet with outward
    normal n and offset c > 0 (facet plane n.x = c) contributes the dual
    vertex n / c.

    Generalizations over an edge-loop implementation:
      * vertices may be given in ANY order (hull computed internally);
      * interior points and collinear boundary points are handled
        automatically (Qhull drops them, so no duplicate dual vertices);
      * works in any dimension via ConvexHull facet equations;
      * optional `center`: dual taken about this point (default: origin;
        pass 'centroid' to use the vertex centroid; pass q0 to match the
        companion paper's convention P = Q^circ about q0).  The returned
        dual lives in the translated frame y -> y - center.

    Returns
    -------
    dual_verts : (n_facets, dim) array of dual vertices
                 (in 2D, ordered counter-clockwise)
    hull       : ConvexHull of the (translated) primal, if return_hull
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    dim = pts.shape[1]

    if center is None:
        c0 = np.zeros(dim)
    elif isinstance(center, str) and center == "centroid":
        c0 = pts.mean(axis=0)
    else:
        c0 = np.asarray(center, dtype=float)

    P = pts - c0
    hull = ConvexHull(P)

    # Qhull facet equations:  A x + b <= 0  for interior points.
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    if np.any(b > -tol):
        raise ValueError(
            "Center is not strictly interior to the polytope; "
            "polar dual has a vertex at infinity. Try center='centroid'.")

    dual = A / (-b)[:, None]                       # n / c per facet
    dual = np.unique(dual.round(12), axis=0)       # merge duplicate facets

    if dim == 2:                                   # order CCW for plotting
        ang = np.arctan2(dual[:, 1], dual[:, 0])
        dual = dual[np.argsort(ang)]

    return (dual, hull) if return_hull else dual


def volume(points):
    """Volume (area in 2D) of conv(points), any dimension."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return ConvexHull(pts).volume


# -- attach dual computation to the Zonotope class --------------------------

def _zonotope_polar_dual(self, orientations=None, center=None, tol=1e-9):
    """
    Polar dual P* of the zonotope P.  The dual is taken about q0 by
    default, matching the companion paper's convention P = Q^circ.
    Pass `orientations` to restrict to surviving (non-empty-cone) sign
    vectors, exactly as in Zonotope.vertices / Zonotope.analyse.
    """
    _, verts = self.vertices(orientations)
    c = self.q0 if center is None else center
    return polar_dual(verts, center=c, tol=tol)


Zonotope.polar_dual = _zonotope_polar_dual


# ---------------------------------------------------------------------------
# Reporting (dual)
# ---------------------------------------------------------------------------

def print_dual_table(z, orientations=None, factorial_dim=None):
    """
    Print primal hull vertices, their dual counterparts, and volumes,
    including the companion-paper normalisation  |F|! * Vol(P*).

    Parameters
    ----------
    z             : Zonotope (dim == 2 for the table layout)
    orientations  : optional restricted sign-vector list (e.g. only the
                    non-empty cones)
    factorial_dim : if given (e.g. |F| = 2), also prints
                    factorial_dim! * Vol(P*)
    """
    res = z.analyse(orientations)
    hull_pts = res["hull_pts"]
    dual = z.polar_dual(orientations)

    print(f"  Primal hull: {res['n_sides']}-gon,  "
          f"dual: {len(dual)} vertices")
    print(f"  {'primal vertex':^24} {'dual vertex':^28}")
    print("  " + "-" * 54)
    n = max(len(hull_pts), len(dual))
    for i in range(n):
        p = (f"({hull_pts[i, 0]:>7.3f}, {hull_pts[i, 1]:>7.3f})"
             if i < len(hull_pts) else "")
        d = (f"({dual[i, 0]:>10.6f}, {dual[i, 1]:>10.6f})"
             if i < len(dual) else "")
        print(f"  {p:^24} {d:^28}")

    vol_p = volume(hull_pts)
    vol_d = volume(dual)
    print(f"\n  Vol(P)  = {vol_p:.6f}")
    print(f"  Vol(P*) = {vol_d:.6f}")
    if factorial_dim:
        import math
        f = math.factorial(factorial_dim)
        print(f"  {factorial_dim}! * Vol(P*) = {f * vol_d:.6f}"
              f"   (companion-paper normalisation |F|! Vol)")
    print()


# ---------------------------------------------------------------------------
# Plot (primal + dual, 2D)
# ---------------------------------------------------------------------------

def plot_primal_dual(cases, orientations=None, figsize=None, titles=None):
    """
    Side-by-side plots of P and its polar dual P*, one row per case.
    Colours match: the k-th facet of P and the dual vertex it generates
    share a colour (facet midpoints are marked with coloured squares).

    Parameters
    ----------
    cases        : list of items; each item is a Zonotope or an
                   (n, 2) array of primal vertices
    orientations : optional sign-vector list applied to every Zonotope
                   in `cases` (e.g. the surviving non-empty cones)
    titles       : optional list of row titles
    """
    if not isinstance(cases, (list, tuple)):
        cases = [cases]
    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=figsize or (12, 6 * n),
                             squeeze=False)
    cmap = plt.get_cmap("tab10")

    for r, case in enumerate(cases):
        if isinstance(case, Zonotope):
            res = case.analyse(orientations)
            hull_pts = res["hull_pts"]
            center = case.q0
        else:
            pts = np.asarray(case, dtype=float)
            h = ConvexHull(pts)
            hull_pts = pts[h.vertices]
            center = np.zeros(2)

        dual = polar_dual(hull_pts, center=center)
        axP, axD = axes[r]

        # ---------- primal ----------
        poly = plt.Polygon(hull_pts, closed=True,
                           facecolor="#3B8BD422", edgecolor="#185FA5",
                           linewidth=1.5, zorder=1)
        axP.add_patch(poly)
        axP.scatter(hull_pts[:, 0], hull_pts[:, 1],
                    color="#185FA5", s=55, zorder=3)
        for i, p in enumerate(hull_pts):
            axP.annotate(f"$q_{{{i + 1}}}$", p, textcoords="offset points",
                         xytext=(6, 4), fontsize=9)

        # facet midpoints coloured to match the dual vertices
        m = len(hull_pts)
        for i in range(m):
            a, b_ = hull_pts[i], hull_pts[(i + 1) % m]
            mid = 0.5 * (a + b_)
            axP.scatter(*mid, color=cmap(i % 10), s=36, marker="s",
                        zorder=4, edgecolors="k", linewidths=0.4)

        axP.scatter(*center, color="k", s=25, marker="x", zorder=5)
        axP.set_title(f"P  ({m}-gon),  Vol = {volume(hull_pts):.4g}",
                      fontsize=10)

        # ---------- dual ----------
        polyD = plt.Polygon(dual, closed=True,
                            facecolor="#D4703B22", edgecolor="#A5471F",
                            linewidth=1.5, zorder=1)
        axD.add_patch(polyD)
        # colour each dual vertex by the primal facet that generated it,
        # recomputed edge-by-edge so the colours line up with the squares
        for i in range(m):
            a, b_ = hull_pts[i], hull_pts[(i + 1) % m]
            edge = b_ - a
            nrm = np.array([edge[1], -edge[0]])
            c = np.dot(nrm, a - center)
            dv = nrm / c
            axD.scatter(*dv, color=cmap(i % 10), s=55, zorder=3,
                        edgecolors="k", linewidths=0.5)
            axD.annotate(f"$q^*_{{{i + 1}}}$", dv,
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=9)
        axD.scatter(0, 0, color="k", s=25, marker="x", zorder=5)
        axD.set_title(f"P*  ({len(dual)} vertices),  "
                      f"Vol = {volume(dual):.4g}", fontsize=10)

        for ax in (axP, axD):
            ax.axhline(0, color="#00000018", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="#00000018", linewidth=0.8, linestyle="--")
            ax.set_aspect("equal")
            ax.margins(0.25)
            ax.set_xlabel("$Q_1$")
            ax.set_ylabel("$Q_2$")

        if titles and r < len(titles):
            axP.text(-0.25, 0.5, titles[r], transform=axP.transAxes,
                     rotation=90, va="center", fontsize=11)

    fig.suptitle("Zonotope P and polar dual P* (matching colours: facet of P $\\leftrightarrow$ "
                 "vertex of P*)\n\n", fontsize=12)
    fig.tight_layout()
    return fig