"""
Generalized zonotope convex-hull analysis.

A zonotope is the Minkowski sum of line segments:

    Z = q0 + sum_e [-d_e * g_e , +d_e * g_e]

with vertices located at

    q^sigma = q0 + sum_e sigma_e * d_e * g_e ,      sigma_e in {-1,+1}.

This module works for ANY number of generators, ANY generator matrix,
and any ambient dimension (hull analysis is dimension-agnostic; plotting
is provided for 2D).

Main entry points
-----------------
Zonotope(generators, d=None, q0=None)   -- the central object
    .vertices(orientations=None)        -- sigma-vertices (all 2^m by default)
    .analyse(orientations=None)         -- hull + on-hull classification
    .direction_dependencies()           -- general replacement for the old
                                           hard-coded check_direction_dependency()
    .predicted_hull_sides()             -- 2 * (# distinct directions) in 2D
plot(zonotope_or_list, ...)             -- side-by-side hull plots (2D)
print_table(zonotope, ...)              -- vertex table with hull status
"""

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull, QhullError

# ---------------------------------------------------------------------------
# Core object
# ---------------------------------------------------------------------------

class Zonotope:
    """
    Parameters
    ----------
    generators : (m, dim) array-like
        One row per generator g_e (any m, any dim >= 1).
    d : array-like of length m, optional
        Segment half-lengths d_e (default: all ones).
    q0 : array-like of length dim, optional
        Base point (default: origin).
    """

    def __init__(self, generators, d=None, q0=None):
        self.G = np.atleast_2d(np.asarray(generators, dtype=float))
        self.m, self.dim = self.G.shape
        self.d = np.ones(self.m) if d is None else np.asarray(d, dtype=float)
        if self.d.shape != (self.m,):
            raise ValueError(f"d must have length {self.m}, got {self.d.shape}")
        self.q0 = (np.zeros(self.dim) if q0 is None
                   else np.asarray(q0, dtype=float))
        if self.q0.shape != (self.dim,):
            raise ValueError(f"q0 must have length {self.dim}")

    # -- effective (scaled) generators --------------------------------------
    @property
    def scaled(self):
        """d_e * g_e, shape (m, dim)."""
        return self.d[:, None] * self.G

    # -- sign vectors --------------------------------------------------------
    def all_orientations(self):
        """All 2^m sign vectors as a list of (label, sigma) pairs."""
        out = []
        for i, signs in enumerate(product([+1, -1], repeat=self.m), start=1):
            out.append((str(i), np.array(signs, dtype=float)))
        return out

    # -- vertices ------------------------------------------------------------
    def vertices(self, orientations=None):
        """
        Vertex candidates q^sigma for the given orientations
        (default: all 2^m sign vectors).

        Returns
        -------
        labels : list of str
        verts  : (n, dim) array
        """
        if orientations is None:
            orientations = self.all_orientations()
        labels, verts = [], []
        for name, sigma in orientations:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape != (self.m,):
                raise ValueError(
                    f"orientation {name!r} has length {len(sigma)}, "
                    f"expected {self.m}")
            labels.append(name)
            verts.append(self.q0 + sigma @ self.scaled)
        return labels, np.array(verts)

    # -- direction dependency (general) --------------------------------------
    def direction_dependencies(self, tol=1e-9, verbose=True):
        """
        General analysis of linear structure among the generators.
        Replaces the old hard-coded check a_4 = a_2 + a_3.

        Reports
        -------
        rank              : rank of the generator matrix
        parallel_classes  : generators grouped by direction (up to sign);
                            parallel generators fuse into a single zonotope
                            direction, reducing the number of hull sides
        null_space        : basis of linear relations  c . G = 0  (each row c
                            is one exact dependency among the generators)
        n_directions      : number of distinct (non-parallel, non-zero)
                            directions among the scaled generators
        """
        S = self.scaled

        # ---- rank ----
        rank = np.linalg.matrix_rank(S, tol=tol)

        # ---- clean dependency relations via RREF of S^T ----
        # Columns of S^T are the scaled generators. RREF expresses every
        # non-pivot generator as a combination of the pivot ones, giving
        # human-readable relations like  g_4 = g_2 + g_3.
        pivots, coeffs = _rref_dependencies(S.T, tol=tol)
        # Each entry of `relations` is (dependent_index, {pivot_index: coeff})
        relations = []
        pivot_set = set(pivots)
        for e in range(self.m):
            if e in pivot_set:
                continue
            combo = {p: coeffs[e][k] for k, p in enumerate(pivots)
                     if abs(coeffs[e][k]) > tol}
            relations.append((e, combo))

        # ---- group by direction (up to sign) ----
        classes = []                     # list of lists of generator indices
        reps = []                        # representative unit vectors
        for e in range(self.m):
            v = S[e]
            n = np.linalg.norm(v)
            if n < tol:                  # zero generator: contributes nothing
                classes.append(None)
                continue
            u_e = v / n
            placed = False
            for ci, r in enumerate(reps):
                if (np.linalg.norm(u_e - r) < tol
                        or np.linalg.norm(u_e + r) < tol):
                    classes[ci].append(e) if isinstance(classes[ci], list) \
                        else None
                    # find the class list to append to
                    placed = True
                    break
            if not placed:
                reps.append(u_e)
                classes.append([e])

        # tidy: build the actual grouping
        parallel_classes = []
        reps2 = []
        used = set()
        for e in range(self.m):
            if e in used:
                continue
            v = S[e]
            n = np.linalg.norm(v)
            if n < tol:
                parallel_classes.append(("zero", [e]))
                used.add(e)
                continue
            u_e = v / n
            group = [e]
            used.add(e)
            for f in range(e + 1, self.m):
                if f in used:
                    continue
                w = S[f]
                nf = np.linalg.norm(w)
                if nf < tol:
                    continue
                u_f = w / nf
                if (np.linalg.norm(u_e - u_f) < tol
                        or np.linalg.norm(u_e + u_f) < tol):
                    group.append(f)
                    used.add(f)
            parallel_classes.append((u_e, group))
            reps2.append(u_e)

        n_directions = sum(1 for r, _ in parallel_classes
                           if not isinstance(r, str))

        if verbose:
            print("Generator dependency analysis")
            print("-" * 46)
            print(f"  generators           : m = {self.m}, dim = {self.dim}")
            print(f"  rank of generator set: {rank}"
                  + ("  (full)" if rank == min(self.m, self.dim) else ""))
            print(f"  distinct directions  : {n_directions}")
            for rep, group in parallel_classes:
                idx = ", ".join(f"g_{e+1}" for e in group)
                if isinstance(rep, str):
                    print(f"    zero generator(s)  : {idx}")
                elif len(group) > 1:
                    print(f"    parallel class     : {{{idx}}}  "
                          f"direction ≈ {np.round(rep, 6)}")
                else:
                    print(f"    single             : {idx}       "
                          f"direction ≈ {np.round(rep, 6)}")
            if relations:
                print(f"  {len(relations)} dependency relation(s) "
                      f"(scaled generators d_e*g_e):")
                for e, combo in relations:
                    if not combo:
                        print(f"    d_{e+1}g_{e+1} = 0  (zero generator)")
                        continue
                    rhs = " + ".join(
                        (f"d_{p+1}g_{p+1}" if abs(c - 1) < tol else
                         f"(-1)·d_{p+1}g_{p+1}" if abs(c + 1) < tol else
                         f"({c:+.4g})·d_{p+1}g_{p+1}")
                        for p, c in sorted(combo.items()))
                    print(f"    d_{e+1}g_{e+1} = {rhs}")
            else:
                print("  generators are linearly independent")
            if self.dim == 2:
                print(f"  => hull is at most a "
                      f"{2 * n_directions}-gon (2 x #directions).")
            print()

        return dict(rank=rank,
                    relations=relations,
                    pivots=pivots,
                    parallel_classes=parallel_classes,
                    n_directions=n_directions)

    def predicted_hull_sides(self, tol=1e-9):
        """For dim == 2: exact side count = 2 x (# distinct directions)."""
        info = self.direction_dependencies(tol=tol, verbose=False)
        return 2 * info["n_directions"]

    # -- hull analysis --------------------------------------------------------
    def analyse(self, orientations=None, tol=1e-9):
        """
        Compute vertices, convex hull, and classify each sigma-vertex.

        Returns dict with keys:
            labels, vertices, hull, hull_pts, on_hull, n_sides,
            collinear, unique_pts
        """
        labels, verts = self.vertices(orientations)

        _, unique_idx = np.unique(verts.round(9), axis=0, return_index=True)
        unique_pts = verts[sorted(unique_idx)]

        base = dict(labels=labels, vertices=verts,
                    on_hull=np.zeros(len(verts), bool),
                    n_sides=0, collinear=[], hull=None,
                    hull_pts=np.zeros((0, self.dim)),
                    unique_pts=unique_pts)

        if len(unique_pts) <= self.dim:
            return base                      # degenerate: no full-dim hull

        try:
            hull = ConvexHull(unique_pts)
        except QhullError:                   # flat / degenerate configuration
            return base

        if self.dim == 2:
            hull_pts = unique_pts[hull.vertices]           # ordered corners
            on_hull = _points_on_hull_2d(verts, hull_pts)
            collinear = _find_collinear(hull_pts)
            n_sides = len(hull.vertices)
        else:
            corner_set = unique_pts[hull.vertices]
            on_hull = np.array([
                np.any(np.all(np.abs(corner_set - p) < 1e-9, axis=1))
                for p in verts])
            collinear = []
            n_sides = len(hull.simplices)    # facet count in dim > 2

        base.update(hull=hull, hull_pts=hull_pts if self.dim == 2
                    else corner_set,
                    on_hull=on_hull, n_sides=n_sides, collinear=collinear)
        return base


# ---------------------------------------------------------------------------
# Linear-algebra helper: RREF-based dependencies
# ---------------------------------------------------------------------------

def _rref_dependencies(M, tol=1e-9):
    """
    Reduced row echelon form of M (dim x m), treating COLUMNS as vectors.

    Returns
    -------
    pivots : list of pivot column indices (an independent subset of columns)
    coeffs : (m, r) array; row e gives the coefficients expressing column e
             as a combination of the pivot columns:
                 col_e = sum_k coeffs[e, k] * col_{pivots[k]}
             (For pivot columns this is a unit vector.)
    """
    A = np.array(M, dtype=float)
    n_rows, n_cols = A.shape
    R = A.copy()
    pivots = []
    row = 0
    for col in range(n_cols):
        if row >= n_rows:
            break
        # find pivot in this column at or below `row`
        piv = row + np.argmax(np.abs(R[row:, col]))
        if abs(R[piv, col]) < tol:
            continue
        R[[row, piv]] = R[[piv, row]]
        R[row] = R[row] / R[row, col]
        for r in range(n_rows):
            if r != row and abs(R[r, col]) > tol:
                R[r] -= R[r, col] * R[row]
        pivots.append(col)
        row += 1

    r = len(pivots)
    coeffs = np.zeros((n_cols, r))
    for e in range(n_cols):
        if e in pivots:
            coeffs[e, pivots.index(e)] = 1.0
        else:
            # In RREF, column e's entries in the pivot rows give the combo
            coeffs[e] = R[:r, e]
    return pivots, coeffs


# ---------------------------------------------------------------------------
# 2D geometric helpers
# ---------------------------------------------------------------------------

def _cross2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, a, b, tol=1e-9):
    if abs(_cross2d(a, b, p)) > tol:
        return False
    return (min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and
            min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol)


def _points_on_hull_2d(pts, hull_pts):
    """True if point is a hull corner or lies on a hull edge."""
    n_h = len(hull_pts)
    on = np.zeros(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if np.any(np.all(np.abs(hull_pts - p) < 1e-9, axis=1)):
            on[i] = True
            continue
        for j in range(n_h):
            a, b = hull_pts[j], hull_pts[(j + 1) % n_h]
            if _on_segment(p, a, b):
                on[i] = True
                break
    return on


def _find_collinear(hull_pts):
    n = len(hull_pts)
    bad = []
    for i in range(n):
        a, b, c = hull_pts[(i - 1) % n], hull_pts[i], hull_pts[(i + 1) % n]
        if abs(_cross2d(a, b, c)) < 1e-9:
            bad.append(i)
    return bad


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(z, orientations=None):
    """Vertex table with hull status for a Zonotope (any m; 2^m rows)."""
    res = z.analyse(orientations)
    labels, verts, on_hull = res["labels"], res["vertices"], res["on_hull"]

    header = (f"  d = ({', '.join(f'{x:g}' for x in z.d)})"
              f"   q0 = ({', '.join(f'{x:g}' for x in z.q0)})")
    print(header)
    print("  " + "-" * max(len(header) - 2, 50))
    coord_hdr = "  ".join(f"{'Q'+str(k+1):>8}" for k in range(z.dim))
    print(f"  {'#':>4}  {'sigma':^{2*z.m+3}}  {coord_hdr}  status")
    print("  " + "-" * max(len(header) - 2, 50))
    for idx, name in enumerate(labels):
        # recover sigma from label position in default enumeration if possible
        sigma_str = ""
        if orientations is None:
            signs = list(product([+1, -1], repeat=z.m))[idx]
            sigma_str = "(" + ",".join("+" if s > 0 else "-" for s in signs) + ")"
        coords = "  ".join(f"{verts[idx, k]:>8.3f}" for k in range(z.dim))
        status = "hull" if on_hull[idx] else "interior"
        print(f"  {name:>4}  {sigma_str:^{2*z.m+3}}  {coords}  {status}")
    hull_desc = (f"{res['n_sides']}-gon" if z.dim == 2 else
                 f"{res['n_sides']} triangulated facets")
    print(f"\n  Hull: {hull_desc}   "
          f"({on_hull.sum()} of {len(verts)} sigma-points on hull)")
    if z.dim == 2:
        pred = z.predicted_hull_sides()
        print(f"  Predicted from directions: {pred}-gon", end="")
        print("  [matches]" if pred == res["n_sides"] else "  [MISMATCH]")
    if res["collinear"]:
        print(f"  !  {len(res['collinear'])} hull corner(s) collinear "
              f"with neighbors")
    print()


# ---------------------------------------------------------------------------
# Plot (2D)
# ---------------------------------------------------------------------------

def plot(zonotopes, figsize=None, label_vertices=True):
    """
    Plot one or several 2D zonotopes side-by-side.

    Parameters
    ----------
    zonotopes : Zonotope or list of Zonotope
    """
    if isinstance(zonotopes, Zonotope):
        zonotopes = [zonotopes]
    for z in zonotopes:
        if z.dim != 2:
            raise ValueError("plot() supports dim == 2 only")

    n = len(zonotopes)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 5),
                             squeeze=False)

    for ax, z in zip(axes[0], zonotopes):
        res = z.analyse()
        verts, on_hull = res["vertices"], res["on_hull"]
        hull_pts = res["hull_pts"]

        if res["hull"] is not None and len(hull_pts) >= 3:
            poly = plt.Polygon(hull_pts, closed=True,
                               facecolor="#3B8BD422", edgecolor="#185FA5",
                               linewidth=1.5, zorder=1)
            ax.add_patch(poly)

        for i, name in enumerate(res["labels"]):
            color = "#185FA5" if on_hull[i] else "#aaaaaa"
            ax.scatter(*verts[i], color=color, s=50, zorder=3,
                       edgecolors="#0C447C" if on_hull[i] else "#888888",
                       linewidths=0.8)
            if label_vertices and len(verts) <= 64:
                ax.annotate(name, verts[i],
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color="#333333")

        ax.axhline(0, color="#00000018", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="#00000018", linewidth=0.8, linestyle="--")
        d_str = ", ".join(f"{x:g}" for x in z.d)
        ax.set_title(f"m = {z.m},  d = ({d_str})\n"
                     f"{res['n_sides']}-gon  "
                     f"({on_hull.sum()}/{len(verts)} on hull)", fontsize=10)
        ax.set_xlabel("$Q_1$")
        ax.set_ylabel("$Q_2$")
        ax.set_aspect("equal")
        ax.margins(0.2)

        hull_patch = mpatches.Patch(color="#185FA5", label="hull vertex")
        interior_patch = mpatches.Patch(color="#aaaaaa", label="interior")
        ax.legend(handles=[hull_patch, interior_patch], fontsize=8,
                  loc="upper right")

    fig.suptitle("Zonotope convex hull")
    fig.tight_layout()
    return fig
