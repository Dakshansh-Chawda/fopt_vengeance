"""
3-D extensions: cone decomposition, zonotope hull, polar dual.
Supports dim <= 3; dim == 2 delegates to the existing plot() / plot_primal_dual().
"""
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401  (activates 3-D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dual_analysis import *

# ---------------------------------------------------------------------------
# Shared 3-D helpers
# ---------------------------------------------------------------------------

def _assign_cones(taus, A, survivors, tol=1e-9):
    """
    Vectorised cone assignment.

    For each row of *taus* (unit vectors), return the index of the first
    surviving cone containing it, or -1 if none.

    Key identity: tau in K_sigma  <=>  sigma_e * (A[e] . tau) >= 0  for all e.
    taus @ A.T gives all dot products in one matrix multiply.
    """
    proj   = taus @ A.T                                    # (n_pts, n_edges)
    assign = np.full(len(taus), -1, dtype=int)
    for i, sigma in enumerate(survivors):
        s  = np.asarray(sigma, dtype=float)                # (n_edges,)
        un = np.where(assign == -1)[0]
        if not un.size:
            break
        in_cone = (s * proj[un] >= -tol).all(axis=1)      # (len(un),)
        assign[un[in_cone]] = i
    return assign


def _draw_hull_3d(ax, pts, simplices,
                  fc='#3B8BD422', ec='#185FA5', alpha=0.25, lw=0.4):
    """Add a translucent Poly3DCollection built from a simplex list."""
    ax.add_collection3d(
        Poly3DCollection([pts[s] for s in simplices],
                         alpha=alpha, facecolor=fc, edgecolor=ec, linewidth=lw))


def _equalise_3d(ax, pts):
    """Set equal data-range scales for a 3-D Axes, centred on *pts*."""
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    h = max((hi - lo).max() / 2 * 1.15, 1e-6)
    for fn, ci in zip([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], c):
        fn(ci - h, ci + h)


# ---------------------------------------------------------------------------
# Cone decomposition  (dim == 2 or 3)
# ---------------------------------------------------------------------------

def plot_cone_decomposition(A, survivors, n_pts=5000, figsize=None, title=None):
    """
    Visualise surviving cones on the unit circle (dim=2) or sphere (dim=3).

    Each surviving cone gets a distinct colour from tab20.
    Boundary hyperplanes are drawn as diameters (2-D) or omitted (3-D).

    Parameters
    ----------
    A         : (n_edges, dim) array – same matrix as find_surviving_sign_vectors
    survivors : list of sign-vector tuples – output of find_surviving_sign_vectors
    n_pts     : Fibonacci lattice size on S^2 (3-D only)
    """
    A    = np.asarray(A, dtype=float)
    dim  = A.shape[1]
    nc   = len(survivors)
    cmap = plt.get_cmap('tab20', max(nc, 1))

    # ---- 2-D: unit circle -------------------------------------------------
    if dim == 2:
        th     = np.linspace(0, 2 * np.pi, 5001, endpoint=False)
        pts    = np.column_stack([np.cos(th), np.sin(th)])
        assign = _assign_cones(pts, A, survivors)

        fig, ax = plt.subplots(figsize=figsize or (6, 6))
        for i in range(nc):
            m = assign == i
            if m.any():
                ax.scatter(pts[m, 0], pts[m, 1],
                           color=cmap(i), s=6, linewidths=0)
        for row in A:
            n_vec = row / np.linalg.norm(row)
            perp  = np.array([-n_vec[1], n_vec[0]])
            ax.plot([-1.3 * perp[0], 1.3 * perp[0]],
                    [-1.3 * perp[1], 1.3 * perp[1]],
                    'k-', lw=0.7, alpha=0.4)
        ax.set_aspect('equal')
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel(r'$\tau_1$'); ax.set_ylabel(r'$\tau_2$')
        ax.set_title(title or f'Cone decomposition  ({nc} surviving cones)')
        return fig

    # ---- 3-D: unit sphere via Fibonacci lattice ---------------------------
    elif dim == 3:
        gold = (1 + 5 ** .5) / 2
        idx  = np.arange(n_pts)
        th   = 2 * np.pi * idx / gold
        ph   = np.arccos(1 - 2 * (idx + .5) / n_pts)
        taus = np.column_stack([np.sin(ph) * np.cos(th),
                                np.sin(ph) * np.sin(th),
                                np.cos(ph)])
        assign = _assign_cones(taus, A, survivors)

        fig = plt.figure(figsize=figsize or (8, 7))
        ax  = fig.add_subplot(111, projection='3d')
        for i in range(nc):
            m = assign == i
            if m.any():
                ax.scatter(taus[m, 0], taus[m, 1], taus[m, 2],
                           color=cmap(i), s=10, alpha=0.85, linewidths=0)
        ax.set_xlabel(r'$\tau_1$'); ax.set_ylabel(r'$\tau_2$')
        ax.set_zlabel(r'$\tau_3$')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(title or
                     f'Cone decomposition on $S^2$  ({nc} surviving cones)')
        return fig

    else:
        raise ValueError(f'dim must be 2 or 3, got {dim}')


# ---------------------------------------------------------------------------
# 3-D zonotope hull
# ---------------------------------------------------------------------------

def plot_3d(zonotopes, orientations=None, figsize=None, label_vertices=True):
    """
    Plot one or several zonotopes side-by-side.

    * dim == 2: delegates to the existing 2-D plot().
    * dim == 3: draws the convex hull as a translucent Poly3DCollection;
                hull corners in blue, interior sigma-points in grey.

    Parameters
    ----------
    zonotopes    : Zonotope or list[Zonotope]
    orientations : optional sign-vector list (subset of sigma to consider);
                   None uses all 2^m sign vectors
    """
    if isinstance(zonotopes, Zonotope):
        zonotopes = [zonotopes]
    if all(z.dim == 2 for z in zonotopes):
        return plot(zonotopes, figsize=figsize, label_vertices=label_vertices)
    for z in zonotopes:
        if z.dim != 3:
            raise ValueError('plot_3d: all zonotopes must have dim == 2 or 3')

    n   = len(zonotopes)
    fig = plt.figure(figsize=figsize or (6 * n, 6))
    for col, z in enumerate(zonotopes, start=1):
        ax  = fig.add_subplot(1, n, col, projection='3d')
        res = z.analyse(orientations)
        verts, on_hull = res['vertices'], res['on_hull']
        upts = res['unique_pts']

        if res['hull'] is not None and len(upts) >= 4:
            _draw_hull_3d(ax, upts, res['hull'].simplices)

        h_v, i_v = verts[on_hull], verts[~on_hull]
        if len(h_v):
            ax.scatter(*h_v.T, color='#185FA5', s=40, zorder=3,
                       edgecolors='#0C447C', linewidths=0.7)
        if len(i_v):
            ax.scatter(*i_v.T, color='#aaaaaa', s=18, alpha=0.5)
        if label_vertices and len(verts) <= 64:
            for i, nm in enumerate(res['labels']):
                ax.text(*verts[i], nm, fontsize=6, color='#444')

        _equalise_3d(ax, upts if len(upts) else verts)
        d_str = ', '.join(f'{x:g}' for x in z.d)
        ax.set_title(f"m={z.m}, d=({d_str})\n"
                     f"{res['n_sides']} facets, "
                     f"{on_hull.sum()}/{len(verts)} on hull", fontsize=9)
        ax.set_xlabel('$Q_1$'); ax.set_ylabel('$Q_2$'); ax.set_zlabel('$Q_3$')

    fig.suptitle('Zonotope convex hull (3-D)')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3-D primal + polar dual
# ---------------------------------------------------------------------------

def plot_primal_dual_3d(cases, orientations=None, figsize=None, titles=None):
    """
    Side-by-side 3-D plots of P (zonotope hull) and its polar dual P*.

    The dual centre is taken as the centroid of the primal hull vertices,
    guaranteeing it is always interior regardless of q0 placement.

    Parameters
    ----------
    cases        : Zonotope or list[Zonotope | (n, 3) ndarray]
    orientations : optional sign-vector list forwarded to Zonotope.analyse()
    titles       : optional row-label strings
    """
    if not isinstance(cases, (list, tuple)):
        cases = [cases]
    n   = len(cases)
    fig = plt.figure(figsize=figsize or (12, 6 * n))

    for r, case in enumerate(cases):
        if isinstance(case, Zonotope):
            res        = case.analyse(orientations)
            primal_pts = res['unique_pts']
        else:
            primal_pts = np.asarray(case, dtype=float)

        center   = primal_pts.mean(axis=0)           # centroid is always interior
        dual_pts = polar_dual(primal_pts, center=center)
        p_hull   = ConvexHull(primal_pts)
        d_hull   = ConvexHull(dual_pts)

        for col, (pts, hull, fc, ec, tag) in enumerate([
            (primal_pts, p_hull, '#3B8BD422', '#185FA5', 'P'),
            (dual_pts,   d_hull, '#D4703B22', '#A5471F', 'P*'),
        ], start=1):
            ax = fig.add_subplot(n, 2, 2 * r + col, projection='3d')
            _draw_hull_3d(ax, pts, hull.simplices, fc, ec)
            corners = pts[hull.vertices]
            ax.scatter(*corners.T, color=ec, s=45, zorder=3)
            for i, p in enumerate(corners):
                ax.text(*p, f'$q_{{{i+1}}}$', fontsize=7)
            _equalise_3d(ax, pts)
            ax.set_xlabel('$Q_1$'); ax.set_ylabel('$Q_2$')
            ax.set_zlabel('$Q_3$')
            ax.set_title(f'{tag}  ({len(corners)} vertices, '
                         f'Vol = {hull.volume:.4g})', fontsize=9)

        if titles and r < len(titles):
            fig.text(0.01, 1 - (r + .5) / n, titles[r],
                     va='center', rotation=90, fontsize=11)

    fig.suptitle('3-D zonotope  P  and polar dual  P*\n', fontsize=12)
    fig.tight_layout()
    return fig