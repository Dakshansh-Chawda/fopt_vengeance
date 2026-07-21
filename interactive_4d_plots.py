"""
Interactive 4-D plots using Plotly.
Companion to interactive_3d_plots.py for dim == 4: the first three
coordinates are plotted as x/y/z, and the 4th coordinate is encoded as a
colour heatmap (marker colour / mesh intensity) rather than a spatial axis.

  plot_cone_decomposition       -> plot_cone_decomposition_interactive_4d
  plot_3d                       -> plot_4d_interactive
  plot_primal_dual_3d           -> plot_primal_dual_4d_interactive

All figures are fully rotatable / zoomable with the mouse; the projected
hull is the 3-D "shadow" of the 4-D polytope (like a tesseract projection),
so some visible edges/faces are internal structure rather than an actual
3-D boundary.
"""
import itertools

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dual_analysis import *
from full_generic_analysis import _assign_cones

# Sequential "blue" ramp (light -> dark), single hue, from the shared
# data-viz palette -- used for every 4th-dimension colour encoding here.
_SEQ_STEPS = [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
    '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
]
HEATMAP_SCALE = [[i / (len(_SEQ_STEPS) - 1), c] for i, c in enumerate(_SEQ_STEPS)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scene_key(flat_idx):
    """Return plotly layout key for the flat subplot index (0-based)."""
    return 'scene' if flat_idx == 0 else f'scene{flat_idx + 1}'


def _scene_cfg(title_x='Q₁', title_y='Q₂', title_z='Q₃'):
    return dict(
        xaxis_title=title_x, yaxis_title=title_y, zaxis_title=title_z,
        aspectmode='cube',
    )


def _tetra_edges(simplices):
    """Unique edges (i, j) across all 4-D hull simplices (tetrahedra)."""
    edges = set()
    for s in simplices:
        for e in itertools.combinations(sorted(s), 2):
            edges.add(e)
    return edges


def _tetra_faces(simplices):
    """
    Triangular faces of every tetrahedron, for a Mesh3d 'shadow' of the
    4-D hull projected onto its first 3 coordinates. Internal faces of the
    4-D hull get drawn too (there is no clean outer/inner split under
    projection) -- same convention as a tesseract-shadow drawing.
    """
    faces = []
    for s in simplices:
        faces.extend(itertools.combinations(s, 3))
    return np.array(faces)


def _hull4d_traces(pts, hull, name='hull', showscale=True, cmin=None, cmax=None):
    """
    Return (Mesh3d, Scatter3d-wireframe) traces for the projection of a
    4-D convex hull onto its first three coordinates. The 4th coordinate
    is encoded as a colour heatmap via Mesh3d vertex `intensity`.

    pts  : (N, 4) array that was passed to ConvexHull
    hull : ConvexHull object (dim == 4) whose simplices are tetrahedra
    """
    w = pts[:, 3]
    faces = _tetra_faces(hull.simplices)
    i_idx, j_idx, k_idx = faces.T
    mesh = go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=i_idx, j=j_idx, k=k_idx,
        intensity=w, intensitymode='vertex',
        colorscale=HEATMAP_SCALE, cmin=cmin, cmax=cmax,
        showscale=showscale, colorbar=dict(title='Q₄') if showscale else None,
        opacity=0.35, name=name, flatshading=True,
        hoverinfo='skip', showlegend=False,
    )
    xe, ye, ze = [], [], []
    for a, b in _tetra_edges(hull.simplices):
        xe += [pts[a, 0], pts[b, 0], None]
        ye += [pts[a, 1], pts[b, 1], None]
        ze += [pts[a, 2], pts[b, 2], None]
    wire = go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color='#52514e', width=1.2),
        name=name + ' edges', showlegend=False, hoverinfo='skip',
    )
    return mesh, wire


# ---------------------------------------------------------------------------
# 1. Interactive cone decomposition on S³ (unit hypersphere)
# ---------------------------------------------------------------------------

def plot_cone_decomposition_interactive_4d(A, survivors, n_pts=20000, title=None, seed=42):
    """
    Interactive Plotly version of plot_cone_decomposition() for dim == 4.

    Samples the unit 3-sphere S³ (normalised Gaussian vectors -- the
    direct generalisation of Marsaglia's method to any dimension),
    assigns each sample to a surviving cone, and plots tau1..tau3 as
    x/y/z with tau4 shown as a colour heatmap. One trace per cone keeps
    cones independently toggleable via the legend; hover shows the full
    4-vector and the assigned cone.

    Parameters
    ----------
    A         : (n_edges, 4) array – generator matrix rows
    survivors : list of sign-vector tuples from find_surviving_sign_vectors
    n_pts     : number of S³ sample points
    seed      : RNG seed for the sample (deterministic by default)
    title     : optional figure title
    """
    A = np.asarray(A, dtype=float)
    if A.shape[1] != 4:
        raise ValueError(f'plot_cone_decomposition_interactive_4d requires dim == 4, got {A.shape[1]}')
    nc = len(survivors)

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_pts, 4))
    taus = g / np.linalg.norm(g, axis=1, keepdims=True)

    assign = _assign_cones(taus, A, survivors)

    traces = []
    for i in range(nc):
        m = assign == i
        if not m.any():
            continue
        traces.append(go.Scatter3d(
            x=taus[m, 0], y=taus[m, 1], z=taus[m, 2],
            mode='markers',
            marker=dict(size=3, color=taus[m, 3], colorscale=HEATMAP_SCALE,
                        cmin=-1, cmax=1, opacity=0.85,
                        showscale=(len(traces) == 0),
                        colorbar=dict(title='τ₄') if len(traces) == 0 else None),
            name=f'cone {i + 1}',
            hovertemplate=('τ = (%{x:.3f}, %{y:.3f}, %{z:.3f}, '
                           '%{marker.color:.3f})<extra>cone %{fullData.name}</extra>'),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title or f'Cone decomposition on S³  ({nc} surviving cones, τ₄ as heatmap)',
        scene=_scene_cfg('τ₁', 'τ₂', 'τ₃'),
        legend=dict(itemsizing='constant'),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Interactive 4-D zonotope hull
# ---------------------------------------------------------------------------

def plot_4d_interactive(zonotopes, orientations=None, label_vertices=True):
    """
    Interactive Plotly version of plot_3d_interactive() for dim == 4.

    Vertices are projected onto their first three coordinates (Q1, Q2,
    Q3) for x/y/z; Q4 is encoded as a colour heatmap on both the hull
    mesh (vertex intensity) and the vertex markers, sharing one colour
    scale per zonotope. Hull corners get an opaque marker outline,
    interior sigma-points are faded.

    Parameters
    ----------
    zonotopes    : Zonotope or list[Zonotope]  (dim must be 4)
    orientations : optional sign-vector list
    label_vertices : show sigma labels next to points (≤ 64 vertices)
    """
    if isinstance(zonotopes, Zonotope):
        zonotopes = [zonotopes]
    for z in zonotopes:
        if z.dim != 4:
            raise ValueError('plot_4d_interactive requires dim == 4')
    n = len(zonotopes)

    specs = [[{'type': 'scene'} for _ in range(n)]]
    subtitles = []
    for z in zonotopes:
        d_str = ', '.join(f'{x:g}' for x in z.d)
        subtitles.append(f'm={z.m}, d=({d_str})')

    fig = make_subplots(rows=1, cols=n, specs=specs,
                        subplot_titles=subtitles,
                        horizontal_spacing=0.02)

    for col, z in enumerate(zonotopes, start=1):
        res = z.analyse(orientations)
        verts, on_hull = res['vertices'], res['on_hull']
        upts = res['unique_pts']
        scene = _scene_key(col - 1)

        w_all = verts[:, 3]
        cmin, cmax = (w_all.min(), w_all.max()) if len(verts) else (0, 1)

        if res['hull'] is not None and len(upts) >= 5:
            mesh, wire = _hull4d_traces(upts, res['hull'], showscale=(col == 1),
                                        cmin=cmin, cmax=cmax)
            fig.add_trace(mesh, row=1, col=col)
            fig.add_trace(wire, row=1, col=col)

        h_v = verts[on_hull]
        h_nm = [nm for nm, o in zip(res['labels'], on_hull) if o]
        if len(h_v):
            fig.add_trace(go.Scatter3d(
                x=h_v[:, 0], y=h_v[:, 1], z=h_v[:, 2],
                mode='markers' + ('+text' if label_vertices and len(verts) <= 64 else ''),
                marker=dict(size=5, color=h_v[:, 3], colorscale=HEATMAP_SCALE,
                            cmin=cmin, cmax=cmax, showscale=False,
                            line=dict(color='#0b0b0b', width=1)),
                text=h_nm,
                textposition='top center', textfont=dict(size=8),
                name='hull vertex', showlegend=(col == 1),
                hovertemplate=('%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}, '
                               '%{marker.color:.3f})<extra>hull</extra>'),
            ), row=1, col=col)

        i_v = verts[~on_hull]
        i_nm = [nm for nm, o in zip(res['labels'], on_hull) if not o]
        if len(i_v):
            fig.add_trace(go.Scatter3d(
                x=i_v[:, 0], y=i_v[:, 1], z=i_v[:, 2],
                mode='markers',
                marker=dict(size=3, color=i_v[:, 3], colorscale=HEATMAP_SCALE,
                            cmin=cmin, cmax=cmax, opacity=0.45, showscale=False),
                text=i_nm,
                name='interior', showlegend=(col == 1),
                hovertemplate=('%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}, '
                               '%{marker.color:.3f})<extra>interior</extra>'),
            ), row=1, col=col)

        fig.update_layout(**{scene: _scene_cfg()})

    fig.update_layout(title='Zonotope convex hull (4-D projection, Q₄ as heatmap, interactive)',
                      height=600)
    return fig


# ---------------------------------------------------------------------------
# 3. Interactive 4-D primal + polar dual
# ---------------------------------------------------------------------------

def plot_primal_dual_4d_interactive(cases, orientations=None, titles=None):
    """
    Interactive Plotly version of plot_primal_dual_3d_interactive() for
    dim == 4.

    Two columns per row (P left, P* right); each panel is the projection
    of the 4-D hull onto its first three coordinates, with the 4th
    coordinate shown as a colour heatmap. Panels can be rotated/zoomed
    independently.

    Parameters
    ----------
    cases        : Zonotope or list[Zonotope | (n, 4) ndarray]
    orientations : optional sign-vector list forwarded to Zonotope.analyse()
    titles       : optional row-label strings
    """
    if not isinstance(cases, (list, tuple)):
        cases = [cases]
    n = len(cases)

    specs     = [[{'type': 'scene'}, {'type': 'scene'}] for _ in range(n)]
    row_ttls  = titles or ['' for _ in range(n)]
    subtitles = []
    for r, case in enumerate(cases):
        pfx = (row_ttls[r] + ' — ') if row_ttls[r] else ''
        subtitles += [pfx + 'P  (primal)', pfx + 'P*  (polar dual)']

    fig = make_subplots(rows=n, cols=2, specs=specs,
                        subplot_titles=subtitles,
                        horizontal_spacing=0.04,
                        vertical_spacing=0.06)

    for r, case in enumerate(cases, start=1):
        if isinstance(case, Zonotope):
            if case.dim != 4:
                raise ValueError('plot_primal_dual_4d_interactive requires dim == 4')
            res        = case.analyse(orientations)
            primal_pts = res['unique_pts']
        else:
            primal_pts = np.asarray(case, dtype=float)
            if primal_pts.shape[1] != 4:
                raise ValueError('plot_primal_dual_4d_interactive requires dim == 4')

        center   = primal_pts.mean(axis=0)
        dual_pts = polar_dual(primal_pts, center=center)
        p_hull   = ConvexHull(primal_pts)
        d_hull   = ConvexHull(dual_pts)

        for col, (pts, hull, tag) in enumerate([
            (primal_pts, p_hull, 'P'),
            (dual_pts,   d_hull, 'P*'),
        ], start=1):
            scene = _scene_key((r - 1) * 2 + (col - 1))
            w = pts[:, 3]

            mesh, wire = _hull4d_traces(pts, hull, name=tag,
                                        showscale=True, cmin=w.min(), cmax=w.max())
            fig.add_trace(mesh, row=r, col=col)
            fig.add_trace(wire, row=r, col=col)

            corners = pts[hull.vertices]
            labels  = [f'q{i+1}' for i in range(len(corners))]
            fig.add_trace(go.Scatter3d(
                x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                mode='markers+text',
                marker=dict(size=5, color=corners[:, 3], colorscale=HEATMAP_SCALE,
                            cmin=w.min(), cmax=w.max(), showscale=False,
                            line=dict(color='#0b0b0b', width=0.5)),
                text=labels, textposition='top center',
                textfont=dict(size=8),
                name=f'{tag} vertices', showlegend=False,
                hovertemplate=('%{text}<br>(%{x:.4f}, %{y:.4f}, %{z:.4f}, '
                               '%{marker.color:.4f})<extra></extra>'),
            ), row=r, col=col)

            fig.update_layout(**{scene: _scene_cfg()})

    fig.update_layout(
        title='4-D zonotope P and polar dual P*  (projection, Q₄ as heatmap, interactive)',
        height=620 * n, width=1240 * n
    )
    return fig
