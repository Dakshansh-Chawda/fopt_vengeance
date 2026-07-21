"""
Interactive 3-D plots using Plotly.
Drop-in interactive replacements for the three matplotlib 3-D functions:
  plot_cone_decomposition       -> plot_cone_decomposition_interactive
  plot_3d                       -> plot_3d_interactive
  plot_primal_dual_3d           -> plot_primal_dual_3d_interactive

All figures are fully rotatable / zoomable with the mouse.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from dual_analysis import *
from full_generic_analysis import _assign_cones


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scene_key(flat_idx):
    """Return plotly layout key for the flat subplot index (0-based)."""
    return 'scene' if flat_idx == 0 else f'scene{flat_idx + 1}'


def _hull_traces(pts, hull, fc='rgba(59,139,212,0.18)', ec='#185FA5', name='hull'):
    """
    Return (Mesh3d, Scatter3d-wireframe) traces for a convex hull.

    pts      : (N, 3) array that was passed to ConvexHull
    hull     : ConvexHull object whose simplices index into pts
    """
    i_idx, j_idx, k_idx = hull.simplices.T
    mesh = go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        i=i_idx, j=j_idx, k=k_idx,
        color=fc, opacity=0.18,
        name=name, flatshading=True,
        hoverinfo='skip', showlegend=False,
    )
    # wireframe
    xe, ye, ze = [], [], []
    for s in hull.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            xe += [pts[a, 0], pts[b, 0], None]
            ye += [pts[a, 1], pts[b, 1], None]
            ze += [pts[a, 2], pts[b, 2], None]
    wire = go.Scatter3d(
        x=xe, y=ye, z=ze, mode='lines',
        line=dict(color=ec, width=1.5),
        name=name + ' edges', showlegend=False, hoverinfo='skip',
    )
    return mesh, wire


def _scene_cfg(title_x='Q₁', title_y='Q₂', title_z='Q₃'):
    return dict(
        xaxis_title=title_x, yaxis_title=title_y, zaxis_title=title_z,
        aspectmode='cube',
    )


# ---------------------------------------------------------------------------
# 1. Interactive cone decomposition on S²
# ---------------------------------------------------------------------------

def plot_cone_decomposition_interactive(A, survivors, n_pts=8000, title=None):
    """
    Interactive Plotly version of plot_cone_decomposition() for dim == 3.

    Each surviving cone gets its own colour; hover shows τ coordinates.
    Rotate / zoom with the mouse.

    Parameters
    ----------
    A         : (n_edges, 3) array – generator matrix rows
    survivors : list of sign-vector tuples from find_surviving_sign_vectors
    n_pts     : Fibonacci lattice resolution on S²
    title     : optional figure title
    """
    A  = np.asarray(A, dtype=float)
    nc = len(survivors)

    # Fibonacci lattice on S²
    gold = (1 + 5 ** .5) / 2
    idx  = np.arange(n_pts)
    th   = 2 * np.pi * idx / gold
    ph   = np.arccos(1 - 2 * (idx + .5) / n_pts)
    taus = np.column_stack([np.sin(ph) * np.cos(th),
                             np.sin(ph) * np.sin(th),
                             np.cos(ph)])
    assign = _assign_cones(taus, A, survivors)

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    traces = []
    for i in range(nc):
        m = assign == i
        if not m.any():
            continue
        c = palette[i % len(palette)]
        traces.append(go.Scatter3d(
            x=taus[m, 0], y=taus[m, 1], z=taus[m, 2],
            mode='markers',
            marker=dict(size=3, color=c, opacity=0.85),
            name=f'cone {i + 1}',
            hovertemplate='τ = (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra>cone %{fullData.name}</extra>',
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title or f'Cone decomposition on S²  ({nc} surviving cones)',
        scene=_scene_cfg('τ₁', 'τ₂', 'τ₃'),
        legend=dict(itemsizing='constant'),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Interactive 3-D zonotope hull
# ---------------------------------------------------------------------------

def plot_3d_interactive(zonotopes, orientations=None, label_vertices=True):
    """
    Interactive Plotly version of plot_3d() for dim == 3.

    Hull is drawn as a translucent mesh; hull corners in blue, interior
    sigma-points in grey.  Hover shows vertex label + coordinates.

    Parameters
    ----------
    zonotopes    : Zonotope or list[Zonotope]  (dim must be 3)
    orientations : optional sign-vector list
    label_vertices : show sigma labels next to points (≤ 64 vertices)
    """
    if isinstance(zonotopes, Zonotope):
        zonotopes = [zonotopes]
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

        # hull mesh + wireframe
        if res['hull'] is not None and len(upts) >= 4:
            mesh, wire = _hull_traces(upts, res['hull'])
            fig.add_trace(mesh, row=1, col=col)
            fig.add_trace(wire, row=1, col=col)

        # hull corners (blue)
        h_v  = verts[on_hull]
        h_nm = [nm for nm, o in zip(res['labels'], on_hull) if o]
        if len(h_v):
            fig.add_trace(go.Scatter3d(
                x=h_v[:, 0], y=h_v[:, 1], z=h_v[:, 2],
                mode='markers' + ('+text' if label_vertices and len(verts) <= 64 else ''),
                marker=dict(size=5, color='#185FA5',
                            line=dict(color='#0C447C', width=1)),
                text=h_nm,
                textposition='top center', textfont=dict(size=8),
                name='hull vertex', showlegend=(col == 1),
                hovertemplate='%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra>hull</extra>',
            ), row=1, col=col)

        # interior sigma-points (grey)
        i_v  = verts[~on_hull]
        i_nm = [nm for nm, o in zip(res['labels'], on_hull) if not o]
        if len(i_v):
            fig.add_trace(go.Scatter3d(
                x=i_v[:, 0], y=i_v[:, 1], z=i_v[:, 2],
                mode='markers',
                marker=dict(size=3, color='#aaaaaa', opacity=0.5),
                text=i_nm,
                name='interior', showlegend=(col == 1),
                hovertemplate='%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra>interior</extra>',
            ), row=1, col=col)

        fig.update_layout(**{scene: _scene_cfg()})

    fig.update_layout(title='Zonotope convex hull (3-D, interactive)',
                      height=600)
    return fig


# ---------------------------------------------------------------------------
# 3. Interactive 3-D primal + polar dual
# ---------------------------------------------------------------------------

def plot_primal_dual_3d_interactive(cases, orientations=None, titles=None):
    """
    Interactive Plotly version of plot_primal_dual_3d().

    Two columns per row (P left, P* right); each panel can be rotated
    and zoomed independently.

    Parameters
    ----------
    cases        : Zonotope or list[Zonotope | (n, 3) ndarray]
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

    p_ec, p_fc = '#185FA5', 'rgba(59,139,212,0.18)'
    d_ec, d_fc = '#A5471F', 'rgba(212,112,59,0.18)'

    for r, case in enumerate(cases, start=1):
        if isinstance(case, Zonotope):
            res        = case.analyse(orientations)
            primal_pts = res['unique_pts']
        else:
            primal_pts = np.asarray(case, dtype=float)

        center   = primal_pts.mean(axis=0)
        dual_pts = polar_dual(primal_pts, center=center)
        p_hull   = ConvexHull(primal_pts)
        d_hull   = ConvexHull(dual_pts)

        for col, (pts, hull, ec, fc, tag) in enumerate([
            (primal_pts, p_hull, p_ec, p_fc, 'P'),
            (dual_pts,   d_hull, d_ec, d_fc, 'P*'),
        ], start=1):
            scene = _scene_key((r - 1) * 2 + (col - 1))

            mesh, wire = _hull_traces(pts, hull, fc=fc, ec=ec, name=tag)
            fig.add_trace(mesh, row=r, col=col)
            fig.add_trace(wire, row=r, col=col)

            corners = pts[hull.vertices]
            labels  = [f'q{i+1}' for i in range(len(corners))]
            fig.add_trace(go.Scatter3d(
                x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
                mode='markers+text',
                marker=dict(size=5, color=ec,
                            line=dict(color='black', width=0.5)),
                text=labels, textposition='top center',
                textfont=dict(size=8),
                name=f'{tag} vertices', showlegend=False,
                hovertemplate='%{text}<br>(%{x:.4f}, %{y:.4f}, %{z:.4f})<extra></extra>',
            ), row=r, col=col)

            fig.update_layout(**{scene: _scene_cfg()})

    fig.update_layout(
        title='3-D zonotope P and polar dual P*  (interactive — rotate each panel independently)',
        height=620 * n, width= 1240 * n
    )
    return fig
