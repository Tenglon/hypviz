"""Publication export: render a Scene (2D chart views) to vector SVG/PDF via
matplotlib. Mirrors the runtime's drawing semantics and constants
(runtime/src/app/views.ts); accepts a state JSON downloaded from the
interactive page to reproduce a hand-arranged configuration exactly.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .kernel import lorentz as L, mobius as M
from .kernel.charts import CHARTS, Poincare
from .kernel.lorentz import _R
from .scene import LogVector, MobiusSum, Point

S = 8  # absolute spatial half-width of the 3D window (shared domain, see views.ts)
COLORS = {"point": "#2a78d6", "curve": "#52514e", "grid": "#dddcd4", "grid_faint": "#edece5",
          "rim": "#a9a79e", "boundary": "#898781", "surface": "#9ec5f4", "ink": "#0b0b0b"}
FRAME = {"poincare": ((0, 0), 1.08), "klein": ((0, 0), 1.08), "halfplane": ((0, 1.35), 1.55)}


def _polar(k, d, th):
    R = _R(k)
    d, th = np.broadcast_arrays(np.asarray(d, float), np.asarray(th, float))
    return np.stack([R * np.cosh(d / R), R * np.sinh(d / R) * np.cos(th), R * np.sinh(d / R) * np.sin(th)], -1)


def _positions(scene, k, overrides):
    """id -> Lorentz point, honoring state overrides for draggable points."""
    pos = {}
    for o in scene.objects:
        if isinstance(o, Point):
            spatial = np.asarray(overrides.get(o.id)) if o.id in overrides else \
                (o.coords if o.chart == "spatial" else CHARTS[o.chart].to_lorentz(o.coords, k)[1:])
            pos[o.id] = L.from_spatial(spatial, k)
    for o in scene.objects:
        if isinstance(o, MobiusSum):
            pa, pb = (Poincare.from_lorentz(pos[x.id], k) for x in (o.a, o.b))
            pos[o.id] = Poincare.to_lorentz(M.add(pa, pb, k), k)
    return pos


def to_figure(scene, view="poincare", state=None, size=5.5):
    """Render one 2D chart view of the scene to a matplotlib Figure."""
    if isinstance(state, (str, Path)):
        state = json.loads(Path(state).read_text())
    state = state or {}
    k = state.get("curvature", scene.curvature)
    overrides = state.get("spatial", {})
    chart, R, D = CHARTS[view], _R(k), _R(k) * np.arcsinh(S / _R(k))
    pos = _positions(scene, k, overrides)
    proj = lambda x: chart.from_lorentz(x, k)

    fig = Figure(figsize=(size, size))
    ax = fig.add_subplot()
    (cx, cy), hh = FRAME[view]
    ax.set_xlim(cx * R - hh * R, cx * R + hh * R)
    ax.set_ylim(cy * R - hh * R, cy * R + hh * R)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # chart fill + ideal boundary
    th = np.linspace(0, 2 * np.pi, 193)
    if view == "halfplane":
        ax.axhspan(0, (cy + hh) * R, color=COLORS["surface"], alpha=0.16, lw=0)
        ax.axhline(0, color=COLORS["boundary"], lw=1)
    else:
        ax.fill(R * np.cos(th), R * np.sin(th), color=COLORS["surface"], alpha=0.16, lw=0)
        ax.plot(R * np.cos(th), R * np.sin(th), color=COLORS["boundary"], lw=1)

    # polar geodesic grid: shared domain (solid) + context beyond (faint) + rim ring
    def curve(pts, color, lw=0.6):
        q = proj(pts)
        ax.plot(q[:, 0], q[:, 1], color=color, lw=lw)

    for j in range(12):
        curve(_polar(k, np.linspace(0, D, 49), j * np.pi / 6), COLORS["grid"])
        curve(_polar(k, np.linspace(D, 6 * R, 33), j * np.pi / 6), COLORS["grid_faint"])
    for d in np.arange(0.5, 6 * R, 0.5):
        curve(_polar(k, d, np.linspace(0, 2 * np.pi, 97)), COLORS["grid"] if d < D else COLORS["grid_faint"])
    curve(_polar(k, D, np.linspace(0, 2 * np.pi, 97)), COLORS["rim"], lw=0.9)

    # objects (same drawing order as the runtime)
    t = np.linspace(0, 1, 65)
    for o in scene.objects:
        cls = type(o).__name__
        if cls == "Geodesic":
            curve(L.geodesic(pos[o.a.id], pos[o.b.id], t, k), o.color or COLORS["curve"], lw=1.3)
        elif cls == "LogVector":
            x, v = pos[o.base.id], L.logmap(pos[o.base.id], pos[o.to.id], k)
            p0, p1 = proj(x), proj(L.expmap(x, 1e-4 * v, k))
            tip = p0 + (p1 - p0) / 1e-4
            ax.annotate("", xy=tip, xytext=p0,
                        arrowprops={"arrowstyle": "-|>", "color": o.color or COLORS["curve"], "lw": 1.4})
        elif cls == "DistanceLabel":
            mid = proj(L.geodesic(pos[o.a.id], pos[o.b.id], 0.5, k))
            d = float(L.dist(pos[o.a.id], pos[o.b.id], k))
            ax.annotate(f"d = {d:.3f}", mid, textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=9, color=COLORS["ink"])
    for o in scene.objects:
        if isinstance(o, (Point, MobiusSum)):
            p = proj(pos[o.id])
            ax.plot(*p, "o", ms=8, color=o.color or COLORS["point"], zorder=5)
            if o.label:
                ax.annotate(o.label, p, textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=10, color=COLORS["ink"], zorder=6)
    return fig


def save(scene, path, view="poincare", state=None, **kw):
    """Save a vector (svg/pdf) or raster (png) figure; format follows the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_figure(scene, view, state, **kw).savefig(path, bbox_inches="tight")
    return path
