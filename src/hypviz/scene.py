"""Scene authoring: declarative primitives compiled to scene JSON, rendered by
the bundled TS runtime as a self-contained interactive HTML page.

Primitives reference each other directly (Geodesic(a, b)); the dependency
graph the runtime needs is just those id references.
"""
import json
from itertools import count
from pathlib import Path

import numpy as np

from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS

_STATIC = Path(__file__).parent / "_static"
_ids = count()


class _Obj:
    def __init__(self):
        self.id = f"o{next(_ids)}"


class Point(_Obj):
    """A point in H^2, given in `chart` coordinates ("spatial" = Lorentz spatial).
    Conversion happens at export time, with the scene's curvature."""

    def __init__(self, coords, chart="poincare", draggable=False, label=None, color=None):
        super().__init__()
        self.coords, self.chart = as_numpy(coords), chart
        self.draggable, self.label, self.color = draggable, label, color

    def to_json(self, k):
        spatial = self.coords if self.chart == "spatial" else CHARTS[self.chart].to_lorentz(self.coords, k)[1:]
        return {"id": self.id, "type": "point", "spatial": spatial.tolist(),
                "draggable": self.draggable, "label": self.label, "color": self.color}


class Geodesic(_Obj):
    def __init__(self, a, b, color=None):
        super().__init__()
        self.a, self.b, self.color = a, b, color

    def to_json(self, k):
        return {"id": self.id, "type": "geodesic", "from": self.a.id, "to": self.b.id, "color": self.color}


class DistanceLabel(_Obj):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def to_json(self, k):
        return {"id": self.id, "type": "distance_label", "from": self.a.id, "to": self.b.id}


class LogVector(_Obj):
    """The tangent vector log_base(to), drawn as a straight arrow at `base`."""

    def __init__(self, base, to, color=None):
        super().__init__()
        self.base, self.to, self.color = base, to, color

    def to_json(self, k):
        return {"id": self.id, "type": "log_vector", "base": self.base.id, "to": self.to.id, "color": self.color}


class MobiusSum(_Obj):
    """The derived point a (+) b (Mobius addition); usable wherever a Point is."""

    def __init__(self, a, b, label=None, color=None):
        super().__init__()
        self.a, self.b, self.label, self.color = a, b, label, color

    def to_json(self, k):
        return {"id": self.id, "type": "mobius_sum", "a": self.a.id, "b": self.b.id,
                "label": self.label, "color": self.color}


class TangentPlane(_Obj):
    """A translucent patch of the tangent plane at a point (3D views only)."""

    def __init__(self, at):
        super().__init__()
        self.at = at

    def to_json(self, k):
        return {"id": self.id, "type": "tangent_plane", "at": self.at.id}


class MetricCircle(_Obj):
    """The metric circle {|v|_hyp = radius} in the tangent space at a point,
    drawn through each view's differential — a Euclidean circle that shrinks
    near the disk's rim (the conformal factor made visible); a tilted circle
    inside the tangent plane in 3D views."""

    def __init__(self, at, radius=0.35, color=None):
        super().__init__()
        self.at, self.radius, self.color = at, radius, color

    def to_json(self, k):
        return {"id": self.id, "type": "metric_circle", "at": self.at.id,
                "radius": self.radius, "color": self.color}


_DEFAULT_HINT = ("Drag the highlighted points — in either view; the 3D view orbits and zooms with the mouse. "
                 "The gray ring marks the edge of the 3D window: the disk beyond it (faint grid) continues to "
                 "infinity, which no isometric embedding can show — dragging stops at the ring in both views.")


class EntailmentCone(_Obj):
    """A hyperbolic entailment cone (Ganea et al. 2018): the cone rooted at `apex`,
    opening radially outward with half-aperture ψ = arcsin(K(1−ρ²)/ρ) (ρ = the
    normalized ball norm) — so deeper points have narrower cones. Points inside are
    'entailed by' the apex (it is their ancestor). The runtime fills the cone and,
    for the optional `test` point, colors the geodesic to it by membership."""

    def __init__(self, apex, test=None, aperture=0.1, fill="#9ec5f4", edge="#2a78d6",
                 yes="#1baf7a", no="#e34948"):
        super().__init__()
        self.apex, self.test, self.aperture = apex, test, aperture
        self.colors = {"fill": fill, "edge": edge, "yes": yes, "no": no}

    def to_json(self, k):
        d = {"id": self.id, "type": "entailment_cone", "apex": self.apex.id,
             "aperture": self.aperture, "colors": self.colors}
        if self.test is not None:
            d["test"] = self.test.id
        return d


class Gyroplane(_Obj):
    """A gyroplane — the hyperbolic decision hyperplane of Ganea et al. (2018):
    the geodesic through `p` normal to the gyrovector w = ⊖p ⊕ (normal handle).
    The runtime draws the boundary, its equidistant contours (hypercycles = MLR
    confidence level sets, colored by side), the normal gyrovector with gyro-scalar
    ruler ticks, and — for the optional `test` point — its signed distance (logit)."""

    def __init__(self, p, normal, test=None, levels=(0.7, 1.4), plane="#52514e",
                 pos="#2a78d6", neg="#1baf7a", normal_color="#4a3aa7", perp="#898781"):
        super().__init__()
        self.p, self.normal, self.test, self.levels = p, normal, test, list(levels)
        self.colors = {"plane": plane, "pos": pos, "neg": neg, "normal": normal_color, "perp": perp}

    def to_json(self, k):
        d = {"id": self.id, "type": "gyroplane", "p": self.p.id, "normal": self.normal.id,
             "levels": self.levels, "colors": self.colors}
        if self.test is not None:
            d["test"] = self.test.id
        return d


class TransportLoop(_Obj):
    """Parallel-transport a unit tangent vector around the closed geodesic loop
    through `points`. The runtime draws the vector at each vertex and the rotated
    vector back at the start, and labels the holonomy angle (= the loop's area =
    the angle deficit). Drag the vertices / slide the curvature to change it."""

    def __init__(self, points, initial="#2a78d6", mid="#1baf7a", returned="#e34948"):
        super().__init__()
        self.points = points
        self.colors = {"initial": initial, "mid": mid, "returned": returned}

    def to_json(self, k):
        return {"id": self.id, "type": "transport_loop",
                "points": [p.id for p in self.points], "colors": self.colors}


class DensityField(_Obj):
    """A precomputed density heatmap rendered as a texture on `chart`'s plane, with
    one texture per metric (data URIs). Smooth (bilinear) and zoomable; the metric
    is switchable in the page."""

    def __init__(self, chart, extent, textures, metric, surface=False, points=None, view=0, curvature=False):
        super().__init__()
        self.chart, self.extent, self.textures, self.metric = chart, list(extent), textures, metric
        self.surface, self.points, self.view, self.curvature = surface, points, view, curvature

    def to_json(self, k):
        d = {"id": self.id, "type": "density", "chart": self.chart, "extent": self.extent,
             "textures": self.textures, "metric": self.metric, "surface": self.surface,
             "view": self.view, "curvature": self.curvature}
        if self.points is not None:
            d["points"] = self.points
        return d


class Cloud(_Obj):
    """A bulk point cloud (typically a sampled Hierarchy): batch-rendered points
    with per-point color and hover label, tree edges derived from `parent`, and
    per-node pruned-leaf counts for honest hover tooltips ('+N leaves not shown').
    `coords` are 2D Lorentz points (N, 3)."""

    def __init__(self, coords, colors, labels=None, parent=None, pruned=None):
        super().__init__()
        self.spatial = as_numpy(coords)[:, 1:]
        self.colors = list(colors)
        self.labels = None if labels is None else [str(x) for x in labels]
        self.parent = None if parent is None else as_numpy(parent).astype(int)
        self.pruned = None if pruned is None else as_numpy(pruned).astype(int)

    def to_json(self, k):
        d = {"id": self.id, "type": "cloud",
             "spatial": np.round(self.spatial, 6).tolist(), "colors": self.colors}
        if self.labels is not None:
            d["labels"] = self.labels
        if self.parent is not None:
            d["parent"] = self.parent.tolist()
        if self.pruned is not None:
            d["pruned"] = self.pruned.tolist()
        return d


class Scene:
    def __init__(self, objects, views=("poincare", "lorentz"), curvature=-1.0, curvature_slider=False,
                 hint=None, legend=(), density_curvatures=None):
        self.objects, self.views = list(objects), list(views)
        self.curvature, self.curvature_slider = curvature, curvature_slider
        self.density_curvatures = density_curvatures
        self.hint = hint or _DEFAULT_HINT
        self.legend = [{"kind": k, "color": c, "label": l} for k, c, l in legend]  # (kind, color, label)

    def to_json(self):
        return {"views": [v if isinstance(v, dict) else {"chart": v} for v in self.views],
                "objects": [o.to_json(self.curvature) for o in self.objects],
                "curvature": self.curvature, "curvatureSlider": self.curvature_slider,
                "densityCurvatures": self.density_curvatures, "legend": self.legend}

    def html(self, title="hypviz"):
        """The self-contained page as a string (inlined runtime + scene)."""
        return ((_STATIC / "template.html").read_text()
                .replace("/*TITLE*/", title)
                .replace("/*HINT*/", self.hint)
                .replace("/*RUNTIME*/", (_STATIC / "hypviz-runtime.js").read_text())
                .replace("/*SCENE*/", json.dumps(self.to_json())))

    def to_html(self, path, title="hypviz"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html(title))
        return path

    def to_svg(self, path, view="poincare", state=None, **kw):
        """Publication vector export of one 2D view (also .pdf/.png by extension).
        Pass `state` (dict or path to a state JSON downloaded from the page) to
        reproduce an interactively arranged configuration."""
        from .export import save
        return save(self, path, view, state, **kw)
