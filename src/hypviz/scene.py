"""Scene authoring: declarative primitives compiled to scene JSON, rendered by
the bundled TS runtime as a self-contained interactive HTML page.

Primitives reference each other directly (Geodesic(a, b)); the dependency
graph the runtime needs is just those id references.
"""
import json
from itertools import count
from pathlib import Path

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
    """A translucent patch of the tangent plane at a point (3D view only)."""

    def __init__(self, at):
        super().__init__()
        self.at = at

    def to_json(self, k):
        return {"id": self.id, "type": "tangent_plane", "at": self.at.id}


_DEFAULT_HINT = ("Drag the highlighted points — in either view; the 3D view orbits and zooms with the mouse. "
                 "The gray ring marks the edge of the 3D window: the disk beyond it (faint grid) continues to "
                 "infinity, which no isometric embedding can show — dragging stops at the ring in both views.")


class Scene:
    def __init__(self, objects, views=("poincare", "lorentz"), curvature=-1.0, curvature_slider=False, hint=None):
        self.objects, self.views = list(objects), list(views)
        self.curvature, self.curvature_slider = curvature, curvature_slider
        self.hint = hint or _DEFAULT_HINT

    def to_json(self):
        return {"views": [{"chart": v} for v in self.views],
                "objects": [o.to_json(self.curvature) for o in self.objects],
                "curvature": self.curvature, "curvatureSlider": self.curvature_slider}

    def to_html(self, path, title="hypviz"):
        html = ((_STATIC / "template.html").read_text()
                .replace("/*TITLE*/", title)
                .replace("/*HINT*/", self.hint)
                .replace("/*RUNTIME*/", (_STATIC / "hypviz-runtime.js").read_text())
                .replace("/*SCENE*/", json.dumps(self.to_json())))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return path

    def to_svg(self, path, view="poincare", state=None, **kw):
        """Publication vector export of one 2D view (also .pdf/.png by extension).
        Pass `state` (dict or path to a state JSON downloaded from the page) to
        reproduce an interactively arranged configuration."""
        from .export import save
        return save(self, path, view, state, **kw)
