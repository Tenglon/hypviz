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


class Scene:
    def __init__(self, objects, views=("poincare", "lorentz"), curvature=-1.0, curvature_slider=False):
        self.objects, self.views = list(objects), list(views)
        self.curvature, self.curvature_slider = curvature, curvature_slider

    def to_json(self):
        return {"views": [{"chart": v} for v in self.views],
                "objects": [o.to_json(self.curvature) for o in self.objects],
                "curvature": self.curvature, "curvatureSlider": self.curvature_slider}

    def to_html(self, path, title="hypviz"):
        html = ((_STATIC / "template.html").read_text()
                .replace("/*TITLE*/", title)
                .replace("/*RUNTIME*/", (_STATIC / "hypviz-runtime.js").read_text())
                .replace("/*SCENE*/", json.dumps(self.to_json())))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return path
