"""Root traversal (MERU, Desai et al. 2023): walk the geodesic from a query embedding
toward the origin (the root of hyperbolic space) and retrieve the nearest concept from
a bank at each step — specific at the boundary, abstract at the center. A traversal is a
1-D radial walk, so the retrieved concepts are strung along a single geodesic spoke at
their true distance-to-root. `traversal_gallery` packs several precomputed walks into one
page with a picker (for encoders like MERU that can't run in the browser)."""
import numpy as np

from . import colors
from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS
from .scene import Geodesic, Point, Scene

_LEGEND = [("point", "#2a78d6", "query — most specific (boundary)"),
           ("point", "#898781", "root — most abstract (center)")]
_HINT = ("A MERU-style root traversal: from the query at the boundary along the geodesic to the root at the "
         "centre, the nearest concept retrieved at each step grows more abstract. A traversal is a 1-D radial "
         "walk, so the concepts are strung along a single geodesic spoke — distance from the centre is the true "
         "distance-to-root (specific → abstract); the angle is layout only. Both charts show the same walk; drag "
         "the hyperboloid.")


def _to_lorentz(x, chart, k, single=False):
    x = as_numpy(x)
    if chart == "lorentz":
        return x
    return CHARTS[chart].to_lorentz(x[None], k)[0] if single else CHARTS[chart].to_lorentz(x, k)


def _objects(q, B, labels, o, k, steps):
    """The Point/Geodesic objects for one query→root walk (q, B in Lorentz coords)."""
    rq = float(L.dist(q[None], o, k))                          # query's distance to root
    seq, kept_r = [], rq + 1.0                                 # thin near-duplicate & noisy retrievals
    for t in np.linspace(0, 1, steps):
        j = int(np.argmin(L.dist(L.geodesic(q, o, t, k)[None], B, k)))
        rj = float(L.dist(B[j][None], o, k))
        if not seq or (j != seq[-1] and kept_r - rj > 0.06 * rq):
            seq.append(j)
            kept_r = rj
    r = L.dist(B[seq], o, k)                                   # distance-to-root, decreasing
    pb = (0.92 / np.sqrt(-k) / r.max()) * np.outer(r, [np.cos(0.9), np.sin(0.9)])
    root = Point([0.0, 0.0], chart="poincare", label="root", draggable=False, color="#898781")
    cols = colors.by_scalar(np.linspace(1, 0, len(seq)))      # specific (boundary) → abstract (center)
    marks = [Point(pb[i].tolist(), chart="poincare", label=labels[j], draggable=False,
                   color="#2a78d6" if i == 0 else cols[i]) for i, j in enumerate(seq)]
    edges = [Geodesic(marks[i], marks[i + 1]) for i in range(len(marks) - 1)] + [Geodesic(marks[-1], root)]
    return [*edges, root, *marks]


def traversal_scene(query, bank, labels, k=-1.0, chart="lorentz", steps=30):
    """Return a Scene of the query→root traversal. `query` is one embedding, `bank` an
    (N, d) array of concept embeddings with `labels`; both in `chart` coordinates."""
    B = _to_lorentz(bank, chart, k)
    o = L.origin(B.shape[-1] - 1, k)
    objs = _objects(_to_lorentz(query, chart, k, single=True), B, list(labels), o, k, steps)
    return Scene(objs, views=("poincare", "lorentz"), curvature=k, legend=_LEGEND, hint=_HINT)


def traversal_gallery(items, bank, labels, k=-1.0, chart="lorentz", steps=30):
    """One page with a picker over several precomputed traversals. `items` is a list of
    (display_label, query_embedding); each query is walked to the root independently."""
    B = _to_lorentz(bank, chart, k)
    o, labels = L.origin(B.shape[-1] - 1, k), list(labels)
    variants = [(name, _objects(_to_lorentz(q, chart, k, single=True), B, labels, o, k, steps))
                for name, q in items]
    return Scene(variants[0][1], views=("poincare", "lorentz"), curvature=k,
                 legend=_LEGEND, hint=_HINT, variants=variants)
