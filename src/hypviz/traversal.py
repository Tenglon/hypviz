"""Root traversal (MERU, Desai et al. 2023): walk the geodesic from a query embedding
toward the origin (the root of hyperbolic space) and retrieve the nearest concept from
a bank at each step — specific at the boundary, abstract at the center. The retrieval
runs in the full embedding space; the path is drawn as a radial spoke in the disk, at
each concept's true distance-to-root."""
import numpy as np

from . import colors
from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS
from .scene import Geodesic, Point, Scene


def traversal_scene(query, bank, labels, k=-1.0, chart="lorentz", steps=30):
    """Return a Scene of the query→root traversal. `query` is one embedding, `bank` an
    (N, d) array of concept embeddings with `labels`; both in `chart` coordinates."""
    q = as_numpy(query)
    q = q if chart == "lorentz" else CHARTS[chart].to_lorentz(q[None], k)[0]
    B = as_numpy(bank)
    B = B if chart == "lorentz" else CHARTS[chart].to_lorentz(B, k)
    labels, o = list(labels), L.origin(B.shape[-1] - 1, k)

    # walk query→root, retrieving the nearest concept per step; keep a step only when it is
    # meaningfully more abstract than the last (drops near-duplicate & noisy retrievals)
    rq = float(L.dist(q[None], o, k))                          # query's distance to root
    seq, kept_r = [], rq + 1.0
    for t in np.linspace(0, 1, steps):
        j = int(np.argmin(L.dist(L.geodesic(q, o, t, k)[None], B, k)))
        rj = float(L.dist(B[j][None], o, k))
        if not seq or (j != seq[-1] and kept_r - rj > 0.06 * rq):
            seq.append(j)
            kept_r = rj

    # a traversal is a 1-D radial walk (specific → abstract), so string the concepts along a
    # single geodesic spoke: distance from the centre is the true distance-to-root; the angle
    # is layout only. Radial scale normalized so the query sits at the rim, at any curvature.
    r = L.dist(B[seq], o, k)                                   # (M,) distance-to-root, decreasing
    pb = (0.92 / np.sqrt(-k) / r.max()) * np.outer(r, [np.cos(0.9), np.sin(0.9)])
    root = Point([0.0, 0.0], chart="poincare", label="root", draggable=False, color="#898781")
    cols = colors.by_scalar(np.linspace(1, 0, len(seq)))      # specific (boundary) → abstract (center)
    marks = [Point(pb[i].tolist(), chart="poincare", label=labels[j], draggable=False,
                   color="#2a78d6" if i == 0 else cols[i]) for i, j in enumerate(seq)]
    edges = [Geodesic(marks[i], marks[i + 1]) for i in range(len(marks) - 1)] + [Geodesic(marks[-1], root)]
    return Scene([*edges, root, *marks], views=("poincare", "lorentz"), curvature=k,
                 legend=[("point", "#2a78d6", "query — most specific (boundary)"),
                         ("point", "#898781", "root — most abstract (center)")],
                 hint=("A MERU-style root traversal: from the query at the boundary along the geodesic to the root "
                       "at the centre, the nearest concept retrieved at each step grows more abstract. A traversal "
                       "is a 1-D radial walk, so the concepts are strung along a single geodesic spoke — distance "
                       "from the centre is the true distance-to-root (specific → abstract); the angle is layout only. "
                       "Both charts show the same walk; drag the hyperboloid."))
