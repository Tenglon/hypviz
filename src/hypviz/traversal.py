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


def traversal_scene(query, bank, labels, k=-1.0, chart="lorentz", steps=24, angle=0.7):
    """Return a Scene of the query→root traversal. `query` is one embedding, `bank` an
    (N, d) array of concept embeddings with `labels`; both in `chart` coordinates."""
    q = as_numpy(query)
    q = q if chart == "lorentz" else CHARTS[chart].to_lorentz(q[None], k)[0]
    B = as_numpy(bank)
    B = B if chart == "lorentz" else CHARTS[chart].to_lorentz(B, k)
    labels, R, o = list(labels), 1 / np.sqrt(-k), L.origin(B.shape[-1] - 1, k)

    seq = []                                                   # (radius, concept) along the walk
    for t in np.linspace(0, 1, steps):
        wp = L.geodesic(q, o, t, k)
        name = labels[int(np.argmin(L.dist(wp[None], B, k)))]
        r = float(L.dist(wp, o, k))
        if not seq or seq[-1][1] != name:                     # dedup consecutive retrievals
            seq.append((r, name))

    # schematic radial spoke: place each concept at disk radius ∝ its distance-to-root,
    # normalized so the query sits near the boundary (robust to the model's curvature scale)
    ca, sa = np.cos(angle), np.sin(angle)
    r_max = max((r for r, _ in seq), default=1.0) or 1.0
    disk = lambda r: [0.92 * R * (r / r_max) * ca, 0.92 * R * (r / r_max) * sa]
    root = Point([0.0, 0.0], chart="poincare", label="root", draggable=False, color="#898781")
    cols = colors.by_scalar(np.linspace(1, 0, len(seq)))      # boundary (specific) → center (abstract)
    marks = [Point(disk(r), chart="poincare", label=name, draggable=False,
                   color="#2a78d6" if i == 0 else cols[i]) for i, (r, name) in enumerate(seq)]
    objs = [Geodesic(marks[0], root), root, *marks]
    return Scene(objs, views=("poincare", "lorentz"), curvature=k,
                 legend=[("point", "#2a78d6", "query — most specific (boundary)"),
                         ("point", "#898781", "root — most abstract (center)")],
                 hint=("A MERU-style root traversal: from the query at the boundary along the geodesic to the root "
                       "at the center, the nearest concept retrieved at each step grows more abstract. Retrieval is "
                       "in the full embedding; the spoke places each concept at its true distance-to-root. Drag the "
                       "hyperboloid view, scroll to zoom."))
