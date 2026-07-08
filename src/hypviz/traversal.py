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
    from . import reduce as _reduce
    q = as_numpy(query)
    q = q if chart == "lorentz" else CHARTS[chart].to_lorentz(q[None], k)[0]
    B = as_numpy(bank)
    B = B if chart == "lorentz" else CHARTS[chart].to_lorentz(B, k)
    labels, o = list(labels), L.origin(B.shape[-1] - 1, k)

    seq = []                                                   # bank indices retrieved along the walk
    for t in np.linspace(0, 1, steps):
        j = int(np.argmin(L.dist(L.geodesic(q, o, t, k)[None], B, k)))
        if not seq or seq[-1] != j:                           # dedup consecutive retrievals
            seq.append(j)

    # a real 2D embedding via radius-preserving PCA: radius = true distance-to-root
    # (the abstraction axis, exact), angle = PCA of the actual concept embeddings; the
    # radial SCALE is then normalized so the query fills the disk at any curvature.
    lo, _ = _reduce.radial_pca(B[seq], 2, k)
    pb = CHARTS["poincare"].from_lorentz(lo, k)
    pb = 0.92 / np.sqrt(-k) * pb / (np.linalg.norm(pb[0]) or 1.0)   # query → near the R-radius rim
    root = Point([0.0, 0.0], chart="poincare", label="root", draggable=False, color="#898781")
    cols = colors.by_scalar(np.linspace(1, 0, len(seq)))      # specific (boundary) → abstract (center)
    marks = [Point(pb[i].tolist(), chart="poincare", label=labels[j], draggable=False,
                   color="#2a78d6" if i == 0 else cols[i]) for i, j in enumerate(seq)]
    edges = [Geodesic(marks[i], marks[i + 1]) for i in range(len(marks) - 1)] + [Geodesic(marks[-1], root)]
    return Scene([*edges, root, *marks], views=("poincare", "lorentz"), curvature=k,
                 legend=[("point", "#2a78d6", "query — most specific (boundary)"),
                         ("point", "#898781", "root — most abstract (center)")],
                 hint=("A MERU-style root traversal: from the query along the geodesic to the root, the nearest "
                       "concept retrieved at each step grows more abstract. Placed by radius-preserving PCA — "
                       "radius is the true distance-to-root (specific → abstract), angle is the concepts' real "
                       "PCA direction (the radial scale is normalized to fill the disk). Drag the 3D view."))
