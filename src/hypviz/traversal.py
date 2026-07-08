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

    seq = []                                                   # bank indices retrieved along the walk
    for t in np.linspace(0, 1, steps):
        j = int(np.argmin(L.dist(L.geodesic(q, o, t, k)[None], B, k)))
        if not seq or seq[-1] != j:                           # dedup consecutive retrievals
            seq.append(j)

    # a faithful 2D embedding centred at the root: keep each concept's exact distance-to-root
    # (the abstraction axis) and get the angle from an UNCENTERED projection of its tangent
    # direction — a traversal runs nearly along one radial ray, so near-collinear concepts stay
    # near one spoke (with their real angular deviations) instead of fanning across the disk.
    v = L.logmap(o, B[seq], k)
    r = np.sqrt(np.maximum(L.mdot(v, v, True), 0.0))               # (M,1) distance to root
    basis = L.tangent_basis(o, k)
    u = v @ basis.T - 2 * np.outer(v[:, 0], basis[:, 0])          # (M,n) spatial tangent coords, |u|=r
    ax = np.linalg.svd(u, full_matrices=False)[2][:2]            # top-2 axes ≈ the shared ray first
    ang = u @ ax.T
    ang = ang / np.maximum(np.linalg.norm(ang, axis=1, keepdims=True), 1e-12)
    pb = 0.92 / np.sqrt(-k) * (r / r.max()) * ang                 # fill the R-radius disk; query → rim
    root = Point([0.0, 0.0], chart="poincare", label="root", draggable=False, color="#898781")
    cols = colors.by_scalar(np.linspace(1, 0, len(seq)))      # specific (boundary) → abstract (center)
    marks = [Point(pb[i].tolist(), chart="poincare", label=labels[j], draggable=False,
                   color="#2a78d6" if i == 0 else cols[i]) for i, j in enumerate(seq)]
    edges = [Geodesic(marks[i], marks[i + 1]) for i in range(len(marks) - 1)] + [Geodesic(marks[-1], root)]
    return Scene([*edges, root, *marks], views=("poincare", "lorentz"), curvature=k,
                 legend=[("point", "#2a78d6", "query — most specific (boundary)"),
                         ("point", "#898781", "root — most abstract (center)")],
                 hint=("A MERU-style root traversal: from the query along the geodesic to the root, the nearest "
                       "concept retrieved at each step grows more abstract. Radius is the true distance-to-root "
                       "(specific → abstract); the angle is a faithful projection of each concept's direction — a "
                       "traversal is nearly radial, so the concepts stay near one spoke, showing their real angular "
                       "deviations (the radial scale is normalized to fill the disk). Drag the 3D view."))
