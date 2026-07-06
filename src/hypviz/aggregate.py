"""Aggregating hyperbolic points: the Fréchet mean (hyperbolic centroid), the
right way to average points on the manifold (the Euclidean mean leaves it)."""
import numpy as np

from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS


def centroid(coords, chart="poincare", weights=None, k=-1.0):
    """Hyperbolic centroid (weighted Fréchet mean) of the points in `coords`, given
    in `chart` coordinates ('poincare' | 'klein' | 'halfplane' | 'lorentz'); returns
    the centroid in the SAME chart. Accepts numpy or torch input, any dimension."""
    x = as_numpy(coords)
    if chart != "lorentz":
        x = CHARTS[chart].to_lorentz(x, k)
    m = L.frechet_mean(x, k, weights=weights)
    return m if chart == "lorentz" else CHARTS[chart].from_lorentz(m, k)


def centroids(coords, groups, chart="poincare", k=-1.0):
    """Per-group hyperbolic centroids. `groups` is a length-N array of labels;
    returns (labels, centroids) where centroids[i] is the centroid of group
    labels[i], in `chart` coordinates."""
    x, g = as_numpy(coords), np.asarray(groups)
    labels = list(dict.fromkeys(g.tolist()))
    return labels, np.stack([centroid(x[g == lab], chart, k=k) for lab in labels])
