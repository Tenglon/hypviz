import numpy as np

import hypviz
from hypviz.kernel import lorentz as L
from hypviz.kernel.charts import Poincare


def test_centroid_returns_same_chart_and_is_the_minimizer():
    pts = np.array([[0.5, 0.1], [0.3, 0.4], [-0.4, 0.3], [-0.2, -0.5], [0.1, -0.3]])
    m = hypviz.centroid(pts, chart="poincare")
    assert m.shape == (2,)                                    # same chart (Poincaré 2D)
    # it minimizes the sum of squared geodesic distances (the Fréchet property)
    xs = Poincare.to_lorentz(pts, -1.0)
    ml = Poincare.to_lorentz(m, -1.0)
    cost = lambda z: float((L.dist(z, xs, -1.0) ** 2).sum())
    rng = np.random.default_rng(0)
    assert all(cost(L.expmap(ml, L.to_tangent(ml, np.concatenate([[0.0], rng.normal(0, 0.1, 2)]), -1.0), -1.0))
               >= cost(ml) - 1e-9 for _ in range(300))


def test_centroid_of_identical_points_and_weighting():
    p = np.array([0.3, -0.2])
    assert np.allclose(hypviz.centroid(np.tile(p, (6, 1)), chart="poincare"), p, atol=1e-6)
    pts = np.array([[0.6, 0.0], [-0.6, 0.0]])
    pulled = hypviz.centroid(pts, chart="poincare", weights=[9, 1])
    assert pulled[0] > 0.2                                    # weight pulls the mean toward point 0


def test_centroids_by_group():
    pts = np.array([[0.5, 0.1], [0.45, 0.15], [-0.4, -0.3], [-0.45, -0.25]])
    labels, cs = hypviz.centroids(pts, ["a", "a", "b", "b"], chart="poincare")
    assert labels == ["a", "b"] and cs.shape == (2, 2)
    assert cs[0][0] > 0 and cs[1][0] < 0                     # group means sit in their clusters
