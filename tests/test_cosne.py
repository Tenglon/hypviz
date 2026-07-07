import numpy as np
import pytest

from hypviz import reduce, synth
from hypviz.kernel import lorentz as L

pytest.importorskip("torch")   # CO-SNE is an optional (torch) feature
K = -1.0


def _knn_overlap(a, b, kk=10):
    da, db = L.dist(a[:, None], a[None], K), L.dist(b[:, None], b[None], K)
    np.fill_diagonal(da, np.inf)
    np.fill_diagonal(db, np.inf)
    na, nb = np.argsort(da, 1)[:, :kk], np.argsort(db, 1)[:, :kk]
    return np.mean([len(set(na[i]) & set(nb[i])) / kk for i in range(len(a))])


def test_cosne_preserves_neighborhoods_better_than_tangent():
    c = synth.diffuse(synth.taxonomy(1200, seed=0), dim=48, k=K, seed=0)
    sub = c[np.random.default_rng(0).choice(len(c), 160, replace=False)]
    lo = reduce.co_sne(sub, dim=2, k=K, iters=500, seed=0)[0]
    assert np.allclose(L.mdot(lo, lo), -1, atol=1e-6) and np.isfinite(lo).all()
    # the whole point of CO-SNE: local neighborhoods, clearly better than a linear map
    assert _knn_overlap(sub, lo) > _knn_overlap(sub, reduce.tangent_pca(sub, 2, K)[0]) + 0.15
