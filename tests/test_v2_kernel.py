import numpy as np
import pytest

from hypviz import embed, reduce, synth
from hypviz.kernel import lorentz as L, mobius as M
from hypviz.tree import Tree

K = -1.0


# ---- Fréchet mean & tangent basis -------------------------------------------

def test_frechet_mean_of_identical_points():
    x = L.from_spatial(np.array([0.3, -0.4, 0.2]), K)
    m = L.frechet_mean(np.tile(x, (5, 1)), K)
    assert np.allclose(m, x, atol=1e-6)


def test_frechet_mean_of_pair_is_midpoint():
    x, y = (L.from_spatial(p, K) for p in (np.array([0.5, 0.0]), np.array([-0.3, 0.4])))
    m = L.frechet_mean(np.stack([x, y]), K)
    assert L.dist(m, L.geodesic(x, y, 0.5, K), K) < 1e-6


def test_tangent_basis_is_minkowski_orthonormal_and_in_tangent():
    m = L.from_spatial(np.array([0.4, -0.2, 0.5, 0.1]), K)
    b = L.tangent_basis(m, K)
    assert b.shape == (4, 5)                                     # dim of T_m = n
    gram = b @ b.T - 2 * np.outer(b[:, 0], b[:, 0])
    assert np.allclose(gram, np.eye(4), atol=1e-9)
    assert np.allclose(np.sum(m * b, -1) - 2 * m[0] * b[:, 0], 0, atol=1e-9)


# ---- tangent PCA ------------------------------------------------------------

def test_tangent_pca_recovers_a_planted_2d_subspace():
    rng = np.random.default_rng(1)
    p2 = L.from_spatial(rng.uniform(-1, 1, (40, 2)), K)          # genuine H^2 points
    hi = np.concatenate([p2, np.zeros((40, 3))], -1)            # embed into H^4 (totally geodesic)
    pts, info = reduce.tangent_pca(hi, dim=2, k=K)
    assert info["explained_variance_ratio"] > 0.999
    # pairwise distances preserved by the projection
    d_hi = L.dist(hi[:, None], hi[None], K)
    d_lo = L.dist(pts[:, None], pts[None], K)
    assert np.allclose(d_hi, d_lo, atol=1e-5)


def test_tangent_pca_output_on_manifold():
    pts, _ = reduce.tangent_pca(synth.diffuse(synth.taxonomy(300, seed=2), dim=16, seed=2), dim=2)
    assert np.allclose(L.mdot(pts, pts), -1, atol=1e-7)


# ---- Sarkar embedding -------------------------------------------------------

def test_sarkar_stays_in_disk_and_matches_tau_on_a_path():
    tau = 1.3
    pos = embed.sarkar(Tree([-1, 0, 1, 2, 3]), tau=tau)          # a path 0-1-2-3-4
    assert np.all(np.linalg.norm(pos, axis=1) < 1)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        assert np.isclose(M.dist(pos[a], pos[b], K), tau, atol=1e-9)


def test_sarkar_low_distortion_on_a_path():
    tau = 2.0
    pos = embed.sarkar(Tree([-1, 0, 1, 2]), tau=tau)             # graph dist root->leaf = 3
    assert np.isclose(M.dist(pos[0], pos[3], K), 3 * tau, atol=1e-6)  # geodesic, no distortion


# ---- synthetic taxonomy -----------------------------------------------------

def test_taxonomy_is_ragged_and_heavy_tailed():
    t = synth.taxonomy(3000, ranks=7, seed=3)
    depth = t.depth()
    leaves = [i for i in range(len(t)) if not t.children[i]]
    assert len({depth[i] for i in leaves}) > 1                   # ragged: leaves at many depths
    n_children = np.array([len(c) for c in t.children if c])
    assert n_children.max() > 3 * n_children.mean()             # heavy-tailed branching
    assert len(t) <= 3000 and t.parent[t.root] < 0


def test_diffuse_makes_depth_track_radius():
    t = synth.taxonomy(2000, seed=4)
    coords = synth.diffuse(t, dim=64, k=K, seed=4)
    assert np.allclose(L.mdot(coords, coords), -1, atol=1e-6)
    norm = L.dist(coords, L.origin(64, K), K)
    depth = t.depth()
    r = np.corrcoef(depth, norm)[0, 1]
    assert r > 0.9                                              # the ground-truth relationship
