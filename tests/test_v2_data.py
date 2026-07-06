import numpy as np

from hypviz import synth
from hypviz.hierarchy import Hierarchy
from hypviz.kernel import lorentz as L
from hypviz.sample import sample

K = -1.0


# ---- sampler invariants -----------------------------------------------------

def test_sample_respects_budget_and_keeps_root():
    t = synth.taxonomy(5000, seed=1)
    s = sample(t, budget=800, seed=1)
    assert len(s.kept) <= 800
    assert t.root in set(s.kept)


def test_sample_is_ancestor_closed():
    t = synth.taxonomy(5000, seed=2)
    kept = set(sample(t, budget=1000, seed=2).kept)
    for i in kept:
        if t.parent[i] >= 0:
            assert int(t.parent[i]) in kept          # every kept node's parent is kept


def test_sample_accounts_for_every_pruned_leaf():
    t = synth.taxonomy(5000, seed=3)
    s = sample(t, budget=600, seed=3)
    total_leaves = sum(not t.children[i] for i in range(len(t)))
    kept_leaves = sum(not t.children[i] for i in s.kept)
    assert s.pruned_leaves[t.root] == total_leaves - kept_leaves   # nothing silently lost


def test_sample_is_deterministic():
    t = synth.taxonomy(4000, seed=4)
    assert np.array_equal(sample(t, budget=500, seed=7).kept, sample(t, budget=500, seed=7).kept)


def test_sample_respects_max_depth():
    t = synth.taxonomy(5000, seed=5)
    s = sample(t, budget=9999, max_depth=3, seed=5)
    d = t.depth()
    assert all(d[i] <= 3 for i in s.kept)


def test_budget_over_tree_keeps_everything():
    t = synth.taxonomy(400, seed=6)
    assert len(sample(t, budget=10_000, seed=6).kept) == len(t)


# ---- Hierarchy sample + reduce ----------------------------------------------

def test_hierarchy_sample_then_reduce_pipeline():
    t = synth.taxonomy(6000, seed=8)
    coords = synth.diffuse(t, dim=128, k=K, seed=8)
    labels = t.labels
    h = Hierarchy(coords, t, labels)
    s = h.sample(1500, seed=8)
    assert len(s) <= 1500 and s.rate == len(s) / len(t)
    assert s.labels is not None and len(s.labels) == len(s)
    # sampled tree is valid: exactly one root, parents point to earlier kept nodes
    assert (s.tree.parent < 0).sum() == 1
    lo = s.reduce(2, "radial")
    assert lo.dim == 2 and np.allclose(L.mdot(lo.coords, lo.coords), -1, atol=1e-7)
    # radial reduction keeps depth↔radius on the sampled cloud
    assert np.corrcoef(lo.depth(), lo.norm())[0, 1] > 0.95


def test_hierarchy_pruned_counts_survive_sampling():
    t = synth.taxonomy(6000, seed=9)
    h = Hierarchy(synth.diffuse(t, dim=32, k=K, seed=9), t)
    s = h.sample(700, seed=9)
    assert s.pruned_leaves.shape == (len(s),)
    assert s.pruned_leaves.sum() > 0                 # something was pruned and recorded
