"""`Hierarchy` — the structured input of the Embedding Atlas: node coordinates +
a rooted tree + labels, with the consistency between them (sampling keeps points
and edges aligned; reduction preserves the root-radius) internalized so callers
cannot desynchronize them. `.sample()` and `.reduce()` return new Hierarchies,
so the atlas pipeline reads `hier.sample(10_000).reduce(2)`.
"""
import numpy as np

from . import reduce as _reduce
from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS
from .sample import sample as _sample
from .tree import Tree


class Hierarchy:
    def __init__(self, coords, tree, labels=None, chart="lorentz", k=-1.0,
                 pruned_leaves=None, rate=1.0):
        coords = as_numpy(coords)
        self.coords = coords if chart == "lorentz" else CHARTS[chart].to_lorentz(coords, k)
        self.tree = tree if isinstance(tree, Tree) else Tree(tree)
        self.labels = None if labels is None else np.asarray(labels)
        self.k = k
        self.pruned_leaves = np.zeros(len(self.tree), int) if pruned_leaves is None else np.asarray(pruned_leaves)
        self.rate = rate

    def __len__(self):
        return len(self.tree)

    @property
    def dim(self):
        return self.coords.shape[-1] - 1

    def depth(self):
        return self.tree.depth()

    def norm(self):
        """Hyperbolic distance of each node from the root (≈ tree depth)."""
        return L.dist(self.coords, self.coords[self.tree.root], self.k)

    def sample(self, budget=10_000, **kw):
        """Down-sample to ≤ budget nodes (hierarchy-aware); returns a new Hierarchy
        on the kept subset, carrying per-node pruned-leaf counts and the rate."""
        s = _sample(self.tree, budget, **kw)
        idx = s.kept
        remap = np.full(len(self.tree), -1)
        remap[idx] = np.arange(len(idx))
        parent = [-1 if self.tree.parent[i] < 0 else int(remap[self.tree.parent[i]]) for i in idx]
        return Hierarchy(self.coords[idx], Tree(parent),
                         None if self.labels is None else self.labels[idx],
                         k=self.k, pruned_leaves=s.pruned_leaves[idx], rate=s.rate)

    def reduce(self, dim=2, method="radial"):
        """Project to `dim` dimensions. 'radial' (default) preserves each node's
        distance to the root — so depth↔radius survives; 'tangent' is plain PCA."""
        if self.dim <= dim:
            lo = self.coords
        elif method == "radial":
            lo, _ = _reduce.radial_pca(self.coords, dim, self.k, center=self.coords[self.tree.root])
        else:
            lo, _ = _reduce.tangent_pca(self.coords, dim, self.k)
        return Hierarchy(lo, self.tree, self.labels, k=self.k,
                         pruned_leaves=self.pruned_leaves, rate=self.rate)
