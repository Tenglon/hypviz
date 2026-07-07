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
                 pruned_leaves=None, rate=1.0, groups=None):
        coords = as_numpy(coords)
        self.coords = coords if chart == "lorentz" else CHARTS[chart].to_lorentz(coords, k)
        self.tree = tree if isinstance(tree, Tree) else Tree(tree)
        self.labels = None if labels is None else np.asarray(labels)
        self.groups = None if groups is None else np.asarray(groups)
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
                         k=self.k, pruned_leaves=s.pruned_leaves[idx], rate=s.rate,
                         groups=None if self.groups is None else self.groups[idx])

    def reduce(self, dim=2, method="radial"):
        """Project to `dim` dimensions.
        'radial'  — radius = distance-to-root (exact), angle from embedding PCA:
                    faithful to the embedding's angular structure (can blob for
                    cone-like embeddings).
        'tree'    — radius = distance-to-root (exact), angle from the tree layout:
                    legible spread (clades fan across the disk); the angle is a
                    layout, not embedding-derived.
        'horo'    — horospherical / Busemann reduction (HoroPCA, Chami et al. 2021):
                    preserves hyperbolic structure better than tangent PCA (k=-1).
        'cosne'   — CO-SNE hyperbolic t-SNE (Guo et al. 2022): best local-neighborhood
                    preservation, distorts global distances; O(N²), torch (k=-1).
        'tangent' — plain tangent-space PCA (no privileged radius)."""
        if self.dim <= dim and method != "tree":
            lo = self.coords
        elif method == "tree" and dim == 2:
            lo = self._tree_layout()
        elif method == "tree":                              # tree layout is 2D-only
            lo, _ = _reduce.radial_pca(self.coords, dim, self.k, center=self.coords[self.tree.root])
        elif method == "radial":
            lo, _ = _reduce.radial_pca(self.coords, dim, self.k, center=self.coords[self.tree.root])
        elif method == "horo":
            lo, _ = _reduce.horo_pca(self.coords, dim, self.k)
        elif method == "cosne":
            lo, _ = _reduce.co_sne(self.coords, dim, self.k)
        else:
            lo, _ = _reduce.tangent_pca(self.coords, dim, self.k)
        return Hierarchy(lo, self.tree, self.labels, k=self.k,
                         pruned_leaves=self.pruned_leaves, rate=self.rate, groups=self.groups)

    def _tree_layout(self):
        """H² layout: real hyperbolic radius from the embedding, angular sector per
        node from the tree (split ∝ subtree size) — a Sarkar-style spread."""
        from .sample import _subtree_sizes
        r = self.norm()
        size = _subtree_sizes(self.tree)
        angle = np.zeros(len(self))
        sector = {self.tree.root: (0.0, 2 * np.pi)}
        for u in self.tree.bfs():
            a0, a1 = sector[u]
            angle[u] = (a0 + a1) / 2
            kids = self.tree.children[u]
            total, acc = sum(size[c] for c in kids), a0
            for c in kids:
                w = (a1 - a0) * size[c] / total
                sector[c] = (acc, acc + w)
                acc += w
        u2 = r[:, None] * np.stack([np.cos(angle), np.sin(angle)], -1)
        return L.expmap(L.origin(2, self.k), np.concatenate([np.zeros((len(self), 1)), u2], -1), self.k)
