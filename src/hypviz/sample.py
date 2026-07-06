"""Hierarchy-aware down-sampling: keep ≤ budget nodes while preserving tree
shape. Budget flows top-down, split among children in proportion to subtree
size; a node that cannot afford all its children keeps a size-weighted sample of
them. Ancestor closure is automatic (we only ever descend into kept nodes). No
silent caps: every kept node exposes how many of its descendant leaves are not
shown, for the viewer to disclose.
"""
import numpy as np

from .tree import Tree


def _subtree_sizes(tree):
    size = np.ones(len(tree), int)
    for u in reversed(tree.bfs()):
        if tree.parent[u] >= 0:
            size[tree.parent[u]] += size[u]
    return size


def _proportional(total, weights):
    """Split `total` into len(weights) integers ≥1, ∝ weights (largest remainder)."""
    w = np.asarray(weights, float)
    exact = 1 + (total - len(w)) * w / w.sum()
    base = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - base))[: total - base.sum()]:
        base[i] += 1
    return base


class Sampling:
    """Result of `sample`: the kept node indices (a subset of the original tree)
    and, per kept node, the number of descendant leaves not shown."""

    def __init__(self, tree, kept, pruned_leaves):
        self.tree, self.kept = tree, np.asarray(kept)
        self.pruned_leaves = pruned_leaves          # full-length array, valid on kept nodes
        self.rate = len(self.kept) / len(tree)


def sample(tree, budget=10_000, max_depth=None, root=None, seed=0):
    if not isinstance(tree, Tree):
        tree = Tree(tree)
    rng = np.random.default_rng(seed)
    root = tree.root if root is None else int(root)
    size, depth = _subtree_sizes(tree), tree.depth()
    kept = []

    def alloc(u, b):                                 # keep ≤ b nodes from u's subtree
        kept.append(u)
        kids = tree.children[u]
        if not kids or b <= 1 or (max_depth is not None and depth[u] >= max_depth):
            return
        rem = b - 1
        if rem >= len(kids):                         # afford all children, split ∝ subtree size
            for c, bc in zip(kids, _proportional(rem, [size[c] for c in kids])):
                alloc(c, int(bc))
        else:                                        # afford only `rem`: size-weighted sample
            w = np.array([size[c] for c in kids], float)
            for c in rng.choice(kids, rem, replace=False, p=w / w.sum()):
                kept.append(int(c))

    alloc(root, budget)

    # pruned-leaf accounting: original leaves under u that are not in the kept set
    kept_set = set(kept)
    is_leaf = np.array([not tree.children[i] for i in range(len(tree))])
    n_leaves = np.where(is_leaf, 1, 0)
    kept_leaves = np.where(is_leaf & np.isin(np.arange(len(tree)), list(kept_set)), 1, 0)
    for u in reversed(tree.bfs()):
        if tree.parent[u] >= 0:
            n_leaves[tree.parent[u]] += n_leaves[u]
            kept_leaves[tree.parent[u]] += kept_leaves[u]
    return Sampling(tree, sorted(kept_set), n_leaves - kept_leaves)
