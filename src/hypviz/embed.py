"""Sarkar's construction: a deterministic, distortion-controlled embedding of a
tree into the Poincaré disk H² (Sarkar 2011). Each node's children are placed on
a circle in the frame that sends the node to the origin, equally spaced in the
angles not occupied by the parent, at hyperbolic distance `tau` (larger tau ->
lower distortion). Returns 2D Poincaré coordinates (k = -1)."""
import numpy as np

from .kernel import mobius as M
from .tree import Tree


def sarkar(tree, tau=1.0):
    """Embed a Tree into the Poincaré disk; returns an (N, 2) array of coords."""
    if not isinstance(tree, Tree):
        tree = Tree(tree)
    rho = np.tanh(tau / 2)                       # Euclidean radius for hyperbolic distance tau
    pos = np.zeros((len(tree), 2))
    for u in tree.bfs():
        kids, p = tree.children[u], tree.parent[u]
        if not kids:
            continue
        # angle toward the parent, in the frame that moves u to the origin
        theta0 = 0.0 if p < 0 else np.arctan2(*reversed(M.add(-pos[u], pos[p])))
        step = 2 * np.pi / (len(kids) + (p >= 0))
        for i, c in enumerate(kids, start=1 if p >= 0 else 0):
            a = theta0 + i * step
            pos[c] = M.add(pos[u], rho * np.array([np.cos(a), np.sin(a)]))
    return pos
