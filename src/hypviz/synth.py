"""Synthetic hierarchical data shaped like a biological taxonomy (à la
TreeOfLife / BioCLIP): a fixed number of Linnaean ranks, ragged (clades may
terminate early), heavy-tailed branching (few huge genera, many small ones).
Paired with a matching H^dim embedding grown by geodesic diffusion, so the
'depth ≈ radius' relationship is a known ground truth for testing reduction.
"""
import numpy as np

from .kernel import lorentz as L
from .tree import Tree


def taxonomy(n_nodes=450_000, ranks=7, branch_mean=1.4, branch_sigma=1.0, stop_prob=0.05, seed=0):
    """Grow a ragged, heavy-tailed rooted tree; returns a Tree with rank labels.

    Children-per-node is round(lognormal) (heavy-tailed); a non-leaf below the
    last rank becomes a leaf early with probability `stop_prob` (raggedness).
    """
    rng = np.random.default_rng(seed)
    parent, rank, frontier = [-1], [0], [0]
    while frontier and len(parent) < n_nodes:
        nxt = []
        for u in frontier:
            if rank[u] >= ranks - 1 or rng.random() < stop_prob:
                continue
            n_kids = 1 + int(rng.lognormal(branch_mean, branch_sigma))
            for _ in range(min(n_kids, n_nodes - len(parent))):
                parent.append(u)
                rank.append(rank[u] + 1)
                nxt.append(len(parent) - 1)
        frontier = nxt
    return Tree(parent, labels=np.array(rank))


def diffuse(tree, dim=128, k=-1.0, step=0.6, wobble=0.55, seed=0):
    """Grow an H^dim embedding along the tree: each child continues its parent's
    outward heading plus angular `wobble`, so lineages stay ballistic and ‖x‖
    tracks depth (the 'deeper = nearer the boundary' ground truth). Root children
    fan out from the origin."""
    rng = np.random.default_rng(seed)
    unit = lambda u: u / np.sqrt(L.mdot(u, u))
    coords = np.zeros((len(tree), dim + 1))
    coords[tree.root] = L.origin(dim, k)
    fwd = {}  # unit tangent each node arrived along, i.e. its outward heading
    for i in tree.bfs()[1:]:
        p = int(tree.parent[i])
        noise = unit(L.to_tangent(coords[p], np.concatenate([[0.0], rng.standard_normal(dim)]), k))
        u = step * unit(noise if p not in fwd else fwd[p] + wobble * noise)
        coords[i] = L.expmap(coords[p], u, k)
        fwd[i] = L.ptransp(coords[p], coords[i], unit(u), k)
    return coords
