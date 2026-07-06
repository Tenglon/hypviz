"""Minimal rooted-tree helper shared by embedding, sampling and synthesis.

Nodes are ints 0..N-1; node 0 is the root by convention. `parent[i]` is the
parent index (-1 for the root). Everything else (children, depth, BFS order) is
derived on demand — the tree is small metadata, not the embedding.
"""
import numpy as np


class Tree:
    @classmethod
    def from_edges(cls, edges, n=None, labels=None):
        """Build from (parent, child) edges; node count inferred if not given."""
        edges = [(int(u), int(v)) for u, v in edges]
        n = n or (max(max(e) for e in edges) + 1)
        parent = np.full(n, -1)
        for u, v in edges:
            parent[v] = u
        return cls(parent, labels)

    def __init__(self, parent, labels=None):
        self.parent = np.asarray(parent, int)
        self.labels = labels
        self.children = [[] for _ in self.parent]
        for i, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(i)

    def __len__(self):
        return len(self.parent)

    @property
    def root(self):
        return int(np.where(self.parent < 0)[0][0])

    def depth(self):
        d = np.zeros(len(self), int)
        for i in self.bfs()[1:]:
            d[i] = d[self.parent[i]] + 1
        return d

    def bfs(self):
        order, frontier = [], [self.root]
        while frontier:
            order += frontier
            frontier = [c for i in frontier for c in self.children[i]]
        return order

    def ancestors(self, i):
        """i, its parent, ... up to the root."""
        chain = [i]
        while self.parent[chain[-1]] >= 0:
            chain.append(int(self.parent[chain[-1]]))
        return chain

    def edges(self):
        return [(int(self.parent[i]), i) for i in range(len(self)) if self.parent[i] >= 0]
