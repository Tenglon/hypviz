"""Static, full-data analysis figures (matplotlib) — the paper 'analysis section'
staples. Computed on the FULL embedding, never the visualization sample, so the
numbers are exact even when the interactive scene shows a subsample."""
import numpy as np
from matplotlib.figure import Figure

from .colors import CAT
from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS


def _norms(coords, k, chart):
    x = as_numpy(coords)
    if chart != "lorentz":
        x = CHARTS[chart].to_lorentz(x, k)
    return L.dist(x, L.origin(x.shape[-1] - 1, k), k)


def _bare(ax):
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def norm_hist(coords, k=-1.0, chart="lorentz", by=None, bins=40, size=(5.2, 3.2)):
    """Distribution of hyperbolic norm ‖x‖ (distance from the origin), optionally
    split by a per-node group label."""
    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    n = _norms(coords, k, chart)
    if by is None:
        ax.hist(n, bins=bins, color="#3987e5")
    else:
        by = np.asarray(by)
        for i, g in enumerate(dict.fromkeys(by)):
            ax.hist(n[by == g], bins=bins, color=CAT[i % len(CAT)], alpha=0.6, label=str(g))
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("hyperbolic norm  d(o, x)")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def depth_norm(coords, depths, k=-1.0, chart="lorentz", size=(5.2, 3.2)):
    """Tree depth vs hyperbolic norm (boxplot per depth) — the visual test of the
    'depth ≈ radius' claim that motivates hyperbolic embeddings."""
    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    n, depths = _norms(coords, k, chart), np.asarray(depths)
    lo, hi = int(depths.min()), int(depths.max())
    ax.boxplot([n[depths == d] for d in range(lo, hi + 1)], positions=range(lo, hi + 1),
               widths=0.6, showfliers=False, medianprops={"color": "#e34948"})
    ax.set_xlabel("tree depth")
    ax.set_ylabel("hyperbolic norm  d(o, x)")
    fig.tight_layout()
    return fig


def _to_lorentz(coords, chart, k):
    x = as_numpy(coords)
    return x if chart == "lorentz" else CHARTS[chart].to_lorentz(x, k)


def distortion(coords, tree, k=-1.0, chart="lorentz", n_pairs=4000, size=(5.2, 3.6), seed=0):
    """How faithfully the embedding reproduces the hierarchy: a scatter of tree
    (graph) distance vs embedding hyperbolic distance over sampled node pairs, with
    the best-fit scale. Reports the average relative distortion around that fit."""
    x = _to_lorentz(coords, chart, k)
    depth, rng = tree.depth(), np.random.default_rng(seed)
    anc = [set(tree.ancestors(i)) for i in range(len(tree))]

    def graph_dist(u, v):
        a = v
        while a not in anc[u]:
            a = int(tree.parent[a])
        return depth[u] + depth[v] - 2 * depth[a]

    u = rng.integers(0, len(tree), n_pairs)
    v = rng.integers(0, len(tree), n_pairs)
    keep = u != v
    u, v = u[keep], v[keep]
    g = np.array([graph_dist(int(a), int(b)) for a, b in zip(u, v)], float)
    e = L.dist(x[u], x[v], k)
    scale = float((g @ e) / (g @ g))                       # best embed ≈ scale · graph
    rel = np.abs(e / np.maximum(scale * g, 1e-9) - 1)

    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    ax.scatter(g, e, s=5, alpha=0.25, color="#3987e5", edgecolors="none")
    gg = np.array([g.min(), g.max()])
    ax.plot(gg, scale * gg, color="#e34948", lw=1.4, label=f"fit  (avg distortion {rel.mean():.1%})")
    ax.set_xlabel("tree (graph) distance")
    ax.set_ylabel("embedding hyperbolic distance")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def delta_hyperbolicity(coords, k=-1.0, chart="lorentz", n=180, n_quads=40000, size=(5.2, 3.2), seed=0):
    """Gromov 4-point δ over sampled quadruples of the embedding, normalized by the
    diameter (δ_rel = 2δ/diam): 0 = perfectly tree-like/hyperbolic, ~1 = far from it.
    Histogram of δ_rel with the worst-case marked."""
    x = _to_lorentz(coords, chart, k)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(n, len(x)), replace=False)
    D = L.dist(x[idx][:, None], x[idx][None], k)           # (n, n) distance matrix
    diam = D.max()
    q = rng.integers(0, len(idx), (n_quads, 4))
    a, b, c, dd = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s1 = D[a, b] + D[c, dd]
    s2 = D[a, c] + D[b, dd]
    s3 = D[a, dd] + D[b, c]
    top = np.sort(np.stack([s1, s2, s3], 1), 1)[:, ::-1]   # descending
    delta = (top[:, 0] - top[:, 1]) / 2 / (diam + 1e-12) * 2  # δ_rel = 2δ/diam

    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    ax.hist(delta, bins=40, color="#1baf7a")
    ax.axvline(delta.max(), color="#e34948", lw=1.4, label=f"worst δ_rel = {delta.max():.2f}")
    ax.set_xlabel("relative 4-point δ  (0 = tree-like)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
