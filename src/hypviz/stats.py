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
