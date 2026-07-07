"""The Embedding Atlas facade: an n-dimensional hyperbolic embedding plus its
hierarchy → a self-contained interactive Scene in one call. Wraps Hierarchy
(sample → reduce), color encoding, and honest disclosure of BOTH the sampling
(pruned-leaf counts) and the projection (which reduction, and its trade-off)."""
import numpy as np

from . import colors
from .aggregate import centroids as _centroids
from .hierarchy import Hierarchy
from .scene import Cloud, Point, Scene
from .tree import Tree

_ENCODE = {
    "depth": (lambda h: h.depth(), "depth (light = shallow)"),
    "norm": (lambda h: h.norm(), "distance from root"),
    "label": (None, "label"),
}


def _resolve_groups(tree, labels, show_centroids):
    """A per-node group array from show_centroids: an explicit array, 'depth', or
    'clade' (the depth-1 ancestor's label)."""
    if not isinstance(show_centroids, str):
        return np.asarray(show_centroids)
    if show_centroids == "depth":
        return tree.depth().astype(str)
    depth = tree.depth()                                  # 'clade' = top-level (depth-1) ancestor
    def top(i):
        while depth[i] > 1 and tree.parent[i] >= 0:
            i = int(tree.parent[i])
        return str(labels[i]) if labels is not None else str(i)
    return np.array([top(i) for i in range(len(tree))])


def atlas(coords, edges, labels=None, *, chart="lorentz", k=-1.0, color_by="depth",
          budget=10_000, reduction="radial", dim=2, views=None,
          show_centroids=None, max_centroids=12, seed=0):
    """Return a Scene: sampled, reduced, colored point cloud with hover/ancestor
    interaction. `edges` is a Tree or a list of (parent, child) pairs. `dim` is the
    reduction target: 2 → linked Poincaré-disk + hyperboloid views; 3 → the H³
    Poincaré-ball view. `show_centroids` ('depth' | 'clade' | a per-node label array)
    overlays each group's hyperbolic centroid; only the `max_centroids` largest are
    shown (disclosed if capped)."""
    views = views if views is not None else (("ball3d",) if dim == 3 else ("poincare", "lorentz"))
    tree = edges if isinstance(edges, Tree) else Tree.from_edges(edges, labels=labels)
    groups = None if show_centroids is None else _resolve_groups(tree, labels, show_centroids)
    full = Hierarchy(coords, tree, labels, chart=chart, k=k, groups=groups)
    orig_dim, n_full = full.dim, len(full)
    h = full.sample(budget, seed=seed).reduce(dim, reduction)

    scalar, leg = _ENCODE[color_by]
    col = colors.by_category(h.labels if h.labels is not None else h.depth()) if color_by == "label" \
        else colors.by_scalar(scalar(h))
    hover = [str(x) for x in h.labels] if h.labels is not None else [f"depth {d}" for d in h.depth()]
    cloud = Cloud(h.coords, col, labels=hover, parent=h.tree.parent, pruned=h.pruned_leaves)

    notes = []
    if n_full > len(h):
        notes.append(f"showing {len(h):,} of {n_full:,} nodes ({h.rate:.0%}) — hover for pruned-leaf counts")
    if orig_dim > dim or reduction == "tree":
        how = {"radial": "radius-preserving, depth↔radius exact",
               "tree": "radius = embedding distance-to-root; angle = tree layout",
               "horo": "horospherical / Busemann (HoroPCA, Chami et al. 2021)",
               "tangent": "tangent-space PCA"}[reduction]
        notes.append(f"{orig_dim}D → {dim}D ({how})")
    objs, legend = [cloud], [("point", "#3987e5", f"nodes — colored by {leg}")]
    if h.groups is not None:                              # overlay each group's hyperbolic centroid
        uniq, counts = np.unique(h.groups, return_counts=True)
        keep = uniq[np.argsort(-counts)[:max_centroids]]  # the largest groups only
        glabels, gc = _centroids(h.coords[np.isin(h.groups, keep)], h.groups[np.isin(h.groups, keep)], chart="lorentz", k=k)
        for lab, c in zip(glabels, gc):
            objs.append(Point(c[1:], chart="spatial", label=str(lab), color="#0b0b0b"))
        what = show_centroids if isinstance(show_centroids, str) else "group"
        legend.append(("point", "#0b0b0b", f"hyperbolic centroid per {what}"))
        if len(uniq) > len(keep):
            notes.append(f"centroids: the {len(keep)} largest of {len(uniq)} {what}s")

    hint = "Hover a node for its label and pruned-leaf count; click a node to highlight its ancestor chain."
    if notes:
        hint += "  " + " · ".join(notes) + "."
    return Scene(objs, views=views, curvature=k, legend=legend, hint=hint)
