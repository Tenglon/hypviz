"""The Embedding Atlas facade: an n-dimensional hyperbolic embedding plus its
hierarchy → a self-contained interactive Scene in one call. Wraps Hierarchy
(sample → reduce), color encoding, and honest disclosure of BOTH the sampling
(pruned-leaf counts) and the projection (which reduction, and its trade-off)."""
from . import colors
from .hierarchy import Hierarchy
from .scene import Cloud, Scene
from .tree import Tree

_ENCODE = {
    "depth": (lambda h: h.depth(), "depth (light = shallow)"),
    "norm": (lambda h: h.norm(), "distance from root"),
    "label": (None, "label"),
}


def atlas(coords, edges, labels=None, *, chart="lorentz", k=-1.0, color_by="depth",
          budget=10_000, reduction="radial", views=("poincare", "lorentz"), seed=0):
    """Return a Scene: sampled, 2D-reduced, colored point cloud with hover/ancestor
    interaction. `edges` is a Tree or a list of (parent, child) pairs."""
    tree = edges if isinstance(edges, Tree) else Tree.from_edges(edges, labels=labels)
    full = Hierarchy(coords, tree, labels, chart=chart, k=k)
    orig_dim, n_full = full.dim, len(full)
    h = full.sample(budget, seed=seed).reduce(2, reduction)

    scalar, leg = _ENCODE[color_by]
    col = colors.by_category(h.labels if h.labels is not None else h.depth()) if color_by == "label" \
        else colors.by_scalar(scalar(h))
    hover = [str(x) for x in h.labels] if h.labels is not None else [f"depth {d}" for d in h.depth()]
    cloud = Cloud(h.coords, col, labels=hover, parent=h.tree.parent, pruned=h.pruned_leaves)

    notes = []
    if n_full > len(h):
        notes.append(f"showing {len(h):,} of {n_full:,} nodes ({h.rate:.0%}) — hover for pruned-leaf counts")
    if orig_dim > 2 or reduction == "tree":
        how = {"radial": "radius-preserving, depth↔radius exact",
               "tree": "radius = embedding distance-to-root; angle = tree layout",
               "tangent": "tangent-space PCA"}[reduction]
        notes.append(f"{orig_dim}D → 2D ({how})")
    hint = "Hover a node for its label and pruned-leaf count; click a node to highlight its ancestor chain."
    if notes:
        hint += "  " + " · ".join(notes) + "."
    return Scene([cloud], views=views, curvature=k,
                 legend=[("point", "#3987e5", f"nodes — colored by {leg}")], hint=hint)
