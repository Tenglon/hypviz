"""V3 harness — render a sampled synthetic 128D hierarchy as an atlas cloud."""
from hypviz import colors, synth
from hypviz.hierarchy import Hierarchy
from hypviz.scene import Cloud, Scene

t = synth.taxonomy(30_000, seed=0)
coords = synth.diffuse(t, dim=128, k=-1.0, seed=0)
h = Hierarchy(coords, t, t.labels).sample(4000, seed=0).reduce(2, "radial")

cloud = Cloud(h.coords, colors.by_scalar(h.depth()),
              labels=[f"depth {d}" for d in h.depth()],
              parent=h.tree.parent, pruned=h.pruned_leaves)

Scene([cloud], views=("poincare", "lorentz"),
      legend=[("point", "#3987e5", "nodes — colored by depth (light = shallow)")],
      hint=(f"A synthetic 128D hyperbolic embedding of a {len(t):,}-node taxonomy, "
            f"radius-reduced to 2D and sampled to {len(h):,} nodes. Hover a node for its label and "
            "pruned-leaf count; click a node to highlight its ancestor chain to the root.")
      ).to_html("examples/out/atlas_synth.html", title="Embedding Atlas — synthetic 128D taxonomy")
print(f"wrote examples/out/atlas_synth.html  ({len(h):,} nodes, sampling rate {h.rate:.1%})")
