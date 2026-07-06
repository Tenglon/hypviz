"""The Embedding Atlas in ~10 lines: a synthetic 128D hierarchical embedding →
interactive HTML + the two full-data analysis figures."""
from hypviz import atlas, stats, synth

t = synth.taxonomy(30_000, seed=0)
coords = synth.diffuse(t, dim=128, k=-1.0, seed=0)         # 128D hyperbolic embedding
edges, depth = t.edges(), t.depth()

atlas(coords, edges, labels=[f"depth {d}" for d in depth], color_by="depth", budget=4000) \
    .to_html("examples/out/atlas_synth.html", title="Embedding Atlas — synthetic 128D taxonomy")

stats.norm_hist(coords).savefig("examples/out/atlas_norm_hist.svg", bbox_inches="tight")
stats.depth_norm(coords, depth).savefig("examples/out/atlas_depth_norm.svg", bbox_inches="tight")
print("wrote atlas_synth.html + atlas_norm_hist.svg + atlas_depth_norm.svg")
