"""The Embedding Atlas in ~10 lines: a synthetic 128D hierarchical embedding →
interactive HTML + the two full-data analysis figures."""
from hypviz import atlas, stats, synth


def build_scene():
    t = synth.taxonomy(30_000, seed=0)
    coords = synth.diffuse(t, dim=128, k=-1.0, seed=0)     # 128D hyperbolic embedding
    return atlas(coords, t.edges(), labels=[f"depth {d}" for d in t.depth()],
                 color_by="depth", budget=4000, reduction="tree")


if __name__ == "__main__":
    t = synth.taxonomy(30_000, seed=0)
    coords = synth.diffuse(t, dim=128, k=-1.0, seed=0)
    build_scene().to_html("examples/out/atlas_synth.html", title="Embedding Atlas — synthetic 128D taxonomy")
    stats.norm_hist(coords).savefig("examples/out/atlas_norm_hist.svg", bbox_inches="tight")
    stats.depth_norm(coords, t.depth()).savefig("examples/out/atlas_depth_norm.svg", bbox_inches="tight")
    stats.distortion(coords, t).savefig("examples/out/atlas_distortion.svg", bbox_inches="tight")
    stats.delta_hyperbolicity(coords).savefig("examples/out/atlas_delta.svg", bbox_inches="tight")
    stats.density_heatmaps(coords).savefig("examples/out/atlas_density.svg", bbox_inches="tight")
    print("wrote atlas_synth.html + norm_hist/depth_norm/distortion/delta/density figures")
