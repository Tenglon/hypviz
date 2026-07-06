from hypviz import Tree, atlas, embed, stats, synth
from hypviz.scene import Cloud


def _data(n=4000, dim=64, seed=0):
    t = synth.taxonomy(n, seed=seed)
    return t, synth.diffuse(t, dim=dim, k=-1.0, seed=seed)


def test_atlas_builds_a_cloud_scene_with_disclosure():
    t, coords = _data()
    scene = atlas(coords, t.edges(), labels=t.labels, budget=800)
    clouds = [o for o in scene.objects if isinstance(o, Cloud)]
    assert len(clouds) == 1
    assert len(clouds[0].spatial) <= 800
    # both the sampling and the 64D→2D projection are disclosed in the hint
    assert "of" in scene.hint and "nodes" in scene.hint
    assert "64D → 2D" in scene.hint


def test_atlas_color_by_variants(tmp_path):
    t, coords = _data(seed=1)
    for cb in ("depth", "norm", "label"):
        scene = atlas(coords, t.edges(), labels=t.labels, color_by=cb, budget=500)
        cloud = next(o for o in scene.objects if isinstance(o, Cloud))
        assert len(cloud.colors) == len(cloud.spatial)
        out = scene.to_html(tmp_path / f"{cb}.html", title=cb)
        assert out.stat().st_size > 50_000


def test_atlas_from_edge_list_and_2d_input_skips_projection():
    # a 2D embedding needs no projection → no projection note
    t = synth.taxonomy(600, seed=2)
    coords2d = synth.diffuse(t, dim=2, k=-1.0, seed=2)
    scene = atlas(coords2d, t.edges(), budget=9999)
    assert "→ 2D" not in scene.hint          # already 2D
    assert "showing" not in scene.hint       # budget over tree → no sampling note


def test_tree_from_paths_merges_shared_prefixes():
    paths = [("Carnivora", "Felidae", "P. leo"), ("Carnivora", "Felidae", "F. catus"),
             ("Carnivora", "Canidae", "C. lupus"), ("Primates", "Hominidae", "H. sapiens")]
    t = Tree.from_paths(paths, root_name="Mammalia")
    # 1 root + 2 orders + 3 families + 4 species = 10 nodes; the shared "Felidae" is one node
    assert len(t) == 10
    assert list(t.labels[[t.root]]) == ["Mammalia"]
    leaves = [i for i in range(len(t)) if not t.children[i]]
    assert len(leaves) == 4 and all(t.depth()[i] == 3 for i in leaves)


def test_atlas_from_sarkar_taxonomy_end_to_end(tmp_path):
    paths = [("A", "A1", "x"), ("A", "A1", "y"), ("A", "A2", "z"), ("B", "B1", "w")]
    tree = Tree.from_paths(paths)
    coords = embed.sarkar(tree, tau=1.1)                 # 2D Poincaré, no reduction
    scene = atlas(coords, tree, labels=tree.labels, chart="poincare")
    out = scene.to_html(tmp_path / "tax.html", title="tax")
    assert out.stat().st_size > 50_000


def test_stats_figures_render():
    t, coords = _data(n=2000, dim=32, seed=3)
    assert len(stats.norm_hist(coords).axes[0].patches) > 0
    assert len(stats.depth_norm(coords, t.depth()).axes[0].get_lines()) > 0


def test_stats_honor_input_chart():
    # Sarkar returns Poincaré coords; feeding them as Lorentz collapses every norm to 0
    tree = Tree.from_paths([("A", "A1", "x"), ("A", "A1", "y"), ("A", "A2", "z"), ("B", "B1", "w")])
    poincare = embed.sarkar(tree, tau=1.0)
    n_norms = len(stats.norm_hist(poincare, chart="poincare").axes[0].patches)
    assert n_norms > 0
    # depth↔norm must actually vary with depth (the bug made it flat at 0)
    ax = stats.depth_norm(poincare, tree.depth(), chart="poincare").axes[0]
    medians = [line.get_ydata()[0] for line in ax.get_lines() if len(line.get_ydata())]
    assert max(medians) > 0.5
