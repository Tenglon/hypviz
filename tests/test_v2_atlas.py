from hypviz import atlas, stats, synth
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


def test_stats_figures_render():
    t, coords = _data(n=2000, dim=32, seed=3)
    assert len(stats.norm_hist(coords).axes[0].patches) > 0
    assert len(stats.depth_norm(coords, t.depth()).axes[0].get_lines()) > 0
