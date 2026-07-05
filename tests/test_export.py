from hypviz.scenes import GALLERY


def test_every_scene_exports_every_2d_view(tmp_path):
    for name, build in GALLERY.items():
        scene = build()
        for view in scene.views:
            if view in ("lorentz", "hemisphere"):  # 3D views: interactive only
                continue
            out = scene.to_svg(tmp_path / f"{name}_{view}.svg", view=view)
            assert out.stat().st_size > 5000  # a real drawing, not an empty canvas


def test_state_override_moves_a_point(tmp_path):
    from hypviz.scenes import geodesic
    scene = geodesic()
    a_id = scene.objects[0].id
    base = (tmp_path / "base.svg", None)
    moved = (tmp_path / "moved.svg", {"curvature": -0.5, "spatial": {a_id: [2.0, 1.0]}})
    for path, state in (base, moved):
        scene.to_svg(path, state=state)
    assert (tmp_path / "base.svg").read_text() != (tmp_path / "moved.svg").read_text()
