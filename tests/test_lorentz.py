import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")

coords = st.floats(-3, 3)


@st.composite
def two_points_and_tangent(draw):
    n = draw(st.integers(2, 5))
    x = L.from_spatial(draw(arrays(float, n, elements=coords)))
    y = L.from_spatial(draw(arrays(float, n, elements=coords)))
    v = L.to_tangent(x, draw(arrays(float, n + 1, elements=coords)))
    v = v / np.maximum(1, np.sqrt(np.maximum(L.mdot(v, v), 0)) / 3)  # cap |v|_L at 3 (teaching scale)
    return x, y, v


@given(two_points_and_tangent())
def test_expmap_stays_on_manifold(xyv):
    x, _, v = xyv
    y = L.expmap(x, v)
    assert np.isclose(L.mdot(y, y), -1, atol=1e-8)


@given(two_points_and_tangent())
def test_exp_log_roundtrip(xyv):
    x, y, _ = xyv
    assert np.allclose(L.expmap(x, L.logmap(x, y)), y, rtol=1e-7, atol=1e-7)


@given(two_points_and_tangent())
def test_dist_symmetric_and_zero_on_diagonal(xyv):
    x, y, _ = xyv
    assert np.isclose(L.dist(x, y), L.dist(y, x))
    assert L.dist(x, x) < 1e-6


@given(two_points_and_tangent())
def test_geodesic_endpoints(xyv):
    x, y, _ = xyv
    assert np.allclose(L.geodesic(x, y, 0.0), x, rtol=1e-7, atol=1e-7)
    assert np.allclose(L.geodesic(x, y, 1.0), y, rtol=1e-7, atol=1e-7)


@given(two_points_and_tangent())
def test_ptransp_is_isometric_into_target_tangent(xyv):
    x, y, v = xyv
    w = L.ptransp(x, y, v)
    assert np.isclose(L.mdot(y, w), 0, atol=1e-7)          # lands in T_y
    assert np.isclose(L.mdot(w, w), L.mdot(v, v), atol=1e-6)  # preserves norm
