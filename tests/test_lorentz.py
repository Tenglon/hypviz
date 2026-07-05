import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")

coords = st.floats(-3, 3)
curvatures = st.sampled_from([-1.0, -0.5, -2.0])


@st.composite
def scene(draw):
    n, k = draw(st.integers(2, 5)), draw(curvatures)
    x = L.from_spatial(draw(arrays(float, n, elements=coords)), k)
    y = L.from_spatial(draw(arrays(float, n, elements=coords)), k)
    v = L.to_tangent(x, draw(arrays(float, n + 1, elements=coords)), k)
    v = v / np.maximum(1, np.sqrt(np.maximum(L.mdot(v, v), 0)) / 3)  # cap |v|_L at 3 (teaching scale)
    return x, y, v, k


@given(scene())
def test_expmap_stays_on_manifold(s):
    x, _, v, k = s
    y = L.expmap(x, v, k)
    assert np.isclose(L.mdot(y, y), -1 / -k, atol=1e-8)  # <y,y>_L = -R^2


@given(scene())
def test_exp_log_roundtrip(s):
    x, y, _, k = s
    assert np.allclose(L.expmap(x, L.logmap(x, y, k), k), y, rtol=1e-7, atol=1e-7)


@given(scene())
def test_dist_symmetric_and_zero_on_diagonal(s):
    x, y, _, k = s
    assert np.isclose(L.dist(x, y, k), L.dist(y, x, k))
    assert L.dist(x, x, k) < 1e-6


@given(scene())
def test_geodesic_endpoints(s):
    x, y, _, k = s
    assert np.allclose(L.geodesic(x, y, 0.0, k), x, rtol=1e-7, atol=1e-7)
    assert np.allclose(L.geodesic(x, y, 1.0, k), y, rtol=1e-7, atol=1e-7)


@given(scene())
def test_ptransp_is_isometric_into_target_tangent(s):
    x, y, v, k = s
    w = L.ptransp(x, y, v, k)
    assert np.isclose(L.mdot(y, w), 0, atol=1e-7)             # lands in T_y
    assert np.isclose(L.mdot(w, w), L.mdot(v, v), atol=1e-6)  # preserves norm
