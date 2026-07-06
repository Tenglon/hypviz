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


@given(curvatures)
def test_gyroplane_signed_distance(k):
    # the math the gyroplane scene draws: normal m = unit(w), plane = {y:<y,m>=0},
    # signed distance to it = asinh(<x,m>_L).
    from hypviz.kernel.charts import Poincare
    R = 1 / np.sqrt(-k)
    p = Poincare.to_lorentz(np.array([0.1, 0.15]) * R, k)
    h = Poincare.to_lorentz(np.array([0.55, 0.35]) * R, k)
    w = L.logmap(p, h, k)
    m = w / np.sqrt(L.mdot(w, w))
    assert np.isclose(L.mdot(p, m), 0, atol=1e-9)               # p lies on the plane
    for d in (0.6, -1.1):                                       # push d perpendicular → signed dist d
        y = L.expmap(p, d * m, k)
        assert np.isclose(np.arcsinh(L.mdot(y, m)), d, atol=1e-8)


@given(curvatures)
def test_holonomy_equals_angle_deficit(k):
    # Gauss-Bonnet: transporting a vector around a geodesic triangle rotates it by
    # the angle deficit π − (α+β+γ) — the math the parallel-transport scene shows.
    from hypviz.kernel.charts import Poincare
    unit = lambda u: u / np.sqrt(L.mdot(u, u))
    ang = lambda a, b: np.arccos(np.clip(L.mdot(a, b) / np.sqrt(L.mdot(a, a) * L.mdot(b, b)), -1, 1))
    x, y, z = (Poincare.to_lorentz(np.array(p), k) for p in ([0, 0.35], [-0.4, -0.3], [0.45, -0.3]))
    v = v0 = unit(L.logmap(x, y, k))
    for a, b in [(x, y), (y, z), (z, x)]:
        v = L.ptransp(a, b, v, k)
    deficit = np.pi - (ang(L.logmap(x, y, k), L.logmap(x, z, k))
                       + ang(L.logmap(y, x, k), L.logmap(y, z, k))
                       + ang(L.logmap(z, x, k), L.logmap(z, y, k)))
    assert np.isclose(ang(v0, v), abs(deficit), atol=1e-6)
