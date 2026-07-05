import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L, mobius as M
from hypviz.kernel.charts import CHARTS, HalfPlane, Klein, Poincare

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")


@st.composite
def ball_points(draw, n=None, count=2):
    n = n or draw(st.integers(2, 5))
    raw = [draw(arrays(float, n, elements=st.floats(-3, 3))) for _ in range(count)]
    return [0.9 * r / (1 + np.linalg.norm(r)) for r in raw]  # always inside the ball


@given(ball_points())
def test_ball_charts_roundtrip_and_land_on_manifold(ps):
    for chart in (Poincare, Klein):
        x = chart.to_lorentz(ps[0])
        assert np.isclose(L.mdot(x, x), -1, atol=1e-9)
        assert np.allclose(chart.from_lorentz(x), ps[0], atol=1e-9)


@given(ball_points())
def test_poincare_chart_is_isometric(ps):
    p, q = ps
    assert np.isclose(M.dist(p, q), L.dist(Poincare.to_lorentz(p), Poincare.to_lorentz(q)), atol=1e-7)


@given(ball_points())
def test_klein_poincare_agree_through_lorentz(ps):
    k, p = Klein.from_lorentz(Poincare.to_lorentz(ps[0])), ps[0]
    assert np.allclose(Klein.to_lorentz(k), Poincare.to_lorentz(p), atol=1e-8)


@given(ball_points(n=2))
def test_halfplane_roundtrip_isometric_and_upper(ps):
    p, q = ps
    w = [HalfPlane.from_lorentz(Poincare.to_lorentz(v)) for v in (p, q)]
    assert all(v[..., 1] > 0 for v in w)  # lands in the upper half-plane
    assert np.allclose(HalfPlane.from_lorentz(HalfPlane.to_lorentz(w[0])), w[0], atol=1e-8)
    assert np.isclose(L.dist(HalfPlane.to_lorentz(w[0]), HalfPlane.to_lorentz(w[1])), M.dist(p, q), atol=1e-7)


def test_registry():
    assert set(CHARTS) == {"poincare", "klein", "halfplane"}
