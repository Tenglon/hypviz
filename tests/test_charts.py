import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L, mobius as M
from hypviz.kernel.charts import CHARTS, HalfPlane, Klein, Poincare
from hypviz.kernel.lorentz import _R

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")

curvatures = st.sampled_from([-1.0, -0.5, -2.0])


@st.composite
def ball_points(draw, n=None, count=2):
    n, k = n or draw(st.integers(2, 5)), draw(curvatures)
    raw = [draw(arrays(float, n, elements=st.floats(-3, 3))) for _ in range(count)]
    return [0.9 * _R(k) * r / (1 + np.linalg.norm(r)) for r in raw], k  # inside the R-ball


@given(ball_points())
def test_ball_charts_roundtrip_and_land_on_manifold(s):
    (p, _), k = s
    for chart in (Poincare, Klein):
        x = chart.to_lorentz(p, k)
        assert np.isclose(L.mdot(x, x), -_R(k) ** 2, atol=1e-9)
        assert np.allclose(chart.from_lorentz(x, k), p, atol=1e-9)


@given(ball_points())
def test_poincare_chart_is_isometric(s):
    (p, q), k = s
    assert np.isclose(M.dist(p, q, k), L.dist(Poincare.to_lorentz(p, k), Poincare.to_lorentz(q, k), k), atol=1e-7)


@given(ball_points())
def test_klein_poincare_agree_through_lorentz(s):
    (p, _), k = s
    kl = Klein.from_lorentz(Poincare.to_lorentz(p, k), k)
    assert np.allclose(Klein.to_lorentz(kl, k), Poincare.to_lorentz(p, k), atol=1e-8)


@given(ball_points(n=2))
def test_halfplane_roundtrip_isometric_and_upper(s):
    (p, q), k = s
    w = [HalfPlane.from_lorentz(Poincare.to_lorentz(v, k), k) for v in (p, q)]
    assert all(v[..., 1] > 0 for v in w)  # lands in the upper half-plane
    assert np.allclose(HalfPlane.from_lorentz(HalfPlane.to_lorentz(w[0], k), k), w[0], atol=1e-8)
    assert np.isclose(L.dist(HalfPlane.to_lorentz(w[0], k), HalfPlane.to_lorentz(w[1], k), k), M.dist(p, q, k), atol=1e-7)


def test_registry():
    assert set(CHARTS) == {"poincare", "klein", "halfplane"}
