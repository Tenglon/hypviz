import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L, mobius as M
from hypviz.kernel.charts import Poincare

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")


@st.composite
def ball_pair(draw):
    n = draw(st.integers(2, 5))
    raw = [draw(arrays(float, n, elements=st.floats(-3, 3))) for _ in range(2)]
    return [0.9 * r / (1 + np.linalg.norm(r)) for r in raw]


@given(ball_pair())
def test_left_cancellation(ab):
    a, b = ab
    assert np.allclose(M.add(-a, M.add(a, b)), b, atol=1e-9)


@given(ball_pair())
def test_scalar_is_geodesic_scaling(ab):
    a, _ = ab
    assert np.allclose(M.add(a, a), M.scalar(2.0, a), atol=1e-9)  # a (+) a = 2 (x) a


@given(ball_pair())
def test_exp0_matches_lorentz_exp_at_apex(ab):
    a, _ = ab
    v = M.logmap0(a)                    # tangent at the ball origin
    assert np.allclose(M.expmap0(v), a, atol=1e-9)
    u = np.concatenate([[0.0], 2 * v])  # same vector in T_apex (factor 2 = conformal lambda_0)
    assert np.allclose(Poincare.from_lorentz(L.expmap(L.origin(len(a)), u)), a, atol=1e-8)


def test_noncommutative_example():
    a, b = np.array([0.5, 0.0]), np.array([0.0, 0.5])
    assert not np.allclose(M.add(a, b), M.add(b, a))
    # ...but the norms agree (gyration is an isometry)
    assert np.isclose(np.linalg.norm(M.add(a, b)), np.linalg.norm(M.add(b, a)))
