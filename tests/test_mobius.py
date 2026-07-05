import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from hypviz.kernel import lorentz as L, mobius as M
from hypviz.kernel.charts import Poincare
from hypviz.kernel.lorentz import _R

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")

curvatures = st.sampled_from([-1.0, -0.5, -2.0])


@st.composite
def ball_pair(draw):
    n, k = draw(st.integers(2, 5)), draw(curvatures)
    raw = [draw(arrays(float, n, elements=st.floats(-3, 3))) for _ in range(2)]
    return [0.9 * _R(k) * r / (1 + np.linalg.norm(r)) for r in raw], k


@given(ball_pair())
def test_left_cancellation(s):
    (a, b), k = s
    assert np.allclose(M.add(-a, M.add(a, b, k), k), b, atol=1e-9)


@given(ball_pair())
def test_scalar_is_geodesic_scaling(s):
    (a, _), k = s
    assert np.allclose(M.add(a, a, k), M.scalar(2.0, a, k), atol=1e-9)  # a (+) a = 2 (x) a


@given(ball_pair())
def test_exp0_matches_lorentz_exp_at_apex(s):
    (a, _), k = s
    v = M.logmap0(a, k)                 # tangent at the ball origin
    assert np.allclose(M.expmap0(v, k), a, atol=1e-9)
    u = np.concatenate([[0.0], 2 * v])  # same vector in T_apex (factor 2 = conformal lambda_0)
    assert np.allclose(Poincare.from_lorentz(L.expmap(L.origin(len(a), k), u, k), k), a, atol=1e-8)


def test_noncommutative_example():
    a, b = np.array([0.5, 0.0]), np.array([0.0, 0.5])
    assert not np.allclose(M.add(a, b), M.add(b, a))
    # ...but the norms agree (gyration is an isometry)
    assert np.isclose(np.linalg.norm(M.add(a, b)), np.linalg.norm(M.add(b, a)))
