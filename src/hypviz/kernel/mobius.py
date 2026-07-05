"""Gyrovector operations native to the Poincare ball of curvature K < 0
(c = -K), after Ungar / Ganea et al. (2018). Dimension-agnostic; last axis is
the ball dimension."""
import numpy as np

EPS = 1e-15


def _sq(p):
    return np.sum(p**2, -1, keepdims=True)


def _dot(a, b):
    return np.sum(a * b, -1, keepdims=True)


def add(a, b, k=-1.0):
    """Mobius addition a (+) b — non-commutative, non-associative."""
    c = -k
    ab, a2, b2 = _dot(a, b), _sq(a), _sq(b)
    return ((1 + 2 * c * ab + c * b2) * a + (1 - c * a2) * b) / (1 + 2 * c * ab + c**2 * a2 * b2)


def scalar(r, a, k=-1.0):
    """Mobius scalar multiplication r (x) a."""
    n = np.sqrt(-k * _sq(a))
    return np.tanh(r * np.arctanh(n)) / np.maximum(n, EPS) * a


def gyr(a, b, v, k=-1.0):
    """Gyration gyr[a,b]v = -(a(+)b) (+) (a (+) (b (+) v))."""
    return add(-add(a, b, k), add(a, add(b, v, k), k), k)


def dist(p, q, k=-1.0):
    """d(p,q) = (2/sqrt(-K)) artanh(sqrt(-K)|(-p) (+) q|) — equals the Lorentz distance."""
    return 2 / np.sqrt(-k) * np.arctanh(np.sqrt(-k) * np.linalg.norm(add(-p, q, k), axis=-1))


def expmap0(v, k=-1.0):
    """exp map at the ball origin."""
    n = np.sqrt(-k * _sq(v))
    return np.tanh(n) / np.maximum(n, EPS) * v


def logmap0(p, k=-1.0):
    """Inverse of expmap0."""
    n = np.sqrt(-k * _sq(p))
    return np.arctanh(n) / np.maximum(n, EPS) * p
