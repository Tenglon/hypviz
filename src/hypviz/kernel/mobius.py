"""Gyrovector operations native to the Poincare ball (c=1), after Ungar /
Ganea et al. (2018). Dimension-agnostic; last axis is the ball dimension."""
import numpy as np

EPS = 1e-15


def _sq(p):
    return np.sum(p**2, -1, keepdims=True)


def _dot(a, b):
    return np.sum(a * b, -1, keepdims=True)


def add(a, b):
    """Mobius addition a (+) b — non-commutative, non-associative."""
    ab, a2, b2 = _dot(a, b), _sq(a), _sq(b)
    return ((1 + 2 * ab + b2) * a + (1 - a2) * b) / (1 + 2 * ab + a2 * b2)


def scalar(r, a):
    """Mobius scalar multiplication r (x) a."""
    n = np.sqrt(_sq(a))
    return np.tanh(r * np.arctanh(n)) / np.maximum(n, EPS) * a


def gyr(a, b, v):
    """Gyration gyr[a,b]v = -(a(+)b) (+) (a (+) (b (+) v))."""
    return add(-add(a, b), add(a, add(b, v)))


def dist(p, q):
    """d(p,q) = 2 artanh|(-p) (+) q| — equals the Lorentz distance."""
    return 2 * np.arctanh(np.linalg.norm(add(-p, q), axis=-1))


def expmap0(v):
    """exp map at the ball origin: tanh(|v|) v/|v|."""
    n = np.sqrt(_sq(v))
    return np.tanh(n) / np.maximum(n, EPS) * v


def logmap0(p):
    """Inverse of expmap0."""
    n = np.sqrt(_sq(p))
    return np.arctanh(n) / np.maximum(n, EPS) * p
