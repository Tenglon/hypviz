"""Lorentz (hyperboloid) model of H^n, curvature K=-1 — the computational hub.

Points live on {x in R^{n+1} : <x,x>_L = -1, x0 > 0} with the Minkowski inner
product <x,y>_L = -x0*y0 + sum_i xi*yi. All functions broadcast over leading
axes; the last axis is the ambient dimension n+1. Formulas follow Nickel &
Kiela (2018) / standard Riemannian geometry texts; the only guards are the
0/0 limits of the formulas themselves (eps in a denominator, arccosh domain).
"""
import numpy as np

EPS = 1e-15


def mdot(x, y, keepdims=False):
    """Minkowski inner product <x,y>_L."""
    s = np.sum(x * y, -1) - 2 * x[..., 0] * y[..., 0]
    return s[..., None] if keepdims else s


def origin(n):
    """The hyperboloid apex (1, 0, ..., 0) in H^n."""
    return np.eye(n + 1)[0]


def from_spatial(xs):
    """Lift spatial coordinates onto the hyperboloid: x0 = sqrt(1+|xs|^2)."""
    x0 = np.sqrt(1 + np.sum(xs**2, -1, keepdims=True))
    return np.concatenate([x0, xs], -1)


def to_tangent(x, u):
    """Project an ambient vector u onto the tangent space at x."""
    return u + mdot(x, u, True) * x


def dist(x, y):
    """Geodesic distance d(x,y) = arccosh(-<x,y>_L)."""
    return np.arccosh(np.maximum(-mdot(x, y), 1.0))


def expmap(x, v):
    """exp_x(v) = cosh(|v|) x + sinh(|v|)/|v| v, for v in T_x."""
    t = np.sqrt(np.maximum(mdot(v, v, True), 0))
    return np.cosh(t) * x + np.sinh(t) / np.maximum(t, EPS) * v


def logmap(x, y):
    """Inverse of expmap: the tangent vector at x pointing to y."""
    a = np.maximum(-mdot(x, y, True), 1.0)
    return np.arccosh(a) / np.maximum(np.sqrt(a**2 - 1), EPS) * (y - a * x)


def ptransp(x, y, v):
    """Parallel transport of v in T_x along the geodesic to T_y."""
    return v + mdot(y, v, True) / (1 - mdot(x, y, True)) * (x + y)


def geodesic(x, y, t):
    """Points gamma(t) on the geodesic with gamma(0)=x, gamma(1)=y; t broadcasts."""
    return expmap(x, np.asarray(t)[..., None] * logmap(x, y))
