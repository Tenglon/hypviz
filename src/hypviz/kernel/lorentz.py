"""Lorentz (hyperboloid) model of H^n with curvature K < 0 — the computational hub.

Points live on {x in R^{n+1} : <x,x>_L = -R^2, x0 > 0} with R = 1/sqrt(-K) and
the Minkowski inner product <x,y>_L = -x0*y0 + sum_i xi*yi. All functions
broadcast over leading axes; the last axis is the ambient dimension n+1.
Formulas follow Nickel & Kiela (2018) / standard Riemannian geometry texts; the
only guards are the 0/0 limits of the formulas themselves (eps in a
denominator, arccosh domain).
"""
import numpy as np

EPS = 1e-15


def _R(k):
    return 1 / np.sqrt(-k)


def mdot(x, y, keepdims=False):
    """Minkowski inner product <x,y>_L."""
    s = np.sum(x * y, -1) - 2 * x[..., 0] * y[..., 0]
    return s[..., None] if keepdims else s


def origin(n, k=-1.0):
    """The hyperboloid apex (R, 0, ..., 0)."""
    return _R(k) * np.eye(n + 1)[0]


def from_spatial(xs, k=-1.0):
    """Lift spatial coordinates onto the hyperboloid: x0 = sqrt(R^2+|xs|^2)."""
    x0 = np.sqrt(_R(k) ** 2 + np.sum(xs**2, -1, keepdims=True))
    return np.concatenate([x0, xs], -1)


def to_tangent(x, u, k=-1.0):
    """Project an ambient vector u onto the tangent space at x."""
    return u + mdot(x, u, True) / _R(k) ** 2 * x


def dist(x, y, k=-1.0):
    """Geodesic distance d(x,y) = R arccosh(-<x,y>_L / R^2)."""
    R = _R(k)
    return R * np.arccosh(np.maximum(-mdot(x, y) / R**2, 1.0))


def expmap(x, v, k=-1.0):
    """exp_x(v) = cosh(|v|/R) x + R sinh(|v|/R) v/|v|, for v in T_x."""
    R = _R(k)
    t = np.sqrt(np.maximum(mdot(v, v, True), 0))
    return np.cosh(t / R) * x + R * np.sinh(t / R) / np.maximum(t, EPS) * v


def logmap(x, y, k=-1.0):
    """Inverse of expmap; the R factors cancel except inside a."""
    a = np.maximum(-mdot(x, y, True) / _R(k) ** 2, 1.0)
    return np.arccosh(a) / np.maximum(np.sqrt(a**2 - 1), EPS) * (y - a * x)


def ptransp(x, y, v, k=-1.0):
    """Parallel transport of v in T_x along the geodesic to T_y."""
    return v + mdot(y, v, True) / (_R(k) ** 2 - mdot(x, y, True)) * (x + y)


def geodesic(x, y, t, k=-1.0):
    """Points gamma(t) on the geodesic with gamma(0)=x, gamma(1)=y; t broadcasts."""
    return expmap(x, np.asarray(t)[..., None] * logmap(x, y, k), k)
