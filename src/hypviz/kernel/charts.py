"""Coordinate charts on H^n with curvature K < 0 (ball/half-plane scale R).
Each chart only knows how to map its coordinates to/from the Lorentz hub;
every operation is computed once in Lorentz coordinates and rendered in any
chart."""
import numpy as np

from .lorentz import _R, from_spatial  # noqa: F401  (re-exported for convenience)


def _sq(p):
    return np.sum(p**2, -1, keepdims=True)


class Poincare:
    """Ball of radius R, conformal (angles are true)."""
    name = "poincare"

    @staticmethod
    def to_lorentz(p, k=-1.0):
        R = _R(k)
        d = R**2 - _sq(p)
        return np.concatenate([R * (R**2 + _sq(p)) / d, 2 * R**2 * p / d], -1)

    @staticmethod
    def from_lorentz(x, k=-1.0):
        return _R(k) * x[..., 1:] / (_R(k) + x[..., :1])


class Klein:
    """Ball of radius R, geodesics are straight chords."""
    name = "klein"

    @staticmethod
    def to_lorentz(p, k=-1.0):
        R = _R(k)
        g = 1 / np.sqrt(1 - _sq(p) / R**2)
        return np.concatenate([g * R, g * p], -1)

    @staticmethod
    def from_lorentz(x, k=-1.0):
        return _R(k) * x[..., 1:] / x[..., :1]


class HalfPlane:
    """Upper half-plane (2D only), via the Cayley map scaled to radius R."""
    name = "halfplane"

    @staticmethod
    def to_lorentz(w, k=-1.0):
        R = _R(k)
        z = (w[..., 0] + 1j * w[..., 1]) / R
        z = (z - 1j) / (z + 1j)
        return Poincare.to_lorentz(R * np.stack([z.real, z.imag], -1), k)

    @staticmethod
    def from_lorentz(x, k=-1.0):
        R = _R(k)
        p = Poincare.from_lorentz(x, k)
        z = (p[..., 0] + 1j * p[..., 1]) / R
        w = 1j * (1 + z) / (1 - z)
        return R * np.stack([w.real, w.imag], -1)


CHARTS = {c.name: c for c in (Poincare, Klein, HalfPlane)}
