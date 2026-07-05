"""Coordinate charts on H^n. Each chart only knows how to map its coordinates
to/from the Lorentz hub; every operation is computed once in Lorentz
coordinates and rendered in any chart."""
import numpy as np

from .lorentz import from_spatial  # noqa: F401  (re-exported for convenience)


def _sq(p):
    return np.sum(p**2, -1, keepdims=True)


class Poincare:
    """Unit ball, conformal (angles are true). x = (1+|p|^2, 2p) / (1-|p|^2)."""
    name = "poincare"

    @staticmethod
    def to_lorentz(p):
        d = 1 - _sq(p)
        return np.concatenate([(1 + _sq(p)) / d, 2 * p / d], -1)

    @staticmethod
    def from_lorentz(x):
        return x[..., 1:] / (1 + x[..., :1])


class Klein:
    """Unit ball, geodesics are straight chords. x = (1, k) / sqrt(1-|k|^2)."""
    name = "klein"

    @staticmethod
    def to_lorentz(k):
        g = 1 / np.sqrt(1 - _sq(k))
        return np.concatenate([g, g * k], -1)

    @staticmethod
    def from_lorentz(x):
        return x[..., 1:] / x[..., :1]


class HalfPlane:
    """Upper half-plane (2D only), via the Cayley map w = i(1+z)/(1-z)."""
    name = "halfplane"

    @staticmethod
    def to_lorentz(w):
        w = w[..., 0] + 1j * w[..., 1]
        z = (w - 1j) / (w + 1j)
        return Poincare.to_lorentz(np.stack([z.real, z.imag], -1))

    @staticmethod
    def from_lorentz(x):
        p = Poincare.from_lorentz(x)
        z = p[..., 0] + 1j * p[..., 1]
        w = 1j * (1 + z) / (1 - z)
        return np.stack([w.real, w.imag], -1)


CHARTS = {c.name: c for c in (Poincare, Klein, HalfPlane)}
