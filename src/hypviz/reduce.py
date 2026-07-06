"""Dimensionality reduction for hyperbolic embeddings.

`tangent_pca` is the naive baseline: log at the Fréchet mean, PCA in the
(Minkowski-orthonormal) tangent space, exp the top components back into a
lower-dimensional hyperbolic space. It is the predecessor of HoroPCA (v2.1) and
CO-SNE (v2.2); the projection is always disclosed in the figure legend (via
`explained_variance_ratio`), never applied silently.

CAVEAT (measured): because PCA is linear, the RADIAL coordinate (‖x‖, i.e. tree
depth) is only preserved when the embedding's angular spread is ~2-dimensional.
For genuinely high-intrinsic-dimension data the depth↔radius signal is lost even
at high explained variance. `radial_pca` fixes this for hierarchical data by
preserving the radius exactly; it is the atlas default for `Hierarchy` inputs.
"""
import numpy as np

from .kernel import lorentz as L
from .kernel.adapters import as_numpy


def tangent_pca(xs, dim=2, k=-1.0):
    """Project points on H^n (Lorentz coords) to H^dim via tangent-space PCA.

    Returns (H^dim points, info) where info carries the mean and explained
    variance ratio so callers can report projection quality.
    """
    xs = as_numpy(xs)
    m = L.frechet_mean(xs, k)
    basis = L.tangent_basis(m, k)                      # (n, n+1)
    v = L.logmap(m, xs, k)                              # (N, n+1) tangent vectors at m
    coords = v @ basis.T - 2 * np.outer(v[:, 0], basis[:, 0])   # <v, e_j>_L  -> (N, n)
    coords = coords - coords.mean(0)
    _, s, wt = np.linalg.svd(coords, full_matrices=False)
    reduced = coords @ wt[:dim].T                       # (N, dim) principal tangent coords
    pts = L.expmap(L.origin(dim, k), np.concatenate([np.zeros((len(reduced), 1)), reduced], -1), k)
    var = s**2
    info = {"mean": m, "explained_variance_ratio": float(var[:dim].sum() / var.sum())}
    return pts, info


def radial_pca(xs, dim=2, k=-1.0, center=None):
    """Radius-preserving reduction for hierarchical data: keep each point's exact
    geodesic distance to `center` (default the origin ≈ tree root), lay out the
    ANGULAR part with PCA, then re-place each point at its true radius. Preserves
    depth↔radius by construction (the #1 feature for hierarchy viz); the angular
    layout is the lossy part. Returns (H^dim points, info)."""
    xs = as_numpy(xs)
    c = L.origin(xs.shape[-1] - 1, k) if center is None else as_numpy(center)
    v = L.logmap(c, xs, k)
    r = np.sqrt(np.maximum(L.mdot(v, v, True), 0))                # (N, 1) = d(center, x)
    basis = L.tangent_basis(c, k)
    ang = v @ basis.T - 2 * np.outer(v[:, 0], basis[:, 0])
    ang = ang / np.maximum(np.linalg.norm(ang, axis=1, keepdims=True), 1e-12)
    ang = ang - ang.mean(0)                          # remove the mean direction (cone axis)
    _, s, wt = np.linalg.svd(ang, full_matrices=False)
    d2 = ang @ wt[:dim].T                            # project the CENTERED dirs → clades spread full circle
    d2 = r * d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)  # restore true radius
    pts = L.expmap(L.origin(dim, k), np.concatenate([np.zeros((len(d2), 1)), d2], -1), k)
    var = s**2
    info = {"center": c, "radius_preserved": True,
            "angular_variance_ratio": float(var[:dim].sum() / var.sum())}
    return pts, info
