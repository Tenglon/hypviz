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

from .kernel import lorentz as L, mobius as M
from .kernel.adapters import as_numpy
from .kernel.charts import Poincare


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


def horo_pca(coords, dim=2, k=-1.0):
    """Horospherical reduction, after Chami et al. (2021, HoroPCA). Center the data
    at its Fréchet mean (Möbius), pick `dim` boundary ideal directions, and reduce
    each point to its Busemann coordinates B_p(x) = ln(‖p−x‖²/(1−‖x‖²)) along them —
    a horospherical projection rather than the tangent linearization. A principled
    alternative to `tangent_pca` whose benefit is data-dependent (it helps most when
    the tangent linearization is poor). This is the Busemann-coordinate form used in
    the paper's whitening application; the full method adds a greedy geodesic-hull
    submanifold projection. (Poincaré-ball math ⇒ k = -1.)"""
    if not np.isclose(k, -1.0):
        raise ValueError("horo_pca currently supports curvature k = -1")
    xs = as_numpy(coords)
    u0 = Poincare.from_lorentz(xs, k)                                    # unit-ball coords
    mean = Poincare.from_lorentz(L.frechet_mean(xs, k)[None], k)[0]
    u = M.add(np.broadcast_to(-mean, u0.shape), u0, k)                   # center: mean → origin
    _, s, wt = np.linalg.svd(u - u.mean(0), full_matrices=False)         # ideal directions ≈ top PCs
    Q = wt[:dim]
    sq = np.sum(u * u, -1)
    busemann = np.stack([np.log(np.maximum(1 - sq, 1e-12)) - np.log(np.sum((q - u) ** 2, -1)) for q in Q], -1)
    busemann = np.clip(busemann, -6, 6)                                 # keep off the ideal boundary
    pts = L.expmap(L.origin(dim, k), np.concatenate([np.zeros((len(busemann), 1)), busemann], -1), k)
    info = {"directions": Q, "explained_variance_ratio": float((s[:dim] ** 2).sum() / (s**2).sum())}
    return pts, info


def _perplexity_probs(dist, perplexity, steps=60):
    """Symmetric joint P from pairwise distances via per-point perplexity (t-SNE);
    each row's bandwidth β is binary-searched to hit the target entropy."""
    n = len(dist)
    d2 = dist**2
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        others = np.arange(n) != i
        di = d2[i][others]                                     # self excluded — no inf/0·inf
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(steps):
            w = np.exp(-di * beta)
            sw = w.sum()
            entropy = np.log(sw) + beta * (di * w).sum() / sw
            if entropy > target:
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        w = np.exp(-di * beta)
        P[i, others] = w / w.sum()
    return (P + P.T) / (2 * n)


def co_sne(coords, dim=2, k=-1.0, perplexity=30, iters=1000, lr=0.3, gamma=0.1,
           l1=10.0, l2=0.01, exaggerate=12.0, exag_iters=250, norm_after=250, grad_clip=0.1, seed=0):
    """CO-SNE (Guo et al. 2022): hyperbolic t-SNE. Preserves both local neighborhoods
    (KL between hyperbolic-distance similarities, low-dim a hyperbolic Cauchy) AND the
    distance-to-origin / hierarchy (the ‖x‖²−‖y‖² norm term). Riemannian SGD in the
    Poincaré ball with early exaggeration + momentum, initialized from tangent PCA.
    Requires torch (optional). O(N²) — use on ≤ ~1500 points. k=-1."""
    if not np.isclose(k, -1.0):
        raise ValueError("co_sne currently supports curvature k = -1")
    import torch

    xs = as_numpy(coords)
    P = _perplexity_probs(L.dist(xs[:, None], xs[None], k), perplexity)
    nx = np.sum(Poincare.from_lorentz(xs, k) ** 2, -1)                 # high-dim ball norm²

    y0 = Poincare.from_lorentz(tangent_pca(xs, dim, k)[0], k)          # structured start
    y0 = 0.3 * y0 / (np.abs(y0).max() + 1e-9)
    Pt, nxt = torch.tensor(np.maximum(P, 1e-12)), torch.tensor(nx)
    Y = torch.tensor(y0, requires_grad=True)
    vel = torch.zeros_like(Y)

    def d2_poincare(y):
        n = (y * y).sum(1)
        sq = ((y[:, None, :] - y[None, :, :]) ** 2).sum(2)
        arg = 1 + 2 * sq / ((1 - n[:, None]) * (1 - n[None, :])).clamp_min(1e-12)
        return torch.arccosh(arg.clamp_min(1 + 1e-12)) ** 2

    for it in range(iters):
        if Y.grad is not None:
            Y.grad.zero_()
        peff = Pt * (exaggerate if it < exag_iters else 1.0)          # early exaggeration
        w = gamma**2 / (d2_poincare(Y) + gamma**2)
        w = w - torch.diag(torch.diag(w))
        Q = (w / w.sum()).clamp_min(1e-12)
        loss = l1 * (peff * (peff.log() - Q.log())).sum()
        if it >= norm_after:
            loss = loss + l2 * ((nxt - (Y * Y).sum(1)) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            rgrad = ((1 - (Y * Y).sum(1, keepdim=True)) ** 2 / 4) * Y.grad   # Riemannian gradient
            g = rgrad.norm(dim=1, keepdim=True)                              # clip: the metric blows
            rgrad = torch.where(g > grad_clip, rgrad * grad_clip / g, rgrad) # up near the boundary
            vel = (0.5 if it < exag_iters else 0.8) * vel - lr * rgrad       # momentum
            Y += vel
            nrm = Y.norm(dim=1, keepdim=True)                                # retract into the ball
            Y.data = torch.where(nrm > 0.999, Y * 0.999 / nrm, Y)

    return Poincare.to_lorentz(Y.detach().numpy(), k), {"perplexity": perplexity}
