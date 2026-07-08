"""Static, full-data analysis figures (matplotlib) — the paper 'analysis section'
staples. Computed on the FULL embedding, never the visualization sample, so the
numbers are exact even when the interactive scene shows a subsample."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .colors import CAT
from .kernel import lorentz as L
from .kernel.adapters import as_numpy
from .kernel.charts import CHARTS


def _norms(coords, k, chart):
    x = as_numpy(coords)
    if chart != "lorentz":
        x = CHARTS[chart].to_lorentz(x, k)
    return L.dist(x, L.origin(x.shape[-1] - 1, k), k)


def _bare(ax):
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def norm_hist(coords, k=-1.0, chart="lorentz", by=None, bins=40, size=(5.2, 3.2)):
    """Distribution of hyperbolic norm ‖x‖ (distance from the origin), optionally
    split by a per-node group label."""
    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    n = _norms(coords, k, chart)
    if by is None:
        ax.hist(n, bins=bins, color="#3987e5")
    else:
        by = np.asarray(by)
        for i, g in enumerate(dict.fromkeys(by)):
            ax.hist(n[by == g], bins=bins, color=CAT[i % len(CAT)], alpha=0.6, label=str(g))
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("hyperbolic norm  d(o, x)")
    ax.set_ylabel("count")
    fig.tight_layout()
    return fig


def depth_norm(coords, depths, k=-1.0, chart="lorentz", size=(5.2, 3.2)):
    """Tree depth vs hyperbolic norm (boxplot per depth) — the visual test of the
    'depth ≈ radius' claim that motivates hyperbolic embeddings."""
    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    n, depths = _norms(coords, k, chart), np.asarray(depths)
    lo, hi = int(depths.min()), int(depths.max())
    ax.boxplot([n[depths == d] for d in range(lo, hi + 1)], positions=range(lo, hi + 1),
               widths=0.6, showfliers=False, medianprops={"color": "#e34948"})
    ax.set_xlabel("tree depth")
    ax.set_ylabel("hyperbolic norm  d(o, x)")
    fig.tight_layout()
    return fig


def _to_lorentz(coords, chart, k):
    x = as_numpy(coords)
    return x if chart == "lorentz" else CHARTS[chart].to_lorentz(x, k)


def distortion(coords, tree, k=-1.0, chart="lorentz", n_pairs=4000, size=(5.2, 3.6), seed=0):
    """How faithfully the embedding reproduces the hierarchy: a scatter of tree
    (graph) distance vs embedding hyperbolic distance over sampled node pairs, with
    the best-fit scale. Reports the average relative distortion around that fit."""
    x = _to_lorentz(coords, chart, k)
    depth, rng = tree.depth(), np.random.default_rng(seed)
    anc = [set(tree.ancestors(i)) for i in range(len(tree))]

    def graph_dist(u, v):
        a = v
        while a not in anc[u]:
            a = int(tree.parent[a])
        return depth[u] + depth[v] - 2 * depth[a]

    u = rng.integers(0, len(tree), n_pairs)
    v = rng.integers(0, len(tree), n_pairs)
    keep = u != v
    u, v = u[keep], v[keep]
    g = np.array([graph_dist(int(a), int(b)) for a, b in zip(u, v)], float)
    e = L.dist(x[u], x[v], k)
    scale = float((g @ e) / (g @ g))                       # best embed ≈ scale · graph
    rel = np.abs(e / np.maximum(scale * g, 1e-9) - 1)

    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    ax.scatter(g, e, s=5, alpha=0.25, color="#3987e5", edgecolors="none")
    gg = np.array([g.min(), g.max()])
    ax.plot(gg, scale * gg, color="#e34948", lw=1.4, label=f"fit  (avg distortion {rel.mean():.1%})")
    ax.set_xlabel("tree (graph) distance")
    ax.set_ylabel("embedding hyperbolic distance")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


_BW = {"hyperbolic": 0.6, "euclidean": 0.12, "cosine": 0.15}   # kernel bandwidths


def _chart_grid(chart, R, grid):
    """Grid of chart coordinates + a validity mask + the imshow extent/shape."""
    if chart == "halfplane":
        gx, gy = np.meshgrid(np.linspace(-2 * R, 2 * R, grid), np.linspace(0.03 * R, 3 * R, grid))
        C = np.stack([gx.ravel(), gy.ravel()], -1)
        return C, np.ones(len(C), bool), (-2 * R, 2 * R, 0, 3 * R)
    a = np.linspace(-R, R, grid)
    gx, gy = np.meshgrid(a, a)
    C = np.stack([gx.ravel(), gy.ravel()], -1)
    return C, np.sum(C**2, 1) < (0.995 * R) ** 2, (-R, R, -R, R)


def _density_field(chart, metric, pts_lorentz, k, grid):
    """Normalized kernel density over `chart`'s UNIT grid (fixed display); hyperbolic
    lifts grid+points to Lorentz at curvature k for the geodesic distance (valid for
    k in [-1, 0) — the unit points stay inside the k-ball), euclidean/cosine use the
    chart coords. Chunked over grid cells to bound memory."""
    C, valid, extent = _chart_grid(chart, 1.0, grid)              # display fixed at unit scale
    Cv = C[valid]
    pp = None if metric == "hyperbolic" else CHARTS[chart].from_lorentz(pts_lorentz, k)
    pn = None if pp is None else np.linalg.norm(pp, axis=1)
    dens = np.empty(len(Cv))
    for s in range(0, len(Cv), 4000):
        g = Cv[s:s + 4000]
        if metric == "hyperbolic":
            d = L.dist(CHARTS[chart].to_lorentz(g, k)[:, None], pts_lorentz[None], k)
        elif metric == "cosine":
            d = 1 - (g @ pp.T) / np.maximum(np.linalg.norm(g, axis=1)[:, None] * pn[None], 1e-9)
        else:
            d = np.sqrt(np.sum((g[:, None] - pp[None]) ** 2, -1))
        dens[s:s + 4000] = np.exp(-(d**2) / (2 * _BW[metric] ** 2)).sum(1)
    field = np.full(len(C), np.nan)
    field[valid] = dens                                        # RAW density (caller normalizes)
    return field.reshape(grid, grid), extent


def _field_to_uri(field, vmax):
    """Render a raw density field to a magma RGBA PNG data URI, normalized by `vmax`
    (a shared max makes brightness comparable across panels)."""
    import base64
    from io import BytesIO

    import matplotlib as mpl

    rgba = mpl.colormaps["magma"](np.clip(np.nan_to_num(field / vmax, nan=0.0), 0, 1))
    rgba[..., 3] = np.where(np.isnan(field), 0.0, 1.0)
    buf = BytesIO()
    mpl.image.imsave(buf, rgba[::-1], format="png")            # flip rows → texture orientation
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def density_scene(coords, chart="poincare", charts=("poincare", "klein", "halfplane", "hyperboloid"),
                  metrics=("hyperbolic", "euclidean", "cosine"), curvatures=(-1.0, -0.5, -0.25, -0.1),
                  res=240, n_points=700, seed=0):
    """Interactive density comparison. The same fixed 2D points, their KDE rendered as a
    smooth, zoomable texture in each hyperbolic model (Poincaré / Klein / half-plane, and
    the hyperboloid as a textured surface — all isometric, so the density is identical and
    only the chart differs). A **kernel** switch picks the metric (hyperbolic / euclidean /
    cosine) and a **curvature** slider varies the hyperbolic kernel's K in [-1, 0): as K→0
    the hyperbolic density morphs toward the Euclidean one. >2D input is radius-reduced to
    2D at K=-1 first (fixing the point positions). Returns a Scene."""
    from . import reduce as _reduce
    from .scene import DensityField, Scene

    x = _to_lorentz(coords, chart, -1.0)
    if x.shape[-1] > 3:
        x, _ = _reduce.radial_pca(x, 2, -1.0)
    rng = np.random.default_rng(seed)
    if len(x) > n_points:
        x = x[rng.choice(len(x), n_points, replace=False)]
    p_unit = CHARTS["poincare"].from_lorentz(x, -1.0)          # fixed unit-disk positions

    # each panel is a well-defined (chart, metric): the hyperbolic density in every model
    # (curvature-controlled), and euclidean/cosine ONLY on the Poincaré disk — those metrics
    # are not intrinsic, so rendering them in Klein / half-plane coordinates is meaningless.
    panels = [(ch, "hyperbolic", True) for ch in charts]
    panels += [("poincare", m, False) for m in metrics if m != "hyperbolic"]

    # pass 1: raw density fields per panel/key; pass 2: normalize by a per-metric shared max
    # (all hyperbolic panels — 4 models × every curvature — share ONE scale, so the colorbar is
    #  meaningful and the curvature slide shows real brightness change; each other metric its own).
    raw, extents, vmax = {}, {}, {}
    for i, (ch, metric, curv) in enumerate(panels):
        tex_chart = "poincare" if ch == "hyperboloid" else ch
        keys = [(f"hyperbolic@{kc:g}", "hyperbolic", CHARTS["poincare"].to_lorentz(p_unit, kc), kc)
                for kc in curvatures] if curv else [(metric, metric, x, -1.0)]
        for key, m, pts_l, kc in keys:
            field, extents[i] = _density_field(tex_chart, m, pts_l, kc, res)
            raw[(i, key)] = field
            vmax[m] = max(vmax.get(m, 0.0), np.nanmax(field))

    objs, views = [], []
    for i, (ch, metric, curv) in enumerate(panels):
        view_chart = "lorentz" if ch == "hyperboloid" else ch
        tex_chart = "poincare" if ch == "hyperboloid" else ch
        textures = {key: _field_to_uri(f, vmax[key.split("@")[0]]) for (j, key), f in raw.items() if j == i}
        default = f"hyperbolic@{curvatures[0]:g}" if curv else metric
        pts = None if ch == "hyperboloid" else np.round(CHARTS[tex_chart].from_lorentz(x, -1.0), 5).tolist()
        objs.append(DensityField(view_chart, extents[i], textures, default,
                                 surface=(ch == "hyperboloid"), points=pts, view=i, curvature=curv))
        views.append({"chart": view_chart, "title": f"{ch} · {metric}"})

    return Scene(objs, views=views, curvature=-1.0,
                 density_curvatures=[float(c) for c in curvatures],
                 legend=[("point", "#22d3ee", "prototypes — the kernel centers whose distances form the density")],
                 hint=("Top row: the same points' HYPERBOLIC density in each model (Poincaré / Klein / half-plane / "
                       "hyperboloid — isometric, so the density is identical; only the chart, hence the appearance, "
                       "differs). The four share ONE color scale (and it holds across the CURVATURE slider, K in "
                       "[-1, 0)), so their brightness is comparable and the flattening as K → 0 is real. Bottom: the "
                       "Euclidean and cosine kernels on the Poincaré disk — non-intrinsic metrics, each on its own "
                       "scale (they don't change with curvature). Scroll to zoom, drag to pan."))


def density_heatmaps(coords, k=-1.0, chart="poincare", grid=110, n_points=600, seed=0,
                     panels=(("poincare", "hyperbolic"), ("klein", "hyperbolic"), ("halfplane", "hyperbolic"),
                             ("hyperboloid", "hyperbolic"), ("poincare", "euclidean"), ("poincare", "cosine"))):
    """Compare the kernel density of the SAME points under different geometries. The
    four hyperbolic panels (Poincaré / Klein / half-plane / hyperboloid) render the
    identical intrinsic density — the models are isometric, so only their coordinate
    charts (hence the appearance) differ — against Euclidean and cosine kernels on the
    disk. >2D input is radius-reduced to 2D first."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    from . import reduce as _reduce
    x = _to_lorentz(coords, chart, k)
    if x.shape[-1] > 3:
        x, _ = _reduce.radial_pca(x, 2, k)
    rng = np.random.default_rng(seed)
    if len(x) > n_points:
        x = x[rng.choice(len(x), n_points, replace=False)]
    R = 1 / np.sqrt(-k)

    cols = min(len(panels), 3)
    rows = -(-len(panels) // cols)
    fig = Figure(figsize=(3.4 * cols, 3.6 * rows))
    for i, (ch, metric) in enumerate(panels):
        if ch == "hyperboloid":                                   # 3D surface colored by density
            C, valid, _ = _chart_grid("poincare", R, grid)
            C = C[valid][np.sum(C[valid] ** 2, 1) < (0.86 * R) ** 2]   # cap radius: height grows ~e^r
            gl = CHARTS["poincare"].to_lorentz(C, k)
            d = L.dist(gl[:, None], x[None], k)
            dens = np.exp(-(d**2) / (2 * _BW["hyperbolic"] ** 2)).sum(1)
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            ax.scatter(gl[:, 1], gl[:, 2], gl[:, 0] - R, c=dens / dens.max(), cmap="magma", s=16, edgecolors="none")
            ax.view_init(elev=22, azim=-60)
            ax.set_axis_off()
            ax.set_title("hyperboloid · hyperbolic", fontsize=9)
            continue
        field, extent = _density_field(ch, metric, x, k, grid)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(field / np.nanmax(field), origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="auto")
        if ch == "halfplane":
            ax.axhline(0, color="#898781", lw=1)                  # x-axis = ideal boundary
        else:
            ax.add_patch(Circle((0, 0), 1, fill=False, ec="#898781", lw=1))
        ax.set_axis_off()
        ax.set_title(f"{ch} · {metric}", fontsize=9)
    fig.tight_layout()
    return fig


def delta_hyperbolicity(coords, k=-1.0, chart="lorentz", n=180, n_quads=40000, size=(5.2, 3.2), seed=0):
    """Gromov 4-point δ over sampled quadruples of the embedding, normalized by the
    diameter (δ_rel = 2δ/diam): 0 = perfectly tree-like/hyperbolic, ~1 = far from it.
    Histogram of δ_rel with the worst-case marked."""
    x = _to_lorentz(coords, chart, k)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(n, len(x)), replace=False)
    D = L.dist(x[idx][:, None], x[idx][None], k)           # (n, n) distance matrix
    diam = D.max()
    q = rng.integers(0, len(idx), (n_quads, 4))
    a, b, c, dd = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s1 = D[a, b] + D[c, dd]
    s2 = D[a, c] + D[b, dd]
    s3 = D[a, dd] + D[b, c]
    top = np.sort(np.stack([s1, s2, s3], 1), 1)[:, ::-1]   # descending
    delta = (top[:, 0] - top[:, 1]) / 2 / (diam + 1e-12) * 2  # δ_rel = 2δ/diam

    fig = Figure(figsize=size)
    ax = _bare(fig.add_subplot())
    ax.hist(delta, bins=40, color="#1baf7a")
    ax.axvline(delta.max(), color="#e34948", lw=1.4, label=f"worst δ_rel = {delta.max():.2f}")
    ax.set_xlabel("relative 4-point δ  (0 = tree-like)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
