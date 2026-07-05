"""The v1 gallery: prebuilt teaching scenes, each assembled from the public
primitives — they double as API examples."""
from .scene import DistanceLabel, Geodesic, LogVector, MobiusSum, Point, Scene, TangentPlane

# paper palette (dataviz categorical slots)
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
MUTED = "#898781"


def geodesic():
    """Scene 1 — geodesics & distance."""
    a = Point([0.45, 0.10], draggable=True, label="a", color=BLUE)
    b = Point([-0.30, 0.45], draggable=True, label="b", color=AQUA)
    return Scene([a, b, Geodesic(a, b), DistanceLabel(a, b)], curvature_slider=True)


def models():
    """Scene 2 — one triangle, four models: the same geometry in every chart.
    Klein draws geodesics as straight chords; Poincare/half-plane as arcs."""
    a = Point([0.42, 0.08], draggable=True, label="a", color=BLUE)
    b = Point([-0.25, 0.42], draggable=True, label="b", color=AQUA)
    c = Point([-0.18, -0.38], draggable=True, label="c", color=YELLOW)
    return Scene(
        [a, b, c, Geodesic(a, b), Geodesic(b, c), Geodesic(c, a)],
        views=("poincare", "klein", "halfplane", "lorentz"),
        curvature_slider=True,
        hint=("The SAME triangle rendered in four models of H² — drag any vertex in any view. "
              "Klein draws geodesics as straight chords (but distorts angles); Poincaré keeps angles "
              "true (but bends geodesics into arcs); the hyperboloid is the model everything else projects from."),
    )


def exp_log():
    """Scene 3 — exp / log maps: the straight arrow is v = log_x(y) in the
    tangent space; the curve is t -> exp_x(t v)."""
    x = Point([0.12, -0.08], draggable=True, label="x", color=BLUE)
    y = Point([0.52, 0.35], draggable=True, label="y", color=AQUA)
    return Scene(
        [x, y, Geodesic(x, y), LogVector(x, y, color=VIOLET), TangentPlane(x), DistanceLabel(x, y)],
        curvature_slider=True,
        hint=("The violet arrow is the tangent vector v = log_x(y): the 'straight-line instruction' that "
              "exp_x turns into a geodesic. Drag x or y — the arrow lives in the tangent plane at x "
              "(shown in 3D), and its length equals the geodesic distance d(x, y)."),
    )


def mobius_add():
    """Scene 4 — Mobius addition is NOT commutative: a(+)b lands away from b(+)a."""
    o = Point([0.0, 0.0], label="o", color=MUTED)
    a = Point([0.35, 0.05], draggable=True, label="a", color=BLUE)
    b = Point([-0.10, 0.40], draggable=True, label="b", color=AQUA)
    ab = MobiusSum(a, b, label="a⊕b", color=YELLOW)
    ba = MobiusSum(b, a, label="b⊕a", color=VIOLET)
    return Scene(
        [o, a, b, ab, ba,
         Geodesic(o, a, color=BLUE), Geodesic(o, b, color=AQUA),
         Geodesic(a, ab, color=YELLOW), Geodesic(b, ba, color=VIOLET),
         DistanceLabel(ab, ba)],
        curvature_slider=True,
        hint=("Möbius addition a⊕b: translate b by a (yellow), against b⊕a (violet) — in hyperbolic "
              "space they do NOT coincide; the label shows their distance, the failure of commutativity. "
              "Drag a and b; watch the gap close as a, b approach the origin or K approaches 0."),
    )


GALLERY = {"geodesic": geodesic, "models": models, "exp_log": exp_log, "mobius_add": mobius_add}
