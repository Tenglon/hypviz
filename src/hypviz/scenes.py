"""The v1 gallery: prebuilt teaching scenes, each assembled from the public
primitives — they double as API examples."""
from .scene import (DistanceLabel, Geodesic, LogVector, MetricCircle, MobiusSum, Point, Scene,
                    TangentPlane, TransportLoop)

# paper palette (dataviz categorical slots)
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
MUTED, INK = "#898781", "#52514e"


def geodesic():
    """Scene 1 — geodesics & distance."""
    a = Point([0.45, 0.10], draggable=True, label="a", color=BLUE)
    b = Point([-0.30, 0.45], draggable=True, label="b", color=AQUA)
    return Scene([a, b, Geodesic(a, b), DistanceLabel(a, b)], curvature_slider=True,
                 legend=[("point", BLUE, "a, b — draggable points"),
                         ("line", INK, "geodesic a ↔ b (shortest path)")])


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
        legend=[("point", BLUE, "a, b, c — draggable vertices"),
                ("line", INK, "geodesic triangle edges — one triangle, four charts")],
        hint=("The SAME triangle rendered in four models of H² — drag any vertex in any view. "
              "Klein draws geodesics as straight chords (but distorts angles); Poincaré keeps angles "
              "true (but bends geodesics into arcs); the hyperboloid is the model everything else projects from. "
              "In the half-plane the x-axis is the IDEAL BOUNDARY (points at infinity, like the disk's rim) — "
              "the hyperbolic origin maps to (0, R), the classical base point i."),
    )


def exp_log():
    """Scene 3 — exp / log maps: the straight arrow is v = log_x(y) in the
    tangent space; the curve is t -> exp_x(t v). The hemisphere is the Poincaré
    disk lifted into 3D (stereographic projection), where tangent planes tilt."""
    x = Point([0.12, -0.08], draggable=True, label="x", color=BLUE)
    y = Point([0.52, 0.35], draggable=True, label="y", color=AQUA)
    return Scene(
        [x, y, Geodesic(x, y), LogVector(x, y, color=VIOLET), TangentPlane(x),
         MetricCircle(x, radius=0.35, color=VIOLET), DistanceLabel(x, y)],
        views=("poincare", "hemisphere", "lorentz"),
        curvature_slider=True,
        legend=[("arrow", VIOLET, "v = log_x(y) — tangent vector at x"),
                ("circle", VIOLET, "metric circle — fixed ruler in T_x"),
                ("area", VIOLET, "tangent plane at x (3D views)"),
                ("line", INK, "geodesic exp_x(t·v)")],
        hint=("The violet arrow is the tangent vector v = log_x(y): the 'straight-line instruction' that "
              "exp_x turns into a geodesic; its length equals d(x, y). The hemisphere is the Poincaré disk "
              "lifted into 3D — there (and on the hyperboloid) the tangent plane at x is a real tilted plane. "
              "On the flat disk the tangent plane IS the screen; the violet metric circle (a fixed hyperbolic "
              "ruler in T_x) shrinks as x approaches the rim — that is the conformal factor."),
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
        legend=[("line", BLUE, "o → a"), ("line", AQUA, "o → b"),
                ("line", YELLOW, "a → a⊕b  (b translated by a)"),
                ("line", VIOLET, "b → b⊕a  (a translated by b)")],
        hint=("Möbius addition a⊕b: translate b by a (yellow), against b⊕a (violet) — in hyperbolic "
              "space they do NOT coincide; the label shows their distance, the failure of commutativity. "
              "Drag a and b; watch the gap close as a, b approach the origin or K approaches 0."),
    )


def parallel_transport():
    """Scene 5 — parallel transport & holonomy: transport a vector around a
    geodesic triangle; it returns rotated by the angle deficit = the area."""
    x = Point([0.0, 0.42], draggable=True, label="x", color=MUTED)
    y = Point([-0.42, -0.28], draggable=True, label="y", color=MUTED)
    z = Point([0.46, -0.28], draggable=True, label="z", color=MUTED)
    return Scene(
        [x, y, z, Geodesic(x, y), Geodesic(y, z), Geodesic(z, x), TransportLoop([x, y, z])],
        curvature_slider=True,
        legend=[("arrow", BLUE, "v — initial vector at x"),
                ("arrow", AQUA, "v parallel-transported along each edge"),
                ("arrow", "#e34948", "v after the full loop — rotated by the holonomy")],
        hint=("Parallel-transport the vector around the geodesic triangle x → y → z → x. In hyperbolic "
              "space it returns ROTATED: the holonomy angle equals the triangle's area = π − (α+β+γ), "
              "the angle deficit. Drag the vertices to change the area; slide the curvature — as K → 0 the "
              "rotation vanishes (Euclidean parallel transport), as K grows more negative it grows."),
    )


GALLERY = {"geodesic": geodesic, "models": models, "exp_log": exp_log,
           "mobius_add": mobius_add, "parallel_transport": parallel_transport}
