"""Scene 1 — geodesics & distance: drag a and b, in either model."""
from hypviz import DistanceLabel, Geodesic, Point, Scene

a = Point([0.45, 0.10], draggable=True, label="a", color="#5b8def")
b = Point([-0.30, 0.45], draggable=True, label="b", color="#ef7d5b")

Scene([a, b, Geodesic(a, b), DistanceLabel(a, b)]).to_html(
    "examples/out/scene1_geodesic.html", title="Geodesic & distance in H²")
print("wrote examples/out/scene1_geodesic.html")
