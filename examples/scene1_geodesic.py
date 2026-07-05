"""Scene 1 — geodesics & distance: drag a and b in either model; slide the curvature."""
from hypviz import DistanceLabel, Geodesic, Point, Scene

a = Point([0.45, 0.10], draggable=True, label="a", color="#2a78d6")
b = Point([-0.30, 0.45], draggable=True, label="b", color="#1baf7a")

Scene([a, b, Geodesic(a, b), DistanceLabel(a, b)], curvature_slider=True).to_html(
    "examples/out/scene1_geodesic.html", title="Geodesic & distance in H²")
print("wrote examples/out/scene1_geodesic.html")
