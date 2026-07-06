"""Build every gallery scene into examples/out/."""
from hypviz.scenes import GALLERY

TITLES = {
    "geodesic": "Geodesic & distance in H²",
    "models": "Four models, one geometry",
    "exp_log": "exp & log maps: the tangent space",
    "mobius_add": "Möbius addition is not commutative",
    "parallel_transport": "Parallel transport & holonomy",
    "gyroplane": "Gyroplanes & gyrovector space",
}

for name, build in GALLERY.items():
    path = build().to_html(f"examples/out/{name}.html", title=TITLES[name])
    print(f"wrote {path}")
