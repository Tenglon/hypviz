"""Build the GitHub Pages site: docs/index.html + docs/gallery/*.html."""
import importlib.util
from pathlib import Path

from hypviz.scenes import GALLERY

ROOT = Path(__file__).resolve().parent.parent
SCENES = {
    "geodesic": ("Geodesic & distance in H²",
                 "Drag two points; the geodesic and its length react live in the Poincaré disk and on the hyperboloid."),
    "models": ("Four models, one geometry",
               "The same triangle in Poincaré, Klein, half-plane and Lorentz coordinates — straight chords vs. arcs."),
    "exp_log": ("exp & log maps: the tangent space",
                "The straight arrow v = log_x(y) and the geodesic exp_x(tv) it unrolls into."),
    "mobius_add": ("Möbius addition is not commutative",
                   "a⊕b lands away from b⊕a; watch the gap close near the origin or as K → 0."),
    "parallel_transport": ("Parallel transport & holonomy",
                           "Transport a vector around a geodesic triangle; it returns rotated by the area — vanishing as K → 0."),
    "gyroplane": ("Gyroplanes & gyrovector space",
                  "The hyperbolic MLR decision boundary and its confidence contours; drag a test point to read its logit."),
    "entailment": ("Entailment cones",
                   "Each point's cone of descendants; the aperture narrows toward the boundary. Drag to test the is-a relation."),
    "atlas_mammals": ("Embedding Atlas — a mammal taxonomy",
                      "75 taxa embedded by Sarkar's construction; hover for species, click to trace an ancestor chain."),
    "atlas_synth": ("Embedding Atlas — 128D, sampled",
                    "A 30k-node synthetic embedding: depth-encoding radius + tree-layout angle, sampled to 4k with honest pruned counts."),
}

# pre-built static pages (need external data/deps to regenerate, so committed as artifacts)
STATIC = {
    "atlas_128d": ("Embedding Atlas — trained 128D (real)",
                   "A real 128-D Poincaré embedding of WordNet 'animal' (3999 synsets), trained with gensim, radius-reduced to 2D."),
    "atlas_centroids": ("Hyperbolic clade centroids (real 128D)",
                        "The same 128-D embedding with each top clade's hyperbolic centroid (Fréchet mean) overlaid as a labeled marker."),
    "atlas_ball3d": ("H³ Poincaré ball — 3D (real 128D)",
                     "The real 128-D embedding reduced to 3D and shown in the rotatable Poincaré ball, instead of the 2D disk."),
}


def _example_scene(mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "examples" / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_scene()

CARD = """    <a class="card" href="gallery/{name}.html">
      <h2>{title}</h2>
      <p>{blurb}</p>
    </a>"""

INDEX = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>hypviz — visual hyperbolic machine learning</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         background: #f9f9f7; color: #0b0b0b; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 56px 28px 48px; }}
  h1 {{ font-size: 30px; font-weight: 700; letter-spacing: -0.015em; margin: 0; }}
  p.tag {{ font-size: 15px; line-height: 1.6; color: #52514e; max-width: 64ch; margin: 12px 0 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 32px; }}
  .card {{ display: block; padding: 22px 24px; text-decoration: none; color: inherit;
          border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; background: #fcfcfb;
          box-shadow: 0 1px 3px rgba(11,11,11,0.05); transition: border-color .15s; }}
  .card:hover {{ border-color: #2a78d6; }}
  .card h2 {{ font-size: 16.5px; font-weight: 650; margin: 0; }}
  .card p {{ font-size: 13.5px; line-height: 1.55; color: #52514e; margin: 8px 0 0; }}
  footer {{ margin-top: 40px; font-size: 13px; color: #898781; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>hypviz</h1>
  <p class="tag">Interactive visualizations of hyperbolic machine learning — drag points on the Poincaré disk
  and watch the Lorentz hyperboloid respond in real time, or drop a real high-dimensional embedding into the
  Embedding Atlas (radius-preserving reduction, hierarchy-aware sampling, hover + ancestor-chain interaction).
  Every scene is a self-contained page: save it, email it, embed it.</p>
  <div class="grid">
{cards}
  </div>
  <footer>hypviz · MIT · Python + three.js</footer>
</div>
</body>
</html>
"""

out = ROOT / "docs" / "gallery"
for name in SCENES:
    scene = GALLERY[name]() if name in GALLERY else _example_scene(name)
    scene.to_html(out / f"{name}.html", title=SCENES[name][0])
    print(f"wrote docs/gallery/{name}.html")
for name in STATIC:                                   # pre-committed; regenerate via examples/train_128d.py
    print(f"kept static docs/gallery/{name}.html" if (out / f"{name}.html").exists() else f"MISSING {name}.html")

cards = "\n".join(CARD.format(name=n, title=t, blurb=b) for n, (t, b) in {**SCENES, **STATIC}.items())
(ROOT / "docs" / "index.html").write_text(INDEX.format(cards=cards))
print("wrote docs/index.html")
