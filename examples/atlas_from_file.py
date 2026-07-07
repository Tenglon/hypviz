"""Drop your own hyperbolic embedding into the atlas.

    python examples/atlas_from_file.py embedding.npz [--chart poincare] [--budget 10000]

Accepts an .npz bundle with arrays: coords (N, d) or (N, d+1), parent (N,) tree
(−1 for root), and optional labels (N,). `chart` says which model `coords` live
in ('poincare' | 'klein' | 'halfplane' | 'lorentz'); >2D is auto-reduced to 2D.
"""
import argparse

import numpy as np

from hypviz import Tree, atlas

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--chart", default="poincare")
ap.add_argument("--color", default="depth", choices=["depth", "norm", "label"])
ap.add_argument("--reduction", default="radial", choices=["radial", "tree", "horo", "tangent"])
ap.add_argument("--centroids", default=None, choices=["depth", "clade"], help="overlay per-group centroids")
ap.add_argument("--dim", type=int, default=2, choices=[2, 3], help="3 = H³ Poincaré-ball view")
ap.add_argument("--budget", type=int, default=10_000)
ap.add_argument("--out", default="my_atlas.html")
ap.add_argument("--title", default=None)
args = ap.parse_args()

d = np.load(args.path, allow_pickle=True)
labels = d["labels"] if "labels" in d else None
tree = Tree(d["parent"], labels=labels)
atlas(d["coords"], tree, labels=labels, chart=args.chart, color_by=args.color,
      reduction=args.reduction, dim=args.dim, budget=args.budget, show_centroids=args.centroids) \
    .to_html(args.out, title=args.title or args.path)
print(f"wrote {args.out}  ({len(tree)} nodes, {d['coords'].shape[1]}-D {args.chart})")
