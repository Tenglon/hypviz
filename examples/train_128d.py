"""Produce a real 128-D hyperbolic embedding artifact: train a Poincaré embedding
of a WordNet hypernym graph and save it as a reusable .npz (coords + parent +
labels) — a downloadable-equivalent, since the field ships no such files. Load it
later with examples/atlas_from_file.py, or drop in your own .npz the same way."""
import sys
from pathlib import Path

import numpy as np
from gensim.models.poincare import PoincareModel
from nltk.corpus import wordnet as wn

from hypviz import Tree, atlas, stats

ROOT = sys.argv[1] if len(sys.argv) > 1 else "mammal.n.01"
DIM, EPOCHS = 128, 100
OUT = Path.home() / "Documents" / "hypviz" / "data"
OUT.mkdir(exist_ok=True)


def build_tree(root_name):
    root = wn.synset(root_name)
    idx, parent, key, label, fr = {root: 0}, [-1], [root.name()], [root.lemmas()[0].name()], [root]
    while fr:
        nxt = []
        for s in fr:
            for c in s.hyponyms():
                if c not in idx:
                    idx[c] = len(parent); parent.append(idx[s]); key.append(c.name())
                    label.append(c.lemmas()[0].name()); nxt.append(c)
        fr = nxt
    return Tree(parent, labels=np.array(label, dtype=object)), key


tree, key = build_tree(ROOT)
rels = [(key[i], key[int(tree.parent[i])]) for i in range(len(tree)) if tree.parent[i] >= 0]
print(f"training {DIM}-D Poincaré on {len(tree)} synsets / {len(rels)} edges ...")
model = PoincareModel(rels, size=DIM, negative=10, seed=0)
model.train(epochs=EPOCHS, print_every=100000)

coords = np.stack([model.kv[k] for k in key])
r = np.linalg.norm(coords, axis=1, keepdims=True)
coords = np.where(r > 0.999, coords * 0.999 / r, coords)     # keep off the boundary

stem = ROOT.split(".")[0]
npz = OUT / f"wordnet_{stem}_poincare_{DIM}d.npz"
np.savez(npz, coords=coords, parent=tree.parent, labels=tree.labels, chart="poincare")
print(f"saved {npz}  ({coords.shape[0]}×{coords.shape[1]}, max radius {np.linalg.norm(coords,axis=1).max():.4f})")

atlas(coords, tree, labels=tree.labels, chart="poincare", color_by="depth", budget=8000) \
    .to_html("examples/out/atlas_128d.html", title=f"Trained 128-D Poincaré — WordNet '{ROOT}'")
stats.depth_norm(coords, tree.depth(), chart="poincare").savefig(
    "examples/out/atlas_128d_depth_norm.svg", bbox_inches="tight")
print("wrote examples/out/atlas_128d.html")
