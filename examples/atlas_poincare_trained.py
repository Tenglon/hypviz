"""Real, *trained* hyperbolic embedding: a Poincaré embedding of the WordNet
'mammal' hypernym graph, trained from scratch with gensim (the Nickel & Kiela
method), then reduced + explored as an atlas. Unlike Sarkar (a deterministic 2D
construction), this is a gradient-trained n-D embedding — so the atlas runs its
full high-dim → radius-preserving 2D reduction on genuine trained vectors.

Requires: nltk (+ 'wordnet' corpus) and gensim.
"""
import numpy as np
from gensim.models.poincare import PoincareModel
from nltk.corpus import wordnet as wn

from hypviz import Tree, atlas, stats

ROOT, DIM, EPOCHS = "mammal.n.01", 10, 200


def build_tree(root_name):
    """Single-parent tree (WordNet is a DAG); keys = synset names, labels = lemmas."""
    root = wn.synset(root_name)
    idx, parent, key, label, frontier = {root: 0}, [-1], [root.name()], [root.lemmas()[0].name()], [root]
    while frontier:
        nxt = []
        for s in frontier:
            for c in s.hyponyms():
                if c not in idx:
                    idx[c] = len(parent)
                    parent.append(idx[s])
                    key.append(c.name())
                    label.append(c.lemmas()[0].name())
                    nxt.append(c)
        frontier = nxt
    return Tree(parent, labels=np.array(label, dtype=object)), key


tree, key = build_tree(ROOT)
relations = [(key[i], key[int(tree.parent[i])]) for i in range(len(tree)) if tree.parent[i] >= 0]

print(f"training {DIM}-D Poincaré embedding on {len(tree)} synsets, {len(relations)} hypernym edges...")
model = PoincareModel(relations, size=DIM, negative=10, seed=0)
model.train(epochs=EPOCHS, print_every=500)

# align trained vectors to tree-node order; clip off the boundary for numerical safety
coords = np.stack([model.kv[k] for k in key])
r = np.linalg.norm(coords, axis=1, keepdims=True)
coords = np.where(r > 0.999, coords * 0.999 / r, coords)
print(f"trained. max Poincaré radius {np.linalg.norm(coords, axis=1).max():.4f}  (dim {coords.shape[1]})")

atlas(coords, tree, labels=tree.labels, chart="poincare", color_by="depth", budget=1000) \
    .to_html("examples/out/atlas_poincare_trained.html",
             title=f"Trained {DIM}-D Poincaré embedding — WordNet '{ROOT}'")
stats.depth_norm(coords, tree.depth(), chart="poincare").savefig(
    "examples/out/poincare_trained_depth_norm.svg", bbox_inches="tight")
print("wrote examples/out/atlas_poincare_trained.html + poincare_trained_depth_norm.svg")
