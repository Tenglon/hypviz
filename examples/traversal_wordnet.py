"""MERU-style root traversal on real data: query a WordNet synset (or any noun in the
embedded 'animal' subtree) and walk the geodesic toward the root, retrieving the nearest
concept at each step — a concrete species at the boundary, broad taxa toward the center.
Uses the trained 128-D Poincaré embedding; the concept bank is the embedded synsets.

    python examples/traversal_wordnet.py [word]      # e.g. beagle, whale, sparrow
"""
import sys

import numpy as np

from hypviz import Tree, traversal_scene

d = np.load("data/wordnet_animal_poincare_128d.npz", allow_pickle=True)
coords, labels = d["coords"], [str(x) for x in d["labels"]]
tree = Tree(d["parent"], labels=d["labels"])

word = sys.argv[1] if len(sys.argv) > 1 else "beagle"
matches = [i for i, name in enumerate(labels) if name.lower() == word.lower()]
if not matches:                                              # fall back to the deepest node containing the word
    cand = [i for i, name in enumerate(labels) if word.lower() in name.lower()]
    matches = [max(cand, key=lambda i: tree.depth()[i])] if cand else [int(np.argmax(tree.depth()))]
q = matches[0]
print(f"query: {labels[q]}  (depth {tree.depth()[q]})")

traversal_scene(coords[q], coords, labels, chart="poincare") \
    .to_html("examples/out/traversal_wordnet.html", title=f"Root traversal — '{labels[q]}'")
print("wrote examples/out/traversal_wordnet.html")
