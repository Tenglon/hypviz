"""Real data: the WordNet 'animal' noun hierarchy (Nickel & Kiela's domain),
embedded into H² by Sarkar's construction and explored as an atlas. Requires
`nltk` with the 'wordnet' corpus (`nltk.download('wordnet')`)."""
import numpy as np
from nltk.corpus import wordnet as wn

from hypviz import Tree, atlas, embed, stats

ROOT = "animal.n.01"


def build_tree(root_name):
    """WordNet is a DAG; take the first-seen hypernym as each synset's parent (BFS)."""
    root = wn.synset(root_name)
    idx, parent, name, frontier = {root: 0}, [-1], [root.lemmas()[0].name()], [root]
    while frontier:
        nxt = []
        for s in frontier:
            for c in s.hyponyms():
                if c not in idx:
                    idx[c] = len(parent)
                    parent.append(idx[s])
                    name.append(c.lemmas()[0].name())
                    nxt.append(c)
        frontier = nxt
    return Tree(parent, labels=name)


tree = build_tree(ROOT)
coords = embed.sarkar(tree, tau=0.6)                       # tree → Poincaré disk
print(f"{len(tree)} synsets, max depth {tree.depth().max()}, "
      f"max Poincaré radius {np.linalg.norm(coords, axis=1).max():.6f}")

atlas(coords, tree, labels=tree.labels, chart="poincare", color_by="depth", budget=1500) \
    .to_html("examples/out/atlas_wordnet.html", title=f"WordNet '{ROOT}' — {len(tree)} synsets")
stats.depth_norm(coords, tree.depth(), chart="poincare").savefig("examples/out/wordnet_depth_norm.svg", bbox_inches="tight")
print("wrote examples/out/atlas_wordnet.html + wordnet_depth_norm.svg")
