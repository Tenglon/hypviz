"""Turn any taxonomy rank-table CSV into an atlas — the format GBIF, iNaturalist
and the BioCLIP/TreeOfLife catalog all export (one row per leaf; columns are the
Linnaean ranks). Downloading such a table is left to the user (licenses/size
vary); pointing this script at it is the whole 'real data' path.

    python examples/atlas_taxonomy_csv.py taxa.csv [rank1 rank2 ...]
"""
import csv
import sys

from hypviz import Tree, atlas, embed

_RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


def build_scene(rows, ranks=None):
    ranks = ranks or [c for c in rows[0] if c.lower() in _RANKS]
    paths = [tuple(r[c] for c in ranks if r.get(c)) for r in rows]
    tree = Tree.from_paths(paths, root_name="life")
    coords = embed.sarkar(tree, tau=1.1)
    return atlas(coords, tree, labels=tree.labels, chart="poincare", color_by="depth"), tree


if __name__ == "__main__":
    with open(sys.argv[1], newline="") as f:
        rows = list(csv.DictReader(f))
    scene, tree = build_scene(rows, sys.argv[2:] or None)
    scene.to_html("taxonomy_atlas.html", title=f"{len(tree)} taxa")
    print(f"wrote taxonomy_atlas.html ({len(tree)} nodes from {len(rows)} rows)")
