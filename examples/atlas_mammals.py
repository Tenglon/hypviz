"""A real (small) mammal taxonomy, embedded into H² by Sarkar's construction and
rendered as an atlas — the deterministic, no-download gallery/CI demo. Any larger
taxonomy (GBIF / BioCLIP / TreeOfLife rank export) drops into the same pipeline
via `atlas_taxonomy_csv.py`."""
from hypviz import Tree, atlas, embed

# root-to-leaf Linnaean paths (Order → Family → Genus → species)
PATHS = [
    ("Carnivora", "Felidae", "Panthera", "P. leo"), ("Carnivora", "Felidae", "Panthera", "P. tigris"),
    ("Carnivora", "Felidae", "Felis", "F. catus"), ("Carnivora", "Felidae", "Acinonyx", "A. jubatus"),
    ("Carnivora", "Canidae", "Canis", "C. lupus"), ("Carnivora", "Canidae", "Canis", "C. familiaris"),
    ("Carnivora", "Canidae", "Vulpes", "V. vulpes"), ("Carnivora", "Ursidae", "Ursus", "U. arctos"),
    ("Carnivora", "Ursidae", "Ursus", "U. maritimus"), ("Carnivora", "Ursidae", "Ailuropoda", "A. melanoleuca"),
    ("Primates", "Hominidae", "Homo", "H. sapiens"), ("Primates", "Hominidae", "Pan", "P. troglodytes"),
    ("Primates", "Hominidae", "Gorilla", "G. gorilla"), ("Primates", "Hominidae", "Pongo", "P. pygmaeus"),
    ("Primates", "Cercopithecidae", "Macaca", "M. mulatta"), ("Primates", "Lemuridae", "Lemur", "L. catta"),
    ("Cetacea", "Balaenopteridae", "Balaenoptera", "B. musculus"),
    ("Cetacea", "Delphinidae", "Tursiops", "T. truncatus"), ("Cetacea", "Delphinidae", "Orcinus", "O. orca"),
    ("Cetacea", "Physeteridae", "Physeter", "P. macrocephalus"),
    ("Rodentia", "Muridae", "Mus", "M. musculus"), ("Rodentia", "Muridae", "Rattus", "R. norvegicus"),
    ("Rodentia", "Sciuridae", "Sciurus", "S. carolinensis"), ("Rodentia", "Castoridae", "Castor", "C. canadensis"),
    ("Chiroptera", "Pteropodidae", "Pteropus", "P. vampyrus"),
    ("Chiroptera", "Vespertilionidae", "Myotis", "M. lucifugus"),
    ("Proboscidea", "Elephantidae", "Loxodonta", "L. africana"),
    ("Proboscidea", "Elephantidae", "Elephas", "E. maximus"),
]


def build_scene():
    tree = Tree.from_paths(PATHS, root_name="Mammalia")
    coords = embed.sarkar(tree, tau=1.15)                  # tree → Poincaré disk (2D, no reduction)
    return atlas(coords, tree, labels=tree.labels, chart="poincare", color_by="depth")


if __name__ == "__main__":
    scene = build_scene()
    scene.to_html("examples/out/atlas_mammals.html", title="Mammal taxonomy — Sarkar embedding")
    print("wrote examples/out/atlas_mammals.html")
