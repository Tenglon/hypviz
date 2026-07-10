"""Real MERU (Desai et al. 2023) root traversal: embed an image OR arbitrary text with
the MERU model, walk the geodesic to the root, and retrieve the nearest concept from a
caption bank at each step — concrete at the boundary, abstract toward the center. Pass
several --text/--image and the page gets a picker to switch between the walks (MERU can't
run in the browser, so the traversals are precomputed here). For a live upload interface
see meru_app.py.

Setup (one-time):
    pip install timm torchvision ftfy regex loguru omegaconf
    git clone https://github.com/facebookresearch/meru        # package source + tokenizer vocab
    curl -sL https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth -o meru_vit_s.pth

Run:
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth --text "a golden retriever"
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth \\
        --text "a bald eagle" --text "a red sports car" --image photo.jpg     # → picker
"""
import argparse

from _meru import BANK, load_meru

from hypviz import traversal_gallery, traversal_scene

DEMO = ["a golden retriever", "a red sports car", "a bald eagle", "an acoustic guitar", "the Eiffel tower"]

ap = argparse.ArgumentParser()
ap.add_argument("--meru-repo", required=True, help="path to a clone of facebookresearch/meru")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--text", action="append", help="repeatable; a text query")
ap.add_argument("--image", action="append", help="repeatable; an image path")
ap.add_argument("--arch", default="vit_small_mocov3_patch16_224")
ap.add_argument("--out", default="meru_traversal.html")
args = ap.parse_args()

encode_text, encode_image, k = load_meru(args.meru_repo, args.checkpoint, args.arch)
bank_l = encode_text(BANK)

texts = args.text or ([] if args.image else DEMO)             # default to the demo set if nothing given
items = list(zip([f"“{t}”" for t in texts], encode_text(texts))) if texts else []
if args.image:
    from PIL import Image
    items += [(img.split("/")[-1], encode_image(Image.open(img))) for img in args.image]

if len(items) == 1:
    scene = traversal_scene(items[0][1], bank_l, BANK, k=k, chart="lorentz")
    title = f"MERU traversal — {items[0][0]}"
else:
    scene = traversal_gallery(items, bank_l, BANK, k=k, chart="lorentz")
    title = "MERU root traversal"

scene.to_html(args.out, title=title)
print(f"wrote {args.out}  ({len(items)} traversal{'s' if len(items) != 1 else ''})")
