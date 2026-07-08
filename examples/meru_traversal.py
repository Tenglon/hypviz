"""Real MERU (Desai et al. 2023) root traversal: embed an image OR arbitrary text with
the MERU model, walk the geodesic to the root, and retrieve the nearest concept from a
caption bank at each step — concrete at the boundary, abstract toward the center. Writes
a self-contained hypviz traversal HTML. For an interactive upload interface, see meru_app.py.

Setup (one-time):
    pip install timm torchvision ftfy regex loguru omegaconf
    git clone https://github.com/facebookresearch/meru        # package source + tokenizer vocab
    curl -sL https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth -o meru_vit_s.pth

Run:
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth --text "a golden retriever"
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth --image photo.jpg
"""
import argparse

from _meru import BANK, load_meru

from hypviz import traversal_scene

ap = argparse.ArgumentParser()
ap.add_argument("--meru-repo", required=True, help="path to a clone of facebookresearch/meru")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--text")
ap.add_argument("--image")
ap.add_argument("--arch", default="vit_small_mocov3_patch16_224")
ap.add_argument("--out", default="meru_traversal.html")
args = ap.parse_args()

encode_text, encode_image, k = load_meru(args.meru_repo, args.checkpoint, args.arch)
bank_l = encode_text(BANK)
if args.image:
    from PIL import Image
    query, title = encode_image(Image.open(args.image)), f"MERU traversal — {args.image}"
else:
    query, title = encode_text([args.text])[0], f"MERU traversal — “{args.text}”"

traversal_scene(query, bank_l, BANK, k=k, chart="lorentz").to_html(args.out, title=title)
print(f"wrote {args.out}")
