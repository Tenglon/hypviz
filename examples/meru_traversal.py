"""Real MERU (Desai et al. 2023) root traversal: embed an image OR arbitrary text with
the MERU model, walk the geodesic to the root, and retrieve the nearest concept from a
caption bank at each step — concrete at the boundary, abstract toward the center. Writes
a self-contained hypviz traversal HTML.

Setup (one-time):
    pip install git+https://github.com/facebookresearch/meru timm torchvision ftfy regex loguru
    git clone https://github.com/facebookresearch/meru        # for the package source + tokenizer vocab
    curl -sL https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth -o meru_vit_s.pth

Run:
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth --text "a golden retriever"
    python examples/meru_traversal.py --meru-repo ./meru --checkpoint meru_vit_s.pth --image photo.jpg
"""
import argparse
import sys

import numpy as np

# a concept bank spanning concrete → abstract; the traversal retrieves from these
BANK = [
    "a golden retriever", "a beagle", "a siamese cat", "a bald eagle", "a red rose", "a monarch butterfly",
    "a sports car", "a mountain bike", "an acoustic guitar", "a cup of espresso", "the Eiffel tower",
    "a dog", "a cat", "a bird", "a fish", "a flower", "an insect", "a car", "a musical instrument",
    "a building", "a tree", "a mammal", "a reptile", "a vehicle", "a plant", "an animal", "a machine",
    "food", "furniture", "a landscape", "a portrait", "a living thing", "an object", "a structure",
    "nature", "a scene", "a photograph", "a texture", "a pattern", "a color", "something abstract",
]

ap = argparse.ArgumentParser()
ap.add_argument("--meru-repo", required=True, help="path to a clone of facebookresearch/meru")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--text")
ap.add_argument("--image")
ap.add_argument("--arch", default="vit_small_mocov3_patch16_224")
ap.add_argument("--out", default="meru_traversal.html")
args = ap.parse_args()

sys.path.insert(0, args.meru_repo)
import torch
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import MERU
from meru.tokenizer import Tokenizer

from hypviz import traversal_scene


def load_model():
    model = MERU(
        visual=build_timm_vit(arch=args.arch, global_pool="token", use_sincos2d_pos=True),
        textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77),
        embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2).eval()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    return model


model = load_model()
tok = Tokenizer()
with torch.no_grad():
    bank = model.encode_text(tok(BANK), project=True).cpu().numpy()          # (N, 512) space components
    if args.image:
        from PIL import Image
        from torchvision import transforms as T
        tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
        img = tfm(Image.open(args.image).convert("RGB"))
        query = model.encode_image(img[None], project=True)[0].cpu().numpy()
        title = f"MERU traversal — {args.image}"
    else:
        query = model.encode_text(tok([args.text]), project=True)[0].cpu().numpy()
        title = f"MERU traversal — “{args.text}”"

k = -float(model.curv.exp())                                                 # MERU curvature c → our k = -c
from hypviz.kernel import lorentz as L
q_l = L.from_spatial(query, k)                                               # reconstruct the Lorentz point
bank_l = L.from_spatial(bank, k)
traversal_scene(q_l, bank_l, BANK, k=k, chart="lorentz").to_html(args.out, title=title)
print(f"wrote {args.out}")
