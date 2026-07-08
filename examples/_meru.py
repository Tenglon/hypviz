"""Shared MERU (Desai et al. 2023) loader for the traversal example + upload app.
Loads the model from a checkpoint and returns encoders that hand back Lorentz points
in hypviz's convention (k = -c, so ⟨x,x⟩ = -1/c). See meru_traversal.py for setup."""
import sys

# a concept bank spanning concrete → abstract; the traversal retrieves from these
BANK = [
    "a golden retriever", "a beagle", "a siamese cat", "a bald eagle", "a red rose", "a monarch butterfly",
    "a sports car", "a mountain bike", "an acoustic guitar", "a cup of espresso", "the Eiffel tower",
    "a dog", "a cat", "a bird", "a fish", "a flower", "an insect", "a car", "a musical instrument",
    "a building", "a tree", "a mammal", "a reptile", "a vehicle", "a plant", "an animal", "a machine",
    "food", "furniture", "a landscape", "a portrait", "a living thing", "an object", "a structure",
    "nature", "a scene", "a photograph", "a texture", "a pattern", "a color", "something abstract",
]


def load_meru(meru_repo, checkpoint, arch="vit_small_mocov3_patch16_224"):
    """Return (encode_text, encode_image, k). `meru_repo` is a clone of facebookresearch/meru
    (namespace package, no __init__.py, so added to sys.path)."""
    sys.path.insert(0, meru_repo)
    import torch
    from meru.encoders.image_encoders import build_timm_vit
    from meru.encoders.text_encoders import TransformerTextEncoder
    from meru.models import MERU
    from meru.tokenizer import Tokenizer

    from hypviz.kernel import lorentz as L

    model = MERU(
        visual=build_timm_vit(arch=arch, global_pool="token", use_sincos2d_pos=True),
        textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77),
        embed_dim=512, curv_init=1.0, learn_curv=True, entail_weight=0.2).eval()
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    tok = Tokenizer()
    k = -float(model.curv.exp())                                     # MERU curvature c → our k = -c

    def encode_text(texts):                                          # (N, 512) space feats → Lorentz points
        with torch.no_grad():
            return L.from_spatial(model.encode_text(tok(list(texts)), project=True).cpu().numpy(), k)

    def encode_image(pil_img):
        from torchvision import transforms as T
        x = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])(pil_img.convert("RGB"))
        with torch.no_grad():
            return L.from_spatial(model.encode_image(x[None], project=True)[0].cpu().numpy(), k)

    return encode_text, encode_image, k
