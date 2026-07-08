"""Upload interface for the MERU root traversal: drop in an image OR type any text, run
the real MERU model, and get the query→root concept traversal — concrete at the boundary,
abstract at the center. MERU can't run inside a self-contained page (strict CSP, no model
in the browser), so this is a small local Flask server; each result IS a self-contained
hypviz page you can save and share.

    pip install flask        # plus the MERU setup from meru_traversal.py
    python examples/meru_app.py --meru-repo ./meru --checkpoint meru_vit_s.pth
    #  → open http://127.0.0.1:7860
"""
import argparse

from _meru import BANK, load_meru
from flask import Flask, Response, request

from hypviz import traversal_scene

ap = argparse.ArgumentParser()
ap.add_argument("--meru-repo", required=True, help="path to a clone of facebookresearch/meru")
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--arch", default="vit_small_mocov3_patch16_224")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=7860)
args = ap.parse_args()

encode_text, encode_image, k = load_meru(args.meru_repo, args.checkpoint, args.arch)
bank_l = encode_text(BANK)                                            # encode the bank once, up front

FORM = """<!doctype html><meta charset=utf-8><title>MERU root traversal</title>
<style>body{font:15px/1.6 system-ui,sans-serif;background:#f9f9f7;color:#0b0b0b;max-width:620px;margin:9vh auto;padding:0 24px}
h1{font-size:24px;margin:0 0 6px}p{color:#52514e}form{margin-top:24px;display:grid;gap:14px}
label{font-weight:600;font-size:13px}input[type=text],input[type=file]{width:100%;padding:9px;border:1px solid #ccc;border-radius:7px;font:inherit;background:#fff}
.or{text-align:center;color:#898781;font-size:13px}button{padding:11px;border:0;border-radius:7px;background:#2a78d6;color:#fff;font:inherit;font-weight:600;cursor:pointer}</style>
<h1>MERU root traversal</h1>
<p>Upload an image <b>or</b> type any text; the real MERU model embeds it, then walks the geodesic
from your query to the root of hyperbolic space — retrieving a more abstract concept at each step.</p>
<form action=/traverse method=post enctype=multipart/form-data>
  <div><label>Image</label><input type=file name=image accept=image/*></div>
  <div class=or>— or —</div>
  <div><label>Text</label><input type=text name=text placeholder="a golden retriever"></div>
  <button type=submit>Traverse →</button>
</form>"""

app = Flask(__name__)


@app.route("/")
def index():
    return FORM


@app.route("/traverse", methods=["POST"])
def traverse():
    file, text = request.files.get("image"), request.form.get("text", "").strip()
    if file and file.filename:
        from PIL import Image
        query, title = encode_image(Image.open(file.stream)), file.filename
    elif text:
        query, title = encode_text([text])[0], f"“{text}”"
    else:
        return FORM
    page = traversal_scene(query, bank_l, BANK, k=k, chart="lorentz").html(title=f"MERU traversal — {title}")
    return Response(page, mimetype="text/html")


print(f"MERU root traversal → http://{args.host}:{args.port}")
app.run(host=args.host, port=args.port)
