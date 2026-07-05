"""Generate golden test vectors for the TS mirror kernel -> runtime/golden/golden.json.

Deterministic (fixed seed). Run from the repo root: .venv/bin/python tools/gen_golden.py
"""
import json
from pathlib import Path

import numpy as np

from hypviz.kernel import lorentz as L, mobius as M
from hypviz.kernel.charts import CHARTS

rng = np.random.default_rng(20260705)
OUT = Path(__file__).resolve().parent.parent / "runtime" / "golden" / "golden.json"


def ball_pt(n, k):
    raw = rng.uniform(-3, 3, n)
    return 0.9 / np.sqrt(-k) * raw / (1 + np.linalg.norm(raw))


cases = {op: [] for op in
          ("dist", "expmap", "logmap", "ptransp", "geodesic",
           "mobius_add", "mobius_scalar", "mobius_dist", "expmap0", "logmap0",
           "poincare", "klein", "halfplane")}

for n, k in ((2, -1.0), (4, -1.0), (2, -0.5), (4, -2.0)):
    for _ in range(8):
        x, y = (L.from_spatial(rng.uniform(-2, 2, n), k) for _ in range(2))
        v = L.to_tangent(x, rng.uniform(-2, 2, n + 1), k)
        v /= max(1, np.sqrt(max(L.mdot(v, v), 0)) / 3)
        t = float(rng.uniform(0, 1))
        cases["dist"].append({"k": k, "x": x, "y": y, "out": L.dist(x, y, k)})
        cases["expmap"].append({"k": k, "x": x, "v": v, "out": L.expmap(x, v, k)})
        cases["logmap"].append({"k": k, "x": x, "y": y, "out": L.logmap(x, y, k)})
        cases["ptransp"].append({"k": k, "x": x, "y": y, "v": v, "out": L.ptransp(x, y, v, k)})
        cases["geodesic"].append({"k": k, "x": x, "y": y, "t": t, "out": L.geodesic(x, y, t, k)})

        a, b = ball_pt(n, k), ball_pt(n, k)
        r = float(rng.uniform(-2, 2))
        cases["mobius_add"].append({"k": k, "a": a, "b": b, "out": M.add(a, b, k)})
        cases["mobius_scalar"].append({"k": k, "r": r, "a": a, "out": M.scalar(r, a, k)})
        cases["mobius_dist"].append({"k": k, "a": a, "b": b, "out": M.dist(a, b, k)})
        cases["expmap0"].append({"k": k, "v": M.logmap0(a, k), "out": a})
        cases["logmap0"].append({"k": k, "a": a, "out": M.logmap0(a, k)})

        for name, chart in CHARTS.items():
            if name == "halfplane" and n != 2:
                continue
            p = ball_pt(n, k) if name != "halfplane" else chart.from_lorentz(L.from_spatial(rng.uniform(-2, 2, 2), k), k)
            cases[name].append({"k": k, "coords": p, "lorentz": chart.to_lorentz(p, k)})


def listify(o):
    if isinstance(o, dict):
        return {k: listify(v) for k, v in o.items()}
    if isinstance(o, list):
        return [listify(v) for v in o]
    return o.tolist() if isinstance(o, np.ndarray) else o


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(listify(cases), indent=1))
print(f"wrote {sum(len(v) for v in cases.values())} cases -> {OUT}")
