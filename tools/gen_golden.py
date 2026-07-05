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


def ball_pt(n):
    raw = rng.uniform(-3, 3, n)
    return 0.9 * raw / (1 + np.linalg.norm(raw))


cases = {op: [] for op in
          ("dist", "expmap", "logmap", "ptransp", "geodesic",
           "mobius_add", "mobius_scalar", "mobius_dist", "expmap0", "logmap0",
           "poincare", "klein", "halfplane")}

for n in (2, 4):
    for _ in range(10):
        x, y = (L.from_spatial(rng.uniform(-2, 2, n)) for _ in range(2))
        v = L.to_tangent(x, rng.uniform(-2, 2, n + 1))
        v /= max(1, np.sqrt(max(L.mdot(v, v), 0)) / 3)
        t = float(rng.uniform(0, 1))
        cases["dist"].append({"x": x, "y": y, "out": L.dist(x, y)})
        cases["expmap"].append({"x": x, "v": v, "out": L.expmap(x, v)})
        cases["logmap"].append({"x": x, "y": y, "out": L.logmap(x, y)})
        cases["ptransp"].append({"x": x, "y": y, "v": v, "out": L.ptransp(x, y, v)})
        cases["geodesic"].append({"x": x, "y": y, "t": t, "out": L.geodesic(x, y, t)})

        a, b = ball_pt(n), ball_pt(n)
        r = float(rng.uniform(-2, 2))
        cases["mobius_add"].append({"a": a, "b": b, "out": M.add(a, b)})
        cases["mobius_scalar"].append({"r": r, "a": a, "out": M.scalar(r, a)})
        cases["mobius_dist"].append({"a": a, "b": b, "out": M.dist(a, b)})
        cases["expmap0"].append({"v": M.logmap0(a), "out": a})
        cases["logmap0"].append({"a": a, "out": M.logmap0(a)})

        for name, chart in CHARTS.items():
            if name == "halfplane" and n != 2:
                continue
            p = ball_pt(n) if name != "halfplane" else chart.from_lorentz(L.from_spatial(rng.uniform(-2, 2, 2)))
            cases[name].append({"coords": p, "lorentz": chart.to_lorentz(p)})


def listify(o):
    if isinstance(o, dict):
        return {k: listify(v) for k, v in o.items()}
    if isinstance(o, list):
        return [listify(v) for v in o]
    return o.tolist() if isinstance(o, np.ndarray) else o


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(listify(cases), indent=1))
print(f"wrote {sum(len(v) for v in cases.values())} cases -> {OUT}")
