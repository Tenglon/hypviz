/** Coordinate charts on H^n, curvature K < 0 — mirror of src/hypviz/kernel/charts.py. */
import { Vec } from "./lorentz";

const R = (k: number) => 1 / Math.sqrt(-k);
const sq = (p: Vec) => p.reduce((s, v) => s + v * v, 0);

/** Complex division (a / b) on [re, im] pairs. */
const cdiv = ([ar, ai]: Vec, [br, bi]: Vec): Vec => {
  const d = br * br + bi * bi;
  return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
};

export const Poincare = {
  name: "poincare" as const,
  toLorentz(p: Vec, k = -1): Vec {
    const r = R(k), d = r * r - sq(p);
    return [(r * (r * r + sq(p))) / d, ...p.map((v) => (2 * r * r * v) / d)];
  },
  fromLorentz(x: Vec, k = -1): Vec {
    const r = R(k);
    return x.slice(1).map((v) => (r * v) / (r + x[0]));
  },
};

export const Klein = {
  name: "klein" as const,
  toLorentz(p: Vec, k = -1): Vec {
    const r = R(k), g = 1 / Math.sqrt(1 - sq(p) / (r * r));
    return [g * r, ...p.map((v) => g * v)];
  },
  fromLorentz(x: Vec, k = -1): Vec {
    return x.slice(1).map((v) => (R(k) * v) / x[0]);
  },
};

export const HalfPlane = {
  name: "halfplane" as const,
  toLorentz(w: Vec, k = -1): Vec {
    const r = R(k);
    const z = cdiv([w[0] / r, w[1] / r - 1], [w[0] / r, w[1] / r + 1]); // z = (w/R-i)/(w/R+i)
    return Poincare.toLorentz([r * z[0], r * z[1]], k);
  },
  fromLorentz(x: Vec, k = -1): Vec {
    const r = R(k);
    const [zr, zi] = Poincare.fromLorentz(x, k).map((v) => v / r);
    const [qr, qi] = cdiv([1 + zr, zi], [1 - zr, -zi]); // (1+z)/(1-z)
    return [-r * qi, r * qr]; // w = R * i * q
  },
};

export const CHARTS = { poincare: Poincare, klein: Klein, halfplane: HalfPlane };
