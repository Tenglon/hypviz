/** Coordinate charts on H^n — mirror of src/hypviz/kernel/charts.py. */
import { Vec } from "./lorentz";

const sq = (p: Vec) => p.reduce((s, v) => s + v * v, 0);

/** Complex division (a / b) on [re, im] pairs. */
const cdiv = ([ar, ai]: Vec, [br, bi]: Vec): Vec => {
  const d = br * br + bi * bi;
  return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
};

export const Poincare = {
  name: "poincare" as const,
  toLorentz(p: Vec): Vec {
    const d = 1 - sq(p);
    return [(1 + sq(p)) / d, ...p.map((v) => (2 * v) / d)];
  },
  fromLorentz(x: Vec): Vec {
    return x.slice(1).map((v) => v / (1 + x[0]));
  },
};

export const Klein = {
  name: "klein" as const,
  toLorentz(k: Vec): Vec {
    const g = 1 / Math.sqrt(1 - sq(k));
    return [g, ...k.map((v) => g * v)];
  },
  fromLorentz(x: Vec): Vec {
    return x.slice(1).map((v) => v / x[0]);
  },
};

export const HalfPlane = {
  name: "halfplane" as const,
  toLorentz(w: Vec): Vec {
    const z = cdiv([w[0], w[1] - 1], [w[0], w[1] + 1]); // z = (w-i)/(w+i)
    return Poincare.toLorentz(z);
  },
  fromLorentz(x: Vec): Vec {
    const [zr, zi] = Poincare.fromLorentz(x);
    const [qr, qi] = cdiv([1 + zr, zi], [1 - zr, -zi]); // (1+z)/(1-z)
    return [-qi, qr]; // w = i * q
  },
};

export const CHARTS = { poincare: Poincare, klein: Klein, halfplane: HalfPlane };
