/** Gyrovector ops on the Poincare ball (c=1) — mirror of src/hypviz/kernel/mobius.py. */
import { Vec } from "./lorentz";

const EPS = 1e-15;
const sq = (p: Vec) => p.reduce((s, v) => s + v * v, 0);
const dot = (a: Vec, b: Vec) => a.reduce((s, ai, i) => s + ai * b[i], 0);
const neg = (a: Vec) => a.map((v) => -v);

/** Mobius addition a (+) b — non-commutative, non-associative. */
export const add = (a: Vec, b: Vec): Vec => {
  const ab = dot(a, b), a2 = sq(a), b2 = sq(b);
  const d = 1 + 2 * ab + a2 * b2;
  return a.map((ai, i) => ((1 + 2 * ab + b2) * ai + (1 - a2) * b[i]) / d);
};

/** Mobius scalar multiplication r (x) a. */
export const scalar = (r: number, a: Vec): Vec => {
  const n = Math.sqrt(sq(a));
  return scaleDir(Math.tanh(r * Math.atanh(n)), n, a);
};

/** Gyration gyr[a,b]v. */
export const gyr = (a: Vec, b: Vec, v: Vec): Vec => add(neg(add(a, b)), add(a, add(b, v)));

/** d(p,q) = 2 artanh|(-p) (+) q| — equals the Lorentz distance. */
export const dist = (p: Vec, q: Vec): number => 2 * Math.atanh(Math.sqrt(sq(add(neg(p), q))));

export const expmap0 = (v: Vec): Vec => scaleDir(Math.tanh(Math.sqrt(sq(v))), Math.sqrt(sq(v)), v);
export const logmap0 = (p: Vec): Vec => scaleDir(Math.atanh(Math.sqrt(sq(p))), Math.sqrt(sq(p)), p);

/** (f / n) * a with the 0/0 limit at n=0. */
const scaleDir = (f: number, n: number, a: Vec): Vec => a.map((ai) => (f / Math.max(n, EPS)) * ai);
