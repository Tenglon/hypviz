/** Gyrovector ops on the Poincare ball of curvature K < 0 (c = -K) — mirror
 * of src/hypviz/kernel/mobius.py. */
import { Vec } from "./lorentz";

const EPS = 1e-15;
const sq = (p: Vec) => p.reduce((s, v) => s + v * v, 0);
const dot = (a: Vec, b: Vec) => a.reduce((s, ai, i) => s + ai * b[i], 0);
const neg = (a: Vec) => a.map((v) => -v);

/** Mobius addition a (+) b — non-commutative, non-associative. */
export const add = (a: Vec, b: Vec, k = -1): Vec => {
  const c = -k, ab = dot(a, b), a2 = sq(a), b2 = sq(b);
  const d = 1 + 2 * c * ab + c * c * a2 * b2;
  return a.map((ai, i) => ((1 + 2 * c * ab + c * b2) * ai + (1 - c * a2) * b[i]) / d);
};

/** Mobius scalar multiplication r (x) a. */
export const scalar = (r: number, a: Vec, k = -1): Vec => {
  const n = Math.sqrt(-k * sq(a));
  return scaleDir(Math.tanh(r * Math.atanh(n)), n, a);
};

/** Gyration gyr[a,b]v. */
export const gyr = (a: Vec, b: Vec, v: Vec, k = -1): Vec =>
  add(neg(add(a, b, k)), add(a, add(b, v, k), k), k);

/** d(p,q) = (2/sqrt(-K)) artanh(sqrt(-K)|(-p) (+) q|) — equals the Lorentz distance. */
export const dist = (p: Vec, q: Vec, k = -1): number =>
  (2 / Math.sqrt(-k)) * Math.atanh(Math.sqrt(-k * sq(add(neg(p), q, k))));

export const expmap0 = (v: Vec, k = -1): Vec => scaleDir(Math.tanh(Math.sqrt(-k * sq(v))), Math.sqrt(-k * sq(v)), v);
export const logmap0 = (p: Vec, k = -1): Vec => scaleDir(Math.atanh(Math.sqrt(-k * sq(p))), Math.sqrt(-k * sq(p)), p);

/** (f / n) * a with the 0/0 limit at n=0. */
const scaleDir = (f: number, n: number, a: Vec): Vec => a.map((ai) => (f / Math.max(n, EPS)) * ai);
