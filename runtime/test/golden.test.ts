import { describe, expect, it } from "vitest";
import golden from "../golden/golden.json";
import { CHARTS } from "../src/kernel/charts";
import * as L from "../src/kernel/lorentz";
import * as M from "../src/kernel/mobius";

const close = (got: number | number[], want: number | number[], tol = 1e-10) => {
  const g = ([] as number[]).concat(got), w = ([] as number[]).concat(want);
  g.forEach((v, i) => expect(Math.abs(v - w[i])).toBeLessThan(tol * (1 + Math.abs(w[i]))));
};

describe("TS kernel matches Python golden vectors", () => {
  it("lorentz.dist", () => golden.dist.forEach((c) => close(L.dist(c.x, c.y), c.out)));
  it("lorentz.expmap", () => golden.expmap.forEach((c) => close(L.expmap(c.x, c.v), c.out)));
  it("lorentz.logmap", () => golden.logmap.forEach((c) => close(L.logmap(c.x, c.y), c.out)));
  it("lorentz.ptransp", () => golden.ptransp.forEach((c) => close(L.ptransp(c.x, c.y, c.v), c.out)));
  it("lorentz.geodesic", () => golden.geodesic.forEach((c) => close(L.geodesic(c.x, c.y, c.t), c.out)));

  it("mobius.add", () => golden.mobius_add.forEach((c) => close(M.add(c.a, c.b), c.out)));
  it("mobius.scalar", () => golden.mobius_scalar.forEach((c) => close(M.scalar(c.r, c.a), c.out)));
  it("mobius.dist", () => golden.mobius_dist.forEach((c) => close(M.dist(c.a, c.b), c.out)));
  it("mobius.expmap0", () => golden.expmap0.forEach((c) => close(M.expmap0(c.v), c.out)));
  it("mobius.logmap0", () => golden.logmap0.forEach((c) => close(M.logmap0(c.a), c.out)));

  for (const [name, chart] of Object.entries(CHARTS)) {
    it(`charts.${name} round-trips through Lorentz`, () =>
      (golden as any)[name].forEach((c: { coords: number[]; lorentz: number[] }) => {
        close(chart.toLorentz(c.coords), c.lorentz);
        close(chart.fromLorentz(c.lorentz), c.coords);
      }));
  }
});
