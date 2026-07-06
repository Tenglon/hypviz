/** Scene JSON schema + reactive state. Scenes are tiny (tens of objects),
 * so "reactivity" is simply: recompute every derived object on any change. */
import { Poincare } from "../kernel/charts";
import * as L from "../kernel/lorentz";
import { Vec } from "../kernel/lorentz";
import * as M from "../kernel/mobius";

export type PointJSON = { id: string; type: "point"; spatial: number[]; draggable?: boolean; label?: string; color?: string };
export type GeodesicJSON = { id: string; type: "geodesic"; from: string; to: string; color?: string };
export type DistanceLabelJSON = { id: string; type: "distance_label"; from: string; to: string };
export type LogVectorJSON = { id: string; type: "log_vector"; base: string; to: string; color?: string };
export type MobiusSumJSON = { id: string; type: "mobius_sum"; a: string; b: string; label?: string; color?: string };
export type TangentPlaneJSON = { id: string; type: "tangent_plane"; at: string };
export type MetricCircleJSON = { id: string; type: "metric_circle"; at: string; radius: number; color?: string };
export type TransportLoopJSON = { id: string; type: "transport_loop"; points: string[]; colors: { initial: string; mid: string; returned: string } };
export type GyroplaneJSON = { id: string; type: "gyroplane"; p: string; normal: string; test?: string; levels: number[]; colors: { plane: string; pos: string; neg: string; normal: string; perp: string } };
export type CloudJSON = { id: string; type: "cloud"; spatial: number[][]; colors: string[]; labels?: string[]; parent?: number[]; pruned?: number[] };
export type ObjJSON = PointJSON | GeodesicJSON | DistanceLabelJSON | LogVectorJSON | MobiusSumJSON | TangentPlaneJSON | MetricCircleJSON | TransportLoopJSON | GyroplaneJSON | CloudJSON;
export type ChartKey = "poincare" | "klein" | "halfplane" | "hemisphere" | "lorentz";
export type LegendEntry = { kind: "line" | "arrow" | "point" | "circle" | "area"; color: string; label: string };
export type SceneJSON = { views: { chart: ChartKey }[]; objects: ObjJSON[]; curvature?: number; curvatureSlider?: boolean; legend?: LegendEntry[] };

/** Everything a view needs to draw, all in Lorentz hub coordinates. */
export interface Derived {
  points: Map<string, { x: Vec; spec: PointJSON }>;
  curves: Map<string, { pts: Vec[]; color?: string }>;
  labels: Map<string, { at: Vec; text: string }>;
  arrows: Map<string, { at: Vec; v: Vec; color?: string }>;
  planes: Map<string, { at: Vec }>;
  tcircles: Map<string, { at: Vec; r: number; color?: string }>;
}

export const sample = (x: Vec, y: Vec, k: number, n = 64): Vec[] =>
  Array.from({ length: n + 1 }, (_, i) => L.geodesic(x, y, i / n, k));

export class SceneState {
  spatial = new Map<string, Vec>(); // the draggable state: Lorentz spatial coords per point
  listeners: (() => void)[] = [];

  constructor(public scene: SceneJSON) {
    for (const o of scene.objects) if (o.type === "point") this.spatial.set(o.id, o.spatial);
  }

  get k() {
    return this.scene.curvature ?? -1;
  }

  private notify() {
    this.listeners.forEach((f) => f());
  }

  movePoint(id: string, spatial: Vec) {
    this.spatial.set(id, spatial);
    this.notify();
  }

  setCurvature(k: number) {
    this.scene.curvature = k;
    this.notify();
  }

  derive(): Derived {
    const k = this.k, objs = this.scene.objects;
    // pass 1: resolve every object that IS a point to its Lorentz position
    const pos = new Map<string, Vec>();
    for (const o of objs) if (o.type === "point") pos.set(o.id, L.fromSpatial(this.spatial.get(o.id)!, k));
    for (const o of objs)
      if (o.type === "mobius_sum") {
        const pa = Poincare.fromLorentz(pos.get(o.a)!, k), pb = Poincare.fromLorentz(pos.get(o.b)!, k);
        pos.set(o.id, Poincare.toLorentz(M.add(pa, pb, k), k));
      }
    // pass 2: build the drawables
    const d: Derived = { points: new Map(), curves: new Map(), labels: new Map(), arrows: new Map(), planes: new Map(), tcircles: new Map() };
    for (const o of objs) {
      if (o.type === "point") d.points.set(o.id, { x: pos.get(o.id)!, spec: o });
      else if (o.type === "mobius_sum")
        d.points.set(o.id, { x: pos.get(o.id)!, spec: { id: o.id, type: "point", spatial: [], label: o.label, color: o.color } });
      else if (o.type === "geodesic") d.curves.set(o.id, { pts: sample(pos.get(o.from)!, pos.get(o.to)!, k), color: o.color });
      else if (o.type === "log_vector")
        d.arrows.set(o.id, { at: pos.get(o.base)!, v: L.logmap(pos.get(o.base)!, pos.get(o.to)!, k), color: o.color });
      else if (o.type === "tangent_plane") d.planes.set(o.id, { at: pos.get(o.at)! });
      else if (o.type === "metric_circle") d.tcircles.set(o.id, { at: pos.get(o.at)!, r: o.radius, color: o.color });
      else if (o.type === "transport_loop") {
        const P = o.points.map((id) => pos.get(id)!);
        const unit = (u: Vec) => L.scale(1 / Math.sqrt(L.mdot(u, u)), u);
        let v = L.scale(0.7, unit(L.logmap(P[0], P[1], k)));   // initial vector at x, along the first edge
        const v0 = v;
        d.arrows.set(`${o.id}:i`, { at: P[0], v: v0, color: o.colors.initial });
        for (let i = 0; i < P.length; i++) {
          const b = P[(i + 1) % P.length];
          v = L.ptransp(P[i], b, v, k);                        // transport along edge i → i+1
          if (i < P.length - 1) d.arrows.set(`${o.id}:${i}`, { at: b, v, color: o.colors.mid });
        }
        d.arrows.set(`${o.id}:r`, { at: P[0], v, color: o.colors.returned });   // returned, rotated
        const cos = L.mdot(v0, v) / Math.sqrt(L.mdot(v0, v0) * L.mdot(v, v));
        const deg = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
        d.labels.set(o.id, { at: P[0], text: `holonomy ${deg.toFixed(1)}°` });
      } else if (o.type === "gyroplane") {
        const P = pos.get(o.p)!, H = pos.get(o.normal)!;
        const unit = (u: Vec) => L.scale(1 / Math.sqrt(L.mdot(u, u)), u);
        const w = L.logmap(P, H, k);
        if (L.mdot(w, w) > 1e-9) {
          const m = unit(w);                                  // Minkowski-unit normal to the plane
          let e = L.toTangent(P, [0, 0, 1], k);               // any tangent not parallel to m
          if (Math.abs(L.mdot(unit(e), m)) > 0.99) e = L.toTangent(P, [0, 1, 0], k);
          const u = unit(L.add(e, L.scale(-L.mdot(e, m), m))); // plane direction (⊥ m) in T_p
          const S = 3, N = 48;
          const contour = (dist: number) => Array.from({ length: N + 1 }, (_, i) => {
            const g = L.expmap(P, L.scale(-S + (2 * S * i) / N, u), k);   // walk along the geodesic
            return L.expmap(g, L.scale(dist, m), k);                      // push perpendicular by `dist`
          });
          d.curves.set(`${o.id}:0`, { pts: contour(0), color: o.colors.plane });
          for (const lv of o.levels) {
            d.curves.set(`${o.id}:+${lv}`, { pts: contour(lv), color: o.colors.pos });
            d.curves.set(`${o.id}:-${lv}`, { pts: contour(-lv), color: o.colors.neg });
          }
          d.arrows.set(`${o.id}:n`, { at: P, v: L.scale(0.7, m), color: o.colors.normal });
          for (let t = 1; t <= 3; t++) {                      // gyro-scalar ruler ticks along w
            const g = L.expmap(P, L.scale(0.45 * t, m), k);
            const up = unit(L.ptransp(P, g, u, k));
            d.curves.set(`${o.id}:t${t}`, {
              pts: [L.expmap(g, L.scale(-0.05, up), k), L.expmap(g, L.scale(0.05, up), k)],
              color: o.colors.normal,
            });
          }
          if (o.test) {
            const X = pos.get(o.test)!;
            const dd = Math.asinh(L.mdot(X, m));              // signed distance = the MLR logit
            const foot = L.expmap(X, L.scale(-dd, unit(L.toTangent(X, m, k))), k);
            d.curves.set(`${o.id}:perp`, { pts: sample(X, foot, k, 24), color: o.colors.perp });
            d.labels.set(o.id, { at: X, text: `d = ${dd >= 0 ? "+" : ""}${dd.toFixed(2)}` });
          }
        }
      }
      else if (o.type === "distance_label")
        d.labels.set(o.id, {
          at: L.geodesic(pos.get(o.from)!, pos.get(o.to)!, 0.5, k),
          text: `d = ${L.dist(pos.get(o.from)!, pos.get(o.to)!, k).toFixed(3)}`,
        });
    }
    return d;
  }
}
