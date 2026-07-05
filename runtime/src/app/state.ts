/** Scene JSON schema + reactive state. Scenes are tiny (tens of objects),
 * so "reactivity" is simply: recompute every derived object on any change. */
import * as L from "../kernel/lorentz";
import { Vec } from "../kernel/lorentz";

export type PointJSON = { id: string; type: "point"; spatial: number[]; draggable?: boolean; label?: string; color?: string };
export type GeodesicJSON = { id: string; type: "geodesic"; from: string; to: string; color?: string };
export type DistanceLabelJSON = { id: string; type: "distance_label"; from: string; to: string };
export type ObjJSON = PointJSON | GeodesicJSON | DistanceLabelJSON;
export type ChartKey = "poincare" | "klein" | "halfplane" | "lorentz";
export type SceneJSON = { views: { chart: ChartKey }[]; objects: ObjJSON[]; curvature?: number; curvatureSlider?: boolean };

/** Everything a view needs to draw, all in Lorentz hub coordinates. */
export interface Derived {
  points: Map<string, { x: Vec; spec: PointJSON }>;
  curves: Map<string, { pts: Vec[]; color?: string }>;
  labels: Map<string, { at: Vec; text: string }>;
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
    const k = this.k;
    const P = (id: string) => L.fromSpatial(this.spatial.get(id)!, k);
    const d: Derived = { points: new Map(), curves: new Map(), labels: new Map() };
    for (const o of this.scene.objects) {
      if (o.type === "point") d.points.set(o.id, { x: P(o.id), spec: o });
      else if (o.type === "geodesic") d.curves.set(o.id, { pts: sample(P(o.from), P(o.to), k), color: o.color });
      else if (o.type === "distance_label")
        d.labels.set(o.id, { at: L.geodesic(P(o.from), P(o.to), 0.5, k), text: `d = ${L.dist(P(o.from), P(o.to), k).toFixed(3)}` });
    }
    return d;
  }
}
