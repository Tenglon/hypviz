/** Bulk point-cloud layer for the Embedding Atlas: THREE.Points for the nodes
 * (per-point color), faint top-level tree edges for structure, hover tooltip
 * carrying the honest pruned-leaf count, and click-to-highlight of a node's
 * ancestor chain to the root. One layer per view; all views share one tooltip. */
import * as THREE from "three";
import * as L from "../kernel/lorentz";
import { Vec } from "../kernel/lorentz";
import { CloudJSON } from "./state";

type Project = (x: Vec) => THREE.Vector3;
const EDGE_DEPTH = 3;                       // top levels drawn by default
const SAMPLES = 24;

export class CloudLayer {
  points: THREE.Points;
  highlight = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: "#e34948" }));
  lorentz: Vec[];
  depth: number[];
  parent: number[];
  selected = -1;

  constructor(public scene3: THREE.Scene, public spec: CloudJSON, public project: Project,
              public k: number, public pointSize: number) {
    const n = spec.spatial.length;
    this.parent = spec.parent ?? new Array(n).fill(-1);
    this.lorentz = spec.spatial.map((s) => L.fromSpatial(s, k));   // 2D or 3D spatial
    this.depth = this.parent.map(() => 0);
    for (let i = 0; i < n; i++) this.depth[i] = this.parent[i] < 0 ? 0 : this.depth[this.parent[i]] + 1;

    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3), c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const p = project(this.lorentz[i]);
      pos.set([p.x, p.y, p.z], i * 3);
      c.set(spec.colors[i]);
      col.set([c.r, c.g, c.b], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    this.points = new THREE.Points(geo,
      new THREE.PointsMaterial({ size: pointSize, sizeAttenuation: false, vertexColors: true }));
    scene3.add(this.points, this.buildEdges((i) => this.depth[i] <= EDGE_DEPTH, "#d4d3cb"), this.highlight);
  }

  private edgeGeometry(nodes: number[]): THREE.BufferGeometry {
    const seg: number[] = [];
    for (const i of nodes) {
      if (this.parent[i] < 0) continue;
      const pts = Array.from({ length: SAMPLES + 1 }, (_, s) =>
        this.project(L.geodesic(this.lorentz[this.parent[i]], this.lorentz[i], s / SAMPLES, this.k)));
      for (let s = 0; s < SAMPLES; s++) seg.push(pts[s].x, pts[s].y, pts[s].z, pts[s + 1].x, pts[s + 1].y, pts[s + 1].z);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(seg, 3));
    return g;
  }

  private buildEdges(pred: (i: number) => boolean, color: string): THREE.LineSegments {
    const nodes = this.parent.map((_, i) => i).filter(pred);
    return new THREE.LineSegments(this.edgeGeometry(nodes), new THREE.LineBasicMaterial({ color }));
  }

  /** node index under the cursor, or -1. */
  pick(raycaster: THREE.Raycaster, threshold: number): number {
    raycaster.params.Points!.threshold = threshold;
    const hit = raycaster.intersectObject(this.points)[0];
    return hit ? hit.index! : -1;
  }

  ancestors(i: number): number[] {
    const chain = [i];
    while (this.parent[chain[chain.length - 1]] >= 0) chain.push(this.parent[chain[chain.length - 1]]);
    return chain;
  }

  select(i: number) {
    this.selected = i;
    this.highlight.geometry.dispose();
    this.highlight.geometry = i < 0 ? new THREE.BufferGeometry() : this.edgeGeometry(this.ancestors(i));
  }

  tooltip(i: number): string {
    const label = this.spec.labels ? this.spec.labels[i] : `node ${i}`;
    const pruned = this.spec.pruned?.[i] ?? 0;
    return pruned > 0 ? `${label}  ·  +${pruned.toLocaleString()} leaves not shown` : label;
  }
}
