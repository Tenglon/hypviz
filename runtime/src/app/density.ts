/** A density field rendered as a smooth texture — on a chart's plane, or, for the
 * hyperboloid, UV-mapped onto the H² surface (reusing the Poincaré-disk texture,
 * since the density is intrinsic). Precomputed high-res in Python and shipped as
 * data URIs keyed by kernel/curvature ("hyperbolic@-0.5", "euclidean", …), so it
 * is crisp, zoomable, and switchable live. The kernel centers (prototypes) are
 * overlaid as points on the disk/half-plane charts. */
import * as THREE from "three";
import { Poincare } from "../kernel/charts";
import { DensityJSON } from "./state";

const RMAX = 0.8;   // hyperboloid disk cap: height grows ~1/(1-r²), so keep the bowl framable

export class DensityLayer {
  mesh: THREE.Mesh;
  textures: Record<string, THREE.Texture> = {};

  constructor(scene3: THREE.Scene, public spec: DensityJSON) {
    const mat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(spec.surface ? DensityLayer.hyperboloid() : this.plane(spec.extent), mat);
    const loader = new THREE.TextureLoader();
    for (const [key, uri] of Object.entries(spec.textures)) {
      const t = loader.load(uri);
      t.colorSpace = THREE.SRGBColorSpace;
      t.minFilter = THREE.LinearFilter;
      this.textures[key] = t;
    }
    this.setKey(spec.metric);
    scene3.add(this.mesh);
    if (spec.points && !spec.surface) this.addPrototypes(scene3, spec.points);
  }

  private addPrototypes(scene3: THREE.Scene, points: number[][]) {
    const pos = new Float32Array(points.length * 3);
    points.forEach((p, i) => pos.set([p[0], p[1], 0.01], i * 3));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    scene3.add(new THREE.Points(g, new THREE.PointsMaterial({ color: "#22d3ee", size: 4, sizeAttenuation: false })));
  }

  private plane([x0, x1, y0, y1]: number[]): THREE.PlaneGeometry {
    const g = new THREE.PlaneGeometry(x1 - x0, y1 - y0);
    g.translate((x0 + x1) / 2, (y0 + y1) / 2, -0.005);       // behind points, over the fill
    return g;
  }

  /** H² surface (unit disk lifted to the Lorentz hyperboloid), UV = disk position,
   * so the Poincaré density texture paints the intrinsic density on the surface. */
  private static hyperboloid(): THREE.BufferGeometry {
    const N = 96, pos: number[] = [], uv: number[] = [], idx: number[] = [];
    for (let i = 0; i <= N; i++)
      for (let j = 0; j <= N; j++) {
        let a = -RMAX + (2 * RMAX * i) / N, b = -RMAX + (2 * RMAX * j) / N;
        const r = Math.hypot(a, b);
        if (r > RMAX) { a *= RMAX / r; b *= RMAX / r; }        // clamp corners to the rim → clean circular edge
        const x = Poincare.toLorentz([a, b], -1);
        pos.push(x[1], x[2], x[0] - 1);
        uv.push((a + 1) / 2, (b + 1) / 2);
      }
    const row = N + 1;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const p = i * row + j;
        idx.push(p, p + 1, p + row, p + 1, p + row + 1, p + row);
      }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx);
    return g;
  }

  setKey(key: string) {
    const mat = this.mesh.material as THREE.MeshBasicMaterial;
    mat.map = this.textures[key] ?? Object.values(this.textures)[0];
    mat.needsUpdate = true;
  }
}
