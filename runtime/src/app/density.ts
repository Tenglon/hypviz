/** A density field rendered as a smooth texture on a chart's plane. Precomputed
 * high-res in Python (one texture per metric) and shipped as data URIs, so it is
 * crisp and zoomable (bilinear filtering) rather than a blocky grid. The metric
 * can be swapped live. */
import * as THREE from "three";
import { DensityJSON } from "./state";

export class DensityLayer {
  mesh: THREE.Mesh;
  textures: Record<string, THREE.Texture> = {};

  constructor(scene3: THREE.Scene, public spec: DensityJSON) {
    const [x0, x1, y0, y1] = spec.extent;
    const mat = new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0), mat);
    this.mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, -0.005);   // behind points, over the fill
    const loader = new THREE.TextureLoader();
    for (const [m, uri] of Object.entries(spec.textures)) {
      const t = loader.load(uri);
      t.colorSpace = THREE.SRGBColorSpace;
      t.minFilter = THREE.LinearFilter;
      this.textures[m] = t;
    }
    this.setMetric(spec.metric);
    scene3.add(this.mesh);
  }

  setMetric(m: string) {
    const mat = this.mesh.material as THREE.MeshBasicMaterial;
    mat.map = this.textures[m] ?? Object.values(this.textures)[0];
    mat.needsUpdate = true;
  }
}
