/** three.js views — single stack: orthographic cameras for 2D charts,
 * a perspective camera for the Lorentz hyperboloid. All views consume the
 * same Derived data (Lorentz coordinates) and only differ in projection. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CHARTS } from "../kernel/charts";
import { Vec } from "../kernel/lorentz";
import { ChartKey, Derived, SceneJSON, SceneState } from "./state";

const RMAX = 3.0;          // geodesic radius of the rendered hyperboloid patch
const CLAMP = 0.9;         // = tanh(RMAX/2): drag limit on ball charts, keeps views consistent
const COLORS = { point: "#5b8def", curve: "#e8e8e8", grid: "#3a3a3a", boundary: "#777777", surface: "#2a3550" };

const FRAME: Record<string, { c: [number, number]; hh: number }> = {
  poincare: { c: [0, 0], hh: 1.15 },
  klein: { c: [0, 0], hh: 1.15 },
  halfplane: { c: [0, 2.1], hh: 2.5 },
};

/** Polar geodesic grid in Lorentz coords: rays from the origin + distance circles. */
const gridCurves = (): Vec[][] => {
  const curves: Vec[][] = [];
  const pt = (t: number, th: number): Vec => [Math.cosh(t), Math.sinh(t) * Math.cos(th), Math.sinh(t) * Math.sin(th)];
  for (let k = 0; k < 12; k++)
    curves.push(Array.from({ length: 33 }, (_, i) => pt((RMAX * i) / 32, (k * Math.PI) / 6)));
  for (let r = 0.5; r <= RMAX + 1e-9; r += 0.5)
    curves.push(Array.from({ length: 97 }, (_, i) => pt(r, (2 * Math.PI * i) / 96)));
  return curves;
};

export class HypView {
  el: HTMLElement;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  scene3 = new THREE.Scene();
  camera!: THREE.Camera;
  controls?: OrbitControls;
  surface?: THREE.Mesh;
  spheres = new Map<string, THREE.Mesh>();
  lines = new Map<string, THREE.Line>();
  labels = new Map<string, { div: HTMLElement; world: THREE.Vector3 }>();
  raycaster = new THREE.Raycaster();
  dragId: string | null = null;

  constructor(parent: HTMLElement, public chart: ChartKey, public state: SceneState) {
    this.el = document.createElement("div");
    this.el.className = "hypview";
    parent.appendChild(this.el);
    const tag = document.createElement("div");
    tag.className = "viewtag";
    tag.textContent = chart === "lorentz" ? "lorentz hyperboloid" : `${chart} model`;
    this.el.appendChild(this.el.appendChild(this.renderer.domElement) && tag);
    this.scene3.background = new THREE.Color("#161616");

    if (chart === "lorentz") {
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(5.2, -7.5, 7.5);
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 0, 2.5);
      this.controls.enableDamping = true;
      this.surface = this.makeSurface();
      this.scene3.add(this.surface);
    } else {
      const { hh, c } = FRAME[chart];
      this.camera = new THREE.OrthographicCamera(-hh, hh, hh, -hh, 0.1, 10);
      this.camera.position.set(c[0], c[1], 5);
      this.addBoundary();
    }
    for (const pts of gridCurves()) this.addStaticCurve(pts, COLORS.grid);

    this.resize();
    window.addEventListener("resize", () => this.resize());
    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", (e) => this.onDown(e));
    dom.addEventListener("pointermove", (e) => this.onMove(e));
    dom.addEventListener("pointerup", () => this.onUp());
  }

  /** Lorentz hub coords -> this view's world coords. */
  project(x: Vec): THREE.Vector3 {
    if (this.chart === "lorentz") return new THREE.Vector3(x[1], x[2], x[0]);
    const p = CHARTS[this.chart].fromLorentz(x);
    return new THREE.Vector3(p[0], p[1], 0);
  }

  private makeSurface(): THREE.Mesh {
    const R = 28, T = 64, pos: number[] = [], idx: number[] = [];
    for (let i = 0; i <= R; i++)
      for (let j = 0; j <= T; j++) {
        const r = (RMAX * i) / R, th = (2 * Math.PI * j) / T;
        pos.push(Math.sinh(r) * Math.cos(th), Math.sinh(r) * Math.sin(th), Math.cosh(r));
      }
    for (let i = 0; i < R; i++)
      for (let j = 0; j < T; j++) {
        const a = i * (T + 1) + j, b = a + T + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: COLORS.surface, transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false,
    }));
  }

  private addBoundary() {
    if (this.chart === "halfplane") {
      this.addStaticCurve([[-8, 0], [8, 0]].map(([a, b]) => [0, a, b]), COLORS.boundary, true);
    } else {
      const pts: Vec[] = Array.from({ length: 129 }, (_, i) =>
        [0, Math.cos((2 * Math.PI * i) / 128), Math.sin((2 * Math.PI * i) / 128)]);
      this.addStaticCurve(pts, COLORS.boundary, true);
    }
  }

  /** raw=true: points are already view coords (x ignored as Lorentz). */
  private addStaticCurve(pts: Vec[], color: string, raw = false) {
    const v3 = pts.map((p) => (raw ? new THREE.Vector3(p[1], p[2], 0) : this.project(p)));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(v3), new THREE.LineBasicMaterial({ color }));
    this.scene3.add(line);
  }

  private ensureLabel(key: string, cls: string): { div: HTMLElement; world: THREE.Vector3 } {
    let l = this.labels.get(key);
    if (!l) {
      const div = document.createElement("div");
      div.className = cls;
      this.el.appendChild(div);
      l = { div, world: new THREE.Vector3() };
      this.labels.set(key, l);
    }
    return l;
  }

  update(d: Derived) {
    for (const [id, { x, spec }] of d.points) {
      let s = this.spheres.get(id);
      if (!s) {
        const r = this.chart === "lorentz" ? 0.09 : FRAME[this.chart].hh * 0.032;
        s = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 12),
          new THREE.MeshBasicMaterial({ color: spec.color ?? COLORS.point }));
        s.userData = spec;
        this.scene3.add(s);
        this.spheres.set(id, s);
      }
      s.position.copy(this.project(x));
      if (spec.label) {
        const l = this.ensureLabel(`pt:${id}`, "hyplabel");
        l.div.textContent = spec.label;
        l.world.copy(s.position);
      }
    }
    for (const [id, { pts, color }] of d.curves) {
      let line = this.lines.get(id);
      if (!line) {
        line = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: color ?? COLORS.curve }));
        this.scene3.add(line);
        this.lines.set(id, line);
      }
      line.geometry.setFromPoints(pts.map((p) => this.project(p)));
    }
    for (const [id, { at, text }] of d.labels) {
      const l = this.ensureLabel(id, "hyplabel");
      l.div.textContent = text;
      l.world.copy(this.project(at));
    }
  }

  frame() {
    this.controls?.update();
    const { clientWidth: w, clientHeight: h } = this.renderer.domElement;
    for (const { div, world } of this.labels.values()) {
      const v = world.clone().project(this.camera);
      div.style.left = `${(v.x * 0.5 + 0.5) * w}px`;
      div.style.top = `${(-v.y * 0.5 + 0.5) * h}px`;
    }
    this.renderer.render(this.scene3, this.camera);
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h, false);
    if (this.camera instanceof THREE.PerspectiveCamera) {
      this.camera.aspect = w / h;
    } else if (this.camera instanceof THREE.OrthographicCamera) {
      const { hh, c } = FRAME[this.chart];
      this.camera.left = c[0] - (hh * w) / h;
      this.camera.right = c[0] + (hh * w) / h;
      this.camera.top = c[1] + hh;
      this.camera.bottom = c[1] - hh;
    }
    (this.camera as THREE.PerspectiveCamera).updateProjectionMatrix();
  }

  private ndc(e: PointerEvent): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  private onDown(e: PointerEvent) {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const hit = this.raycaster.intersectObjects([...this.spheres.values()])
      .find((h) => (h.object.userData as { draggable?: boolean }).draggable);
    if (hit) {
      this.dragId = (hit.object.userData as { id: string }).id;
      if (this.controls) this.controls.enabled = false;
      this.renderer.domElement.setPointerCapture(e.pointerId);
    }
  }

  private onMove(e: PointerEvent) {
    if (!this.dragId) return;
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    if (this.chart === "lorentz") {
      const hit = this.raycaster.intersectObject(this.surface!)[0];
      if (hit) this.state.movePoint(this.dragId, [hit.point.x, hit.point.y]);
    } else {
      const p = new THREE.Vector3();
      if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), p)) return;
      let c: Vec = [p.x, p.y];
      if (this.chart === "halfplane") c[1] = Math.max(c[1], 0.03);
      else {
        const n = Math.hypot(c[0], c[1]);
        if (n > CLAMP) c = [(c[0] * CLAMP) / n, (c[1] * CLAMP) / n];
      }
      const x = CHARTS[this.chart].toLorentz(c);
      this.state.movePoint(this.dragId, [x[1], x[2]]);
    }
  }

  private onUp() {
    this.dragId = null;
    if (this.controls) this.controls.enabled = true;
  }
}

export function mount(root: HTMLElement, scene: SceneJSON) {
  const state = new SceneState(scene);
  const views = scene.views.map((v) => new HypView(root, v.chart, state));
  views.forEach((v) => v.resize()); // re-measure: flex widths settle only once all views exist
  const rerender = () => {
    const d = state.derive();
    views.forEach((v) => v.update(d));
  };
  state.listeners.push(rerender);
  rerender();
  const loop = () => {
    views.forEach((v) => v.frame());
    requestAnimationFrame(loop);
  };
  loop();
}
