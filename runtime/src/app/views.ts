/** three.js views — single stack: orthographic cameras for 2D charts,
 * a perspective camera for the Lorentz hyperboloid. All views consume the
 * same Derived data (Lorentz coordinates) and only differ in projection.
 *
 * 2D frames scale with R = 1/sqrt(-K) (the disk always fills the panel; the
 * metric objects migrate). The 3D view is the opposite: apex pinned at the
 * origin (z = x0 - R), camera fixed in ABSOLUTE coordinates over a fixed
 * spatial window — so changing K visibly flattens/sharpens the bowl.
 * Static geometry is rebuilt when the curvature changes. */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CHARTS } from "../kernel/charts";
import { Vec } from "../kernel/lorentz";
import { ChartKey, Derived, SceneJSON, SceneState } from "./state";

const S = 6;              // absolute spatial half-width of the 3D window
const BALL_EDGE = 0.995;  // chart-coord clamp (keeps the ball formulas in domain)

// paper palette (validated: dataviz slots 1+2 on light surface #fcfcfb)
const COLORS = {
  point: "#2a78d6", curve: "#52514e", grid: "#dddcd4", boundary: "#898781",
  surface: "#9ec5f4", bg: "#fcfcfb",
};

const FRAME: Record<string, { c: [number, number]; hh: number }> = {
  poincare: { c: [0, 0], hh: 1.08 },
  klein: { c: [0, 0], hh: 1.08 },
  halfplane: { c: [0, 2.1], hh: 2.5 },
};

/** Polar geodesic grid in Lorentz hub coords: rays from the origin + circles
 * at absolute hyperbolic distances (0.5, 1, ...) — under a curvature change
 * the disk stays put on screen and the circles migrate: that IS the lesson. */
const gridCurves = (k: number, dmax: number): Vec[][] => {
  const R = 1 / Math.sqrt(-k);
  const pt = (d: number, th: number): Vec =>
    [R * Math.cosh(d / R), R * Math.sinh(d / R) * Math.cos(th), R * Math.sinh(d / R) * Math.sin(th)];
  const curves: Vec[][] = [];
  for (let j = 0; j < 12; j++)
    curves.push(Array.from({ length: 49 }, (_, i) => pt((dmax * i) / 48, (j * Math.PI) / 6)));
  for (let d = 0.5; d <= dmax + 1e-9; d += 0.5)
    curves.push(Array.from({ length: 97 }, (_, i) => pt(d, (2 * Math.PI * i) / 96)));
  return curves;
};

export class HypView {
  el: HTMLElement;
  renderer = new THREE.WebGLRenderer({ antialias: true });
  scene3 = new THREE.Scene();
  camera!: THREE.Camera;
  controls?: OrbitControls;
  statics = new THREE.Group(); // fill + grid + boundary + surface; rebuilt on curvature change
  surface?: THREE.Mesh;
  kNow = NaN;
  spheres = new Map<string, THREE.Mesh>();
  lines = new Map<string, THREE.Line>();
  labels = new Map<string, { div: HTMLElement; world: THREE.Vector3 }>();
  raycaster = new THREE.Raycaster();
  dragId: string | null = null;

  constructor(parent: HTMLElement, public chart: ChartKey, public state: SceneState) {
    this.el = document.createElement("div");
    this.el.className = "hypview";
    parent.appendChild(this.el);
    this.el.appendChild(this.renderer.domElement);
    const tag = document.createElement("div");
    tag.className = "viewtag";
    tag.textContent = chart === "lorentz" ? "lorentz hyperboloid" : `${chart} model`;
    this.el.appendChild(tag);
    this.scene3.background = new THREE.Color(COLORS.bg);
    this.scene3.add(this.statics);

    if (chart === "lorentz") {
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(7.5, -11, 8); // fixed in absolute coords — curvature must be visible
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.target.set(0, 0, 1.6);
      this.controls.enableDamping = true;
    } else {
      this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    }
    this.setCurvature(state.k);

    window.addEventListener("resize", () => this.resize());
    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", (e) => this.onDown(e));
    dom.addEventListener("pointermove", (e) => this.onMove(e));
    dom.addEventListener("pointerup", () => this.onUp());
  }

  get R() {
    return 1 / Math.sqrt(-this.kNow);
  }

  /** Lorentz hub coords -> this view's world coords (3D: apex pinned at z=0). */
  project(x: Vec): THREE.Vector3 {
    if (this.chart === "lorentz") return new THREE.Vector3(x[1], x[2], x[0] - this.R);
    const p = CHARTS[this.chart].fromLorentz(x, this.kNow);
    return new THREE.Vector3(p[0], p[1], 0);
  }

  /** Rebuild all curvature-dependent statics and reframe 2D cameras. */
  setCurvature(k: number) {
    if (k === this.kNow) return;
    this.kNow = k;
    const R = this.R;
    this.statics.clear();
    this.surface = undefined;
    if (this.chart === "lorentz") {
      this.surface = this.makeSurface();
      this.statics.add(this.surface);
      this.statics.add(...gridCurves(k, R * Math.asinh(S / R)).map((pts) => this.curve(pts, COLORS.grid)));
    } else {
      this.addFillAndBoundary();
      // rays reach chart radius tanh(3) R = 0.995 R — visually touching the rim
      this.statics.add(...gridCurves(k, 6 * R).map((pts) => this.curve(pts, COLORS.grid)));
      this.resize();
    }
  }

  private makeSurface(): THREE.Mesh {
    const R = this.R, tmax = Math.asinh(S / R), N = 32, T = 64, pos: number[] = [], idx: number[] = [];
    for (let i = 0; i <= N; i++)
      for (let j = 0; j <= T; j++) {
        const t = (tmax * i) / N, th = (2 * Math.PI * j) / T;
        pos.push(R * Math.sinh(t) * Math.cos(th), R * Math.sinh(t) * Math.sin(th), R * Math.cosh(t) - R);
      }
    for (let i = 0; i < N; i++)
      for (let j = 0; j < T; j++) {
        const a = i * (T + 1) + j, b = a + T + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setIndex(idx);
    return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: COLORS.surface, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false,
    }));
  }

  private addFillAndBoundary() {
    const R = this.R;
    const fill = new THREE.MeshBasicMaterial({ color: COLORS.surface, transparent: true, opacity: 0.16, depthWrite: false });
    if (this.chart === "halfplane") {
      const rect = new THREE.Mesh(new THREE.PlaneGeometry(16 * R, 8 * R), fill);
      rect.position.set(0, 4 * R, -0.01);
      this.statics.add(rect);
      this.statics.add(this.curve([[-8 * R, 0], [8 * R, 0]].map(([a, b]) => [0, a, b]), COLORS.boundary, true));
    } else {
      const disk = new THREE.Mesh(new THREE.CircleGeometry(R, 128), fill);
      disk.position.z = -0.01;
      this.statics.add(disk);
      const pts: Vec[] = Array.from({ length: 193 }, (_, i) =>
        [0, R * Math.cos((2 * Math.PI * i) / 192), R * Math.sin((2 * Math.PI * i) / 192)]);
      this.statics.add(this.curve(pts, COLORS.boundary, true));
    }
  }

  /** raw=true: points are already view coords (index 0 unused). */
  private curve(pts: Vec[], color: string, raw = false): THREE.Line {
    const v3 = pts.map((p) => (raw ? new THREE.Vector3(p[1], p[2], 0) : this.project(p)));
    return new THREE.Line(new THREE.BufferGeometry().setFromPoints(v3), new THREE.LineBasicMaterial({ color }));
  }

  private ensureLabel(key: string): { div: HTMLElement; world: THREE.Vector3 } {
    let l = this.labels.get(key);
    if (!l) {
      const div = document.createElement("div");
      div.className = "hyplabel";
      this.el.appendChild(div);
      l = { div, world: new THREE.Vector3() };
      this.labels.set(key, l);
    }
    return l;
  }

  update(d: Derived) {
    this.setCurvature(this.state.k);
    for (const [id, { x, spec }] of d.points) {
      let s = this.spheres.get(id);
      if (!s) {
        s = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12),
          new THREE.MeshBasicMaterial({ color: spec.color ?? COLORS.point }));
        s.userData = spec;
        this.scene3.add(s);
        this.spheres.set(id, s);
      }
      s.scale.setScalar(this.chart === "lorentz" ? 0.11 : FRAME[this.chart].hh * this.R * 0.03);
      s.position.copy(this.project(x));
      if (spec.label) {
        const l = this.ensureLabel(`pt:${id}`);
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
      const l = this.ensureLabel(id);
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
      const { hh, c } = FRAME[this.chart], R = this.R;
      this.camera.left = c[0] * R - (hh * R * w) / h;
      this.camera.right = c[0] * R + (hh * R * w) / h;
      this.camera.top = (c[1] + hh) * R;
      this.camera.bottom = (c[1] - hh) * R;
      this.camera.position.set(c[0] * R, c[1] * R, 5);
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
    let spatial: Vec | undefined;
    if (this.chart === "lorentz") {
      const hit = this.raycaster.intersectObject(this.surface!)[0];
      if (hit) spatial = [hit.point.x, hit.point.y];
    } else {
      const p = new THREE.Vector3();
      if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), p)) return;
      let c: Vec = [p.x, p.y];
      if (this.chart === "halfplane") c[1] = Math.max(c[1], 0.03 * this.R);
      else {
        const lim = BALL_EDGE * this.R, n = Math.hypot(c[0], c[1]);
        if (n > lim) c = [(c[0] * lim) / n, (c[1] * lim) / n];
      }
      const x = CHARTS[this.chart].toLorentz(c, this.kNow);
      spatial = [x[1], x[2]];
    }
    if (!spatial) return;
    const n = Math.hypot(spatial[0], spatial[1]); // keep inside the 3D window so views stay consistent
    if (n > 0.98 * S) spatial = [(spatial[0] * 0.98 * S) / n, (spatial[1] * 0.98 * S) / n];
    this.state.movePoint(this.dragId, spatial);
  }

  private onUp() {
    this.dragId = null;
    if (this.controls) this.controls.enabled = true;
  }
}

export function mount(root: HTMLElement, scene: SceneJSON) {
  const state = new SceneState(scene);
  if (scene.curvatureSlider) {
    const bar = document.createElement("div");
    bar.className = "hypctl";
    const readout = document.createElement("span");
    readout.textContent = `K = ${state.k.toFixed(2)}`;
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: "-2.5", max: "-0.25", step: "0.05", value: String(state.k) });
    input.addEventListener("input", () => {
      state.setCurvature(parseFloat(input.value));
      readout.textContent = `K = ${state.k.toFixed(2)}`;
    });
    bar.append("curvature ", input, readout);
    root.before(bar);
  }
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
