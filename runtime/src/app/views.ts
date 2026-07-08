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
import { CloudLayer } from "./cloud";
import { DensityLayer } from "./density";
import { CHARTS } from "../kernel/charts";
import * as L from "../kernel/lorentz";
import { Vec } from "../kernel/lorentz";
import { ChartKey, Derived, LegendEntry, SceneJSON, SceneState } from "./state";

const S = 8;              // absolute spatial half-width of the 3D window
const BALL_EDGE = 0.995;  // chart-coord clamp (keeps the ball formulas in domain)

// paper palette (validated: dataviz slots 1+2 on light surface #fcfcfb)
const COLORS = {
  point: "#2a78d6", curve: "#52514e", grid: "#dddcd4", gridFaint: "#edece5",
  rim: "#a9a79e", boundary: "#898781", surface: "#9ec5f4", bg: "#fcfcfb",
};

const FRAME: Record<string, { c: [number, number]; hh: number }> = {
  poincare: { c: [0, 0], hh: 1.08 },
  klein: { c: [0, 0], hh: 1.08 },
  halfplane: { c: [0, 1.35], hh: 1.55 },
};

/** Polar geodesic grid in Lorentz hub coords: circles at absolute hyperbolic
 * distances (0.5, 1, ...) and rays — under a curvature change the disk stays
 * put on screen and the circles migrate: that IS the lesson. */
const polar = (k: number, d: number, th: number): Vec => {
  const R = 1 / Math.sqrt(-k);
  return [R * Math.cosh(d / R), R * Math.sinh(d / R) * Math.cos(th), R * Math.sinh(d / R) * Math.sin(th)];
};
const ring = (k: number, d: number): Vec[] =>
  Array.from({ length: 97 }, (_, i) => polar(k, d, (2 * Math.PI * i) / 96));
const ray = (k: number, th: number, d0: number, d1: number): Vec[] =>
  Array.from({ length: 33 }, (_, i) => polar(k, d0 + ((d1 - d0) * i) / 32, th));
const THETAS = Array.from({ length: 12 }, (_, j) => (j * Math.PI) / 6);

/** The shared interactive domain: the geodesic ball the 3D window can show. */
const domain = (k: number) => {
  const R = 1 / Math.sqrt(-k);
  return R * Math.asinh(S / R);
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
  arrows = new Map<string, { line: THREE.Line; cone: THREE.Mesh }>();
  planes = new Map<string, THREE.Mesh>();
  regions = new Map<string, THREE.Mesh>();
  tcircles = new Map<string, THREE.LineLoop>();
  labels = new Map<string, { div: HTMLElement; world: THREE.Vector3 }>();
  clouds: CloudLayer[] = [];
  densities: DensityLayer[] = [];
  hasDensity = false;
  tooltipEl?: HTMLElement;
  onPick?: (i: number) => void;
  raycaster = new THREE.Raycaster();
  dragId: string | null = null;
  zoom2d = 1;
  panX = 0;
  panY = 0;
  private pan?: { sx: number; sy: number; px: number; py: number; moved: boolean };

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

    if (this.is3D) {
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
      this.camera.up.set(0, 0, 1);
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      if (chart === "lorentz") {
        this.camera.position.set(9.5, -14, 10); // fixed in absolute coords — curvature must be visible
        this.controls.target.set(0, 0, 2);
      } // hemisphere camera scales with R, set in setCurvature
    } else {
      this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    }
    this.hasDensity = state.scene.objects.some((o) => o.type === "density" && o.chart === this.chart);
    this.setCurvature(state.k);
    for (const o of state.scene.objects) {
      if (o.type === "cloud")
        this.clouds.push(new CloudLayer(this.scene3, o, (x) => this.project(x), this.kNow, this.is3D ? 5 : 6));
      else if (o.type === "density" && o.chart === this.chart)
        this.densities.push(new DensityLayer(this.scene3, o));
    }

    window.addEventListener("resize", () => this.resize());
    const dom = this.renderer.domElement;
    dom.addEventListener("pointerdown", (e) => this.onDown(e));
    dom.addEventListener("pointermove", (e) => this.onMove(e));
    dom.addEventListener("pointerup", () => this.onUp());
    dom.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    dom.addEventListener("dblclick", () => this.resetView());   // 2D: reset zoom/pan
  }

  get R() {
    return 1 / Math.sqrt(-this.kNow);
  }

  get is3D() {
    return this.chart === "lorentz" || this.chart === "hemisphere" || this.chart === "ball3d";
  }

  /** Lorentz hub coords -> this view's world coords (3D: apex pinned at z=0;
   * hemisphere: the Klein disk lifted onto the upper sphere of radius R). */
  project(x: Vec): THREE.Vector3 {
    if (this.chart === "lorentz") return new THREE.Vector3(x[1], x[2], x[0] - this.R);
    if (this.chart === "hemisphere") {
      const [a, b] = CHARTS.klein.fromLorentz(x, this.kNow);
      return new THREE.Vector3(a, b, Math.sqrt(Math.max(this.R ** 2 - a * a - b * b, 0)));
    }
    if (this.chart === "ball3d") {
      const p = CHARTS.poincare.fromLorentz(x, this.kNow);   // 3D Poincaré ball coords
      return new THREE.Vector3(p[0], p[1], p[2] ?? 0);
    }
    const p = CHARTS[this.chart].fromLorentz(x, this.kNow);
    return new THREE.Vector3(p[0], p[1], 0);
  }

  /** Differential of `project` at x: tangent vector v -> view-space vector. */
  pushforward(x: Vec, v: Vec): THREE.Vector3 {
    const e = 1e-4;
    return this.project(L.expmap(x, L.scale(e, v), this.kNow)).sub(this.project(x)).divideScalar(e);
  }

  /** Rebuild all curvature-dependent statics and reframe cameras.
   * Every view draws the SAME grid over the shared domain, plus the domain rim
   * ring — all views stay in visual lockstep out to that ring. */
  setCurvature(k: number) {
    if (k === this.kNow) return;
    this.kNow = k;
    const R = this.R, D = domain(k);
    this.statics.clear();
    this.surface = undefined;
    if (this.chart === "lorentz" && this.hasDensity) {        // density surface is the content — frame the whole bowl
      this.camera.position.set(5, -6.5, 5);
      this.controls!.target.set(0, 0, 1.4);
      return;
    }
    if (this.chart === "ball3d") {                            // H³ Poincaré ball: boundary sphere only
      const geo = new THREE.SphereGeometry(R, 32, 24);
      this.statics.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: COLORS.surface, transparent: true, opacity: 0.06, side: THREE.BackSide, depthWrite: false,
      })));
      this.statics.add(new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(R, 16, 12)),
        new THREE.LineBasicMaterial({ color: COLORS.grid, transparent: true, opacity: 0.5 })));
      this.camera.position.set(2.6 * R, -2.6 * R, 1.8 * R);
      this.controls!.target.set(0, 0, 0);
      return;
    }
    if (this.chart === "lorentz") {
      this.surface = this.makeSurface((t, th) =>
        [R * Math.sinh(t) * Math.cos(th), R * Math.sinh(t) * Math.sin(th), R * Math.cosh(t) - R], Math.asinh(S / R));
      this.statics.add(this.surface);
    } else {
      if (this.chart === "hemisphere") {
        this.surface = this.makeSurface((t, th) =>
          [R * Math.sin(t) * Math.cos(th), R * Math.sin(t) * Math.sin(th), R * Math.cos(t)], Math.PI / 2);
        this.statics.add(this.surface);
        const eq: Vec[] = Array.from({ length: 129 }, (_, i) =>
          [0, R * Math.cos((2 * Math.PI * i) / 128), R * Math.sin((2 * Math.PI * i) / 128)]);
        this.statics.add(this.curve(eq, COLORS.boundary, true)); // equator = ideal boundary
        this.camera.position.set(2.7 * R, -3.8 * R, 2.9 * R);
        this.controls!.target.set(0, 0, 0.35 * R);
      } else {
        this.addFillAndBoundary();
        this.resize();
      }
      // context beyond the 3D window: faint grid toward the ideal boundary (the space goes on)
      for (const th of THETAS) this.statics.add(this.curve(ray(k, th, D, 6 * R), COLORS.gridFaint));
      for (let d = Math.ceil(D / 0.5) * 0.5; d <= 6 * R + 1e-9; d += 0.5)
        this.statics.add(this.curve(ring(k, d), COLORS.gridFaint));
    }
    for (const th of THETAS) this.statics.add(this.curve(ray(k, th, 0, D), COLORS.grid));
    for (let d = 0.5; d < D; d += 0.5) this.statics.add(this.curve(ring(k, d), COLORS.grid));
    this.statics.add(this.curve(ring(k, D), COLORS.rim)); // the shared window edge
  }

  private makeSurface(f: (t: number, th: number) => number[], tmax: number): THREE.Mesh {
    const N = 32, T = 64, pos: number[] = [], idx: number[] = [];
    for (let i = 0; i <= N; i++)
      for (let j = 0; j <= T; j++) pos.push(...f((tmax * i) / N, (2 * Math.PI * j) / T));
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

  /** Minkowski-orthonormal basis of T_at, for tangent-space glyphs. */
  private tangentBasis(at: Vec): [Vec, Vec] {
    const nrm = (u: Vec) => L.scale(1 / Math.sqrt(L.mdot(u, u)), u);
    const e1 = nrm(L.toTangent(at, [0, 1, 0], this.kNow));
    const raw = L.toTangent(at, [0, 0, 1], this.kNow);
    return [e1, nrm(L.add(raw, L.scale(-L.mdot(e1, raw), e1)))];
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
      s.scale.setScalar(this.chart === "lorentz" ? 0.11 :
        this.chart === "hemisphere" ? 0.055 * this.R : FRAME[this.chart].hh * this.R * 0.03);
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
    for (const [id, { at, v, color }] of d.arrows) {
      let ar = this.arrows.get(id);
      if (!ar) {
        const mat = new THREE.MeshBasicMaterial({ color: color ?? COLORS.curve });
        ar = {
          line: new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: color ?? COLORS.curve })),
          cone: new THREE.Mesh(new THREE.ConeGeometry(0.36, 1, 12), mat),
        };
        this.scene3.add(ar.line, ar.cone);
        this.arrows.set(id, ar);
      }
      const p0 = this.project(at), dir = this.pushforward(at, v), tip = p0.clone().add(dir);
      const hs = this.chart === "lorentz" ? 0.28 :
        this.chart === "hemisphere" ? 0.14 * this.R : FRAME[this.chart].hh * this.R * 0.07;
      ar.line.geometry.setFromPoints([p0, tip]);
      ar.cone.scale.setScalar(hs);
      ar.cone.position.copy(tip).addScaledVector(dir.clone().normalize(), -hs / 2);
      ar.cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    }
    for (const [id, { at }] of d.planes) {
      if (!this.is3D) continue; // in a 2D chart the tangent plane IS the picture plane
      let mesh = this.planes.get(id);
      if (!mesh) {
        mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
          color: "#4a3aa7", transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false,
        }));
        this.scene3.add(mesh);
        this.planes.set(id, mesh);
      }
      const [e1, e2] = this.tangentBasis(at);
      const p0 = this.project(at), sz = this.chart === "hemisphere" ? 0.75 * this.R : 1.4;
      const u1 = this.pushforward(at, e1).setLength(sz), u2 = this.pushforward(at, e2).setLength(sz);
      const c = (s1: number, s2: number) => p0.clone().addScaledVector(u1, s1).addScaledVector(u2, s2);
      mesh.geometry.setFromPoints([c(-1, -1), c(1, -1), c(1, 1), c(-1, -1), c(1, 1), c(-1, 1)]);
    }
    for (const [id, { loop, color }] of d.regions) {
      let mesh = this.regions.get(id);
      if (!mesh) {
        mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
          color, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false,
        }));
        this.scene3.add(mesh);
        this.regions.set(id, mesh);
      }
      const p = loop.map((x) => this.project(x));           // triangle fan from the apex (p[0])
      const v: number[] = [];
      for (let i = 1; i < p.length - 1; i++) v.push(p[0].x, p[0].y, p[0].z, p[i].x, p[i].y, p[i].z, p[i + 1].x, p[i + 1].y, p[i + 1].z);
      mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
    }
    for (const [id, { at, r, color }] of d.tcircles) {
      let loop = this.tcircles.get(id);
      if (!loop) {
        loop = new THREE.LineLoop(new THREE.BufferGeometry(),
          new THREE.LineBasicMaterial({ color: color ?? COLORS.curve }));
        this.scene3.add(loop);
        this.tcircles.set(id, loop);
      }
      // the metric circle {|v|_hyp = r} in T_at, drawn through each view's differential:
      // a Euclidean circle shrinking near the rim on the disk; a tilted circle in 3D
      const [e1, e2] = this.tangentBasis(at);
      const p0 = this.project(at);
      const u1 = this.pushforward(at, e1), u2 = this.pushforward(at, e2);
      loop.geometry.setFromPoints(Array.from({ length: 48 }, (_, i) => {
        const t = (2 * Math.PI * i) / 48;
        return p0.clone().addScaledVector(u1, r * Math.cos(t)).addScaledVector(u2, r * Math.sin(t));
      }));
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
      this.camera.updateProjectionMatrix();
    } else {
      this.applyCamera2D();
    }
  }

  /** Ortho frustum from zoom + pan (bounds are relative to the camera position). */
  private applyCamera2D() {
    if (!(this.camera instanceof THREE.OrthographicCamera)) return;
    const { hh, c } = FRAME[this.chart], R = this.R;
    const el = this.renderer.domElement, half = (hh * R) / this.zoom2d;
    this.camera.left = (-half * el.clientWidth) / el.clientHeight;
    this.camera.right = (half * el.clientWidth) / el.clientHeight;
    this.camera.top = half;
    this.camera.bottom = -half;
    this.camera.position.set(c[0] * R + this.panX, c[1] * R + this.panY, 5);
    this.camera.updateProjectionMatrix();
  }

  private resetView() {
    if (this.is3D) return;
    this.zoom2d = 1;
    this.panX = this.panY = 0;
    this.applyCamera2D();
  }

  private worldXY(cx: number, cy: number): THREE.Vector3 {
    return new THREE.Vector3(this.ndcXY(cx, cy).x, this.ndcXY(cx, cy).y, 0).unproject(this.camera);
  }

  private onWheel(e: WheelEvent) {
    if (this.is3D) return;                              // OrbitControls handles 3D zoom
    e.preventDefault();
    const before = this.worldXY(e.clientX, e.clientY);
    this.zoom2d = Math.min(300, Math.max(0.6, this.zoom2d * Math.exp(-e.deltaY * 0.0015)));
    this.applyCamera2D();
    const after = this.worldXY(e.clientX, e.clientY);   // keep the point under the cursor fixed
    this.panX += before.x - after.x;
    this.panY += before.y - after.y;
    this.applyCamera2D();
  }

  private ndcXY(cx: number, cy: number): THREE.Vector2 {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
  }

  private ndc(e: PointerEvent): THREE.Vector2 {
    return this.ndcXY(e.clientX, e.clientY);
  }

  private pickR() {
    return (this.is3D ? 0.06 : 0.03) * this.R;
  }

  private onDown(e: PointerEvent) {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const hit = this.raycaster.intersectObjects([...this.spheres.values()])
      .find((h) => (h.object.userData as { draggable?: boolean }).draggable);
    if (hit) {
      this.dragId = (hit.object.userData as { id: string }).id;
      if (this.controls) this.controls.enabled = false;
      this.renderer.domElement.setPointerCapture(e.pointerId);
      return;
    }
    if (!this.is3D) {                               // 2D: drag pans; a click (no drag) selects on release
      this.pan = { sx: e.clientX, sy: e.clientY, px: this.panX, py: this.panY, moved: false };
      this.renderer.domElement.setPointerCapture(e.pointerId);
      return;
    }
    this.clickSelect(e.clientX, e.clientY);         // 3D: OrbitControls owns the drag, click selects
  }

  private clickSelect(cx: number, cy: number) {
    if (!this.clouds.length) return;
    this.raycaster.setFromCamera(this.ndcXY(cx, cy), this.camera);
    for (const cl of this.clouds) {
      const i = cl.pick(this.raycaster, this.pickR());
      if (i >= 0) return this.onPick?.(i);
    }
    this.onPick?.(-1);                              // empty click → clear
  }

  private hover(e: PointerEvent) {
    if (!this.clouds.length || !this.tooltipEl) return;
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    for (const cl of this.clouds) {
      const i = cl.pick(this.raycaster, this.pickR());
      if (i >= 0) {
        this.tooltipEl.textContent = cl.tooltip(i);
        Object.assign(this.tooltipEl.style, { display: "block", left: `${e.clientX + 12}px`, top: `${e.clientY + 12}px` });
        return;
      }
    }
    this.tooltipEl.style.display = "none";
  }

  private onMove(e: PointerEvent) {
    if (this.pan) {                                 // 2D drag → pan
      const dpx = e.clientX - this.pan.sx, dpy = e.clientY - this.pan.sy;
      if (Math.hypot(dpx, dpy) > 3) this.pan.moved = true;
      const el = this.renderer.domElement, upp = (2 * FRAME[this.chart].hh * this.R) / this.zoom2d / el.clientHeight;
      this.panX = this.pan.px - dpx * upp;
      this.panY = this.pan.py + dpy * upp;
      this.applyCamera2D();
      return;
    }
    if (!this.dragId) return this.hover(e);
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    let spatial: Vec | undefined;
    if (this.is3D) {
      const hit = this.raycaster.intersectObject(this.surface!)[0];
      if (!hit) return;
      if (this.chart === "lorentz") spatial = [hit.point.x, hit.point.y];
      else {
        let c: Vec = [hit.point.x, hit.point.y]; // hemisphere xy = Klein coords
        const lim = BALL_EDGE * this.R, n = Math.hypot(c[0], c[1]);
        if (n > lim) c = [(c[0] * lim) / n, (c[1] * lim) / n];
        const x = CHARTS.klein.toLorentz(c, this.kNow);
        spatial = [x[1], x[2]];
      }
    } else {
      const p = new THREE.Vector3();
      if (!this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), 0), p)) return;
      let c: Vec = [p.x, p.y];
      if (this.chart === "halfplane") c[1] = Math.max(c[1], 0.03 * this.R);
      else {
        const lim = BALL_EDGE * this.R, n = Math.hypot(c[0], c[1]);
        if (n > lim) c = [(c[0] * lim) / n, (c[1] * lim) / n];
      }
      const x = CHARTS[this.chart as keyof typeof CHARTS].toLorentz(c, this.kNow);
      spatial = [x[1], x[2]];
    }
    if (!spatial) return;
    // clamp to the shared domain: both views stop at the same hyperbolic point
    const R = this.R, lim = R * Math.sinh((0.99 * domain(this.kNow)) / R);
    const n = Math.hypot(spatial[0], spatial[1]);
    if (n > lim) spatial = [(spatial[0] * lim) / n, (spatial[1] * lim) / n];
    this.state.movePoint(this.dragId, spatial);
  }

  private onUp() {
    if (this.pan) {
      if (!this.pan.moved) this.clickSelect(this.pan.sx, this.pan.sy);   // a click, not a pan → select
      this.pan = undefined;
    }
    this.dragId = null;
    if (this.controls) this.controls.enabled = true;
  }
}

export function mount(root: HTMLElement, scene: SceneJSON) {
  const state = new SceneState(scene);
  const bar = document.createElement("div");
  bar.className = "hypctl";
  if (scene.curvatureSlider) {
    const readout = document.createElement("span");
    readout.textContent = `K = ${state.k.toFixed(2)}`;
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: "-2.5", max: "-0.25", step: "0.05", value: String(state.k) });
    input.addEventListener("input", () => {
      state.setCurvature(parseFloat(input.value));
      readout.textContent = `K = ${state.k.toFixed(2)}`;
    });
    bar.append("curvature ", input, readout);
  }
  // legend: scene-specific entries + the standard geometry, identity never color-alone
  const legend = document.createElement("div");
  legend.className = "hyplegend";
  const geom: LegendEntry[] = scene.objects.some((o) => o.type === "density") ? [] : [
    { kind: "line", color: COLORS.grid, label: "distance grid (0.5 apart)" },
    { kind: "circle", color: COLORS.rim, label: "3D-window edge" },
    { kind: "circle", color: COLORS.boundary, label: "ideal boundary (∞)" }];
  const entries: LegendEntry[] = [...(scene.legend ?? []), ...geom];
  for (const { kind, color, label } of entries) {
    const item = document.createElement("span");
    item.className = "item";
    const sw = document.createElement("span");
    sw.className = `sw sw-${kind}`;
    if (kind === "arrow") { sw.textContent = "⟶"; sw.style.color = color; }
    else if (kind === "area") { sw.style.background = color + "2e"; sw.style.borderColor = color; }
    else if (kind === "circle") sw.style.borderColor = color;
    else sw.style.background = color;
    item.append(sw, label);
    legend.appendChild(item);
  }
  root.before(legend);

  const btn = document.createElement("button");
  btn.textContent = "download state";
  btn.title = "Save the current arrangement as JSON — reproduce it in Python via Scene.to_svg(..., state=...)";
  btn.addEventListener("click", () => {
    const data = { curvature: state.k, spatial: Object.fromEntries(state.spatial) };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }));
    a.download = "scene-state.json";
    a.click();
  });
  bar.append(btn);
  root.before(bar);
  const views = scene.views.map((v) => new HypView(root, v.chart, state));
  views.forEach((v) => v.resize()); // re-measure: flex widths settle only once all views exist
  const density = scene.objects.find((o) => o.type === "density") as { textures: Record<string, string> } | undefined;
  if (density) {                                          // kernel switch + curvature slider for density
    const keys = Object.keys(density.textures);
    const metrics = [...new Set(keys.map((k) => k.split("@")[0]))];
    const curvs = scene.densityCurvatures ?? [-1];
    let metric = metrics[0], kIdx = 0;
    const apply = () => {
      const key = metric === "hyperbolic" ? `hyperbolic@${curvs[kIdx]}` : metric;
      views.forEach((v) => v.densities.forEach((d) => d.setKey(key)));
    };
    const bar2 = document.createElement("div");
    bar2.className = "hypctl";
    bar2.append("kernel ");
    metrics.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.textContent = m;
      if (i === 0) btn.classList.add("on");
      btn.addEventListener("click", () => {
        bar2.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
        btn.classList.add("on");
        metric = m;
        slider.style.opacity = m === "hyperbolic" ? "1" : "0.35";
        (slider as HTMLInputElement).disabled = m !== "hyperbolic";
        apply();
      });
      bar2.appendChild(btn);
    });
    const readout = document.createElement("span");
    const slider = document.createElement("input");
    Object.assign(slider, { type: "range", min: "0", max: String(curvs.length - 1), step: "1", value: "0" });
    slider.addEventListener("input", () => {
      kIdx = +slider.value;
      readout.textContent = `K = ${(+curvs[kIdx]).toFixed(2)}`;
      apply();
    });
    readout.textContent = `K = ${(+curvs[0]).toFixed(2)}`;
    bar2.append("  curvature ", slider, readout);
    root.before(bar2);
  }
  // shared tooltip + selection broadcast, so clouds in every view stay in sync
  const tip = document.createElement("div");
  tip.className = "hyptooltip";
  document.body.appendChild(tip);
  const broadcast = (i: number) => views.forEach((v) => v.clouds.forEach((cl) => cl.select(i)));
  views.forEach((v) => { v.tooltipEl = tip; v.onPick = broadcast; });
  (window as unknown as { __hypviz: object }).__hypviz = { state, views }; // console/debug access
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
