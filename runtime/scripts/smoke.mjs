/** Headless end-to-end smoke: load scene 1, drag point a on the Poincaré view,
 * assert the distance label reacts and no JS errors surface. */
import { chromium } from "playwright";

const url = "file:///Users/longteng/Documents/hypviz/examples/out/scene1_geodesic.html";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 700 } });
const errors = [];
page.on("pageerror", (e) => errors.push(`pageerror: ${e}`));
page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));

await page.goto(url);
await page.waitForTimeout(800);
const nCanvas = await page.locator("canvas").count();
const distLabel = page.locator(".hyplabel", { hasText: /^d = / }).first();
const dist0 = await distLabel.textContent();

// drag point a (poincaré coords 0.45, 0.10) toward (-0.2, -0.3) in the first view
const r = await page.locator("canvas").first().boundingBox();
const hh = 1.15, aspect = r.width / r.height;
const S = (x, y) => [r.x + ((x / (hh * aspect)) * 0.5 + 0.5) * r.width,
                     r.y + ((-y / hh) * 0.5 + 0.5) * r.height];
const [sx, sy] = S(0.45, 0.10), [tx, ty] = S(-0.2, -0.3);
await page.mouse.move(sx, sy);
await page.mouse.down();
for (let i = 1; i <= 10; i++) await page.mouse.move(sx + ((tx - sx) * i) / 10, sy + ((ty - sy) * i) / 10);
await page.mouse.up();
await page.waitForTimeout(300);
const dist1 = await distLabel.textContent();
await browser.close();

const ok = !errors.length && nCanvas === 2 && dist0 !== dist1;
console.log(JSON.stringify({ nCanvas, dist0, dist1, errors }));
console.log(ok ? "SMOKE PASSED" : "SMOKE FAILED");
process.exit(ok ? 0 : 1);
