/** Load every generated gallery page headlessly; fail on any JS error. */
import { readdirSync } from "node:fs";
import { chromium } from "playwright";

const dir = "/Users/longteng/Documents/hypviz/examples/out";
const browser = await chromium.launch();
let failed = false;
for (const f of readdirSync(dir).filter((f) => f.endsWith(".html"))) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 800 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(`file://${dir}/${f}`);
  await page.waitForTimeout(700);
  const canvases = await page.locator("canvas").count();
  console.log(`${errors.length || !canvases ? "FAIL" : "ok  "} ${f} — ${canvases} canvases ${errors.join("; ")}`);
  failed ||= errors.length > 0 || !canvases;
  await page.close();
}
await browser.close();
process.exit(failed ? 1 : 0);
