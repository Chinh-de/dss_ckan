// Capture the demo screenshots used in the report by driving Edge over the DevTools protocol.
// Needs the backend on :8000 and the frontend on :5173.
//   node docs/report/capture_screens.mjs <outDir> <browserProfileDir>
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";

const [outDir, profile] = process.argv.slice(2);
const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const proc = spawn(edge, ["--headless=new", "--remote-debugging-port=9335", "--user-data-dir=" + profile, "--no-first-run", "--hide-scrollbars", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let target;
for (let i = 0; i < 40 && !target; i++) {
  try { target = (await (await fetch("http://127.0.0.1:9335/json")).json()).find((t) => t.type === "page"); } catch {}
  await sleep(250);
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => {
  const my = ++id; pending.set(my, r); ws.send(JSON.stringify({ id: my, method, params }));
  setTimeout(() => { if (pending.has(my)) { pending.delete(my); r({}); } }, 15000);
});
await send("Runtime.enable"); await send("Page.enable"); await send("Network.enable");
// Amazon's cover CDN rejects the default headless user agent.
await send("Network.setUserAgentOverride", { userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0" });

const ev = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.value;
const viewport = (w, h, mobile = false) => send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: mobile ? 2.5 : 1.5, mobile });
const open = async (ls, wait = 4500) => {
  await send("Page.navigate", { url: "http://localhost:5173/" }); await sleep(700);
  await ev(`localStorage.clear(); ${Object.entries(ls).map(([k, v]) => `localStorage.setItem(${JSON.stringify(k)}, ${JSON.stringify(v)})`).join(";")}`);
  await send("Page.reload"); await sleep(wait);
};
const button = (text) => `[...document.querySelectorAll('button')].find(b => b.textContent.trim().includes(${JSON.stringify(text)}))`;
const click = async (text, wait = 1500) => { await ev(`${button(text)}?.click()`); await sleep(wait); };
const type = async (selector, text) => { await ev(`document.querySelector(${JSON.stringify(selector)}).focus()`); await send("Input.insertText", { text }); };
const key = async (k, code) => { for (const t of ["keyDown", "keyUp"]) await send("Input.dispatchKeyEvent", { type: t, key: k, code: k, windowsVirtualKeyCode: code }); };
const shot = async (name, full = false) => {
  // Wait until every cover in the page has either loaded or failed, so no tile is caught half-way.
  await ev("document.querySelectorAll('img').forEach(i => i.loading = 'eager')");
  for (let i = 0; i < 20; i++) {
    if (await ev("[...document.querySelectorAll('img')].every(i => i.complete)")) break;
    await sleep(400);
  }
  await sleep(500);
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: full });
  writeFileSync(`${outDir}/${name}.png`, Buffer.from(r.data, "base64"));
  console.log("screen", name);
};
// The page uses smooth scrolling, so jump instantly or the capture happens mid-animation.
const scrollTo = async (selector) => {
  await ev(`(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 90, behavior: "instant" }); })()`);
  await sleep(900);
};

const DOMAIN = { movie: { ckan_selected_domain: "movie", ckan_current_user_id: "1" }, book: { ckan_selected_domain: "book", ckan_current_user_id: "790" }, music: { ckan_selected_domain: "music", ckan_current_user_id: "774" } };

await viewport(1440, 1500);
await open(DOMAIN.movie, 7000); // warm-up: first load compiles the dev bundle

// 1. Recommendations, one tall capture per dataset so both the spotlight and the ranked grid show.
for (const d of ["movie", "book", "music"]) {
  await viewport(1440, 1500);
  await open(DOMAIN[d], 6000);
  await shot(`rec_${d}`);
}

// 2. Explanation panel (movie: top recommendation; book: a recommendation that has KG paths).
await viewport(1440, 1000);
await open(DOMAIN.movie);
await click("Xem đường dẫn tri thức", 4500);
await shot("explain_movie");
await ev("document.querySelector('[role=dialog] button[title=\"Mở rộng\"]')?.click()"); await sleep(3500);
await shot("explain_movie_wide");
await open(DOMAIN.book);
await ev("[...document.querySelectorAll('article')].find(a => a.innerText.includes('Cùng tác giả'))?.querySelector('button')?.click()"); await sleep(4500);
await shot("explain_book");
await open(DOMAIN.music);
await click("Xem đường dẫn tri thức", 4000);
await shot("explain_music_no_path");

// 3. Knowledge graph tab.
await viewport(1440, 1180);
await open({ ...DOMAIN.movie, ckan_active_tab: "graph" }, 16000);
await shot("graph_movie");
await click("Đạo diễn", 4500);
await shot("graph_movie_director");
await open({ ...DOMAIN.book, ckan_active_tab: "graph" }, 9000);
await shot("graph_book");

// 4. Sparsity tab: measured results, then the live comparison.
for (const d of ["movie", "book", "music"]) {
  await viewport(1440, 1010);
  await open({ ...DOMAIN[d], ckan_active_tab: "coldstart" }, 5000);
  await shot(`sparsity_${d}`);
  if (d === "book") {
    await scrollTo("section[aria-labelledby=sparsity-table]");
    await shot("sparsity_book_tables");
    await scrollTo("section[aria-labelledby=live-title]");
    await shot("sparsity_book_live");
  }
}

// 5. Catalogue and search.
await viewport(1440, 1000);
await open({ ...DOMAIN.movie, ckan_active_tab: "explore" }, 6000);
await shot("catalog_movie");
await open({ ...DOMAIN.book, ckan_active_tab: "explore" }, 5000);
await type("input[type=search]", "tolkien");
await ev("document.querySelector('form[role=search]').requestSubmit()"); await sleep(3500);
await ev(`${button("Thích")}.click()`); await sleep(900);
await shot("catalog_book_search");

// 6. User dialog, and a brand-new user in the music dataset.
await viewport(1440, 900);
await open(DOMAIN.movie);
await click("Đổi người dùng", 2500);
await shot("users_movie");
await open(DOMAIN.music);
await click("Đổi người dùng", 2000);
await click("Người dùng mới", 800);
await shot("users_music_new");
await click("Tạo và dùng ngay", 5000);
await viewport(1440, 1300);
await sleep(1500);
await shot("rec_music_new_user");

// 7. Phone layout.
await viewport(390, 844, true);
await open(DOMAIN.movie, 5000);
await shot("mobile_rec");
await open({ ...DOMAIN.music, ckan_active_tab: "coldstart" }, 4500);
await shot("mobile_sparsity");

ws.close();
proc.kill();
