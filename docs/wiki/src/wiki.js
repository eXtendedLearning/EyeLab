/* EyeLab wiki runtime: language, routing, navigation, search, maths, figures. No dependencies
   except KaTeX (optional; raw TeX is shown if it fails to load). */
(function () {
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const root = document.documentElement;
const lang = () => (root.getAttribute("data-lang") === "it" ? "it" : "en");
const T = (en, it) => (lang() === "it" ? it : en);
const fmt = (x, d = 2) => { const s = x.toFixed(d); return lang() === "it" ? s.replace(".", ",") : s; };

const GROUPS = [
  ["overview", "Overview", "Panoramica"],
  ["system", "System", "Sistema"],
  ["validation", "Validation", "Validazione"],
  ["xreal", "XREAL and platforms", "XREAL e piattaforme"],
  ["decisions", "Decisions and roadmap", "Decisioni e roadmap"],
];
const pages = $$("article.page");
const byId = Object.fromEntries(pages.map(p => [p.id.slice(2), p]));

/* ---------- navigation ---------- */
const nav = $("#nav");
function buildNav() {
  const host = $("#nav-groups");
  host.innerHTML = "";
  GROUPS.forEach(([g, en, it]) => {
    const list = pages.filter(p => p.dataset.group === g);
    if (!list.length) return;
    const div = document.createElement("div");
    div.className = "grp";
    div.innerHTML = `<h4>${T(en, it)}</h4><ul>${list.map(p =>
      `<li><a href="#${p.id.slice(2)}" data-k="${p.id.slice(2)}">${p.dataset[lang()]}</a></li>`).join("")}</ul>`;
    host.appendChild(div);
  });
  markCurrent();
}
function markCurrent() {
  const k = current();
  $$("#nav a[data-k]").forEach(a => { if (a.dataset.k === k) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
}
function current() {
  const h = (location.hash || "").slice(1);
  return byId[h] ? h : "home";
}
function pager(k) {
  const order = GROUPS.flatMap(([g]) => pages.filter(p => p.dataset.group === g));
  const i = order.findIndex(p => p.id === "p-" + k);
  const prev = order[i - 1], next = order[i + 1];
  const host = $("#pager");
  host.innerHTML = (prev ? `<a href="#${prev.id.slice(2)}"><span>${T("Previous", "Precedente")}</span><b>${prev.dataset[lang()]}</b></a>` : "") +
    (next ? `<a class="nx" href="#${next.id.slice(2)}"><span>${T("Next", "Successiva")}</span><b>${next.dataset[lang()]}</b></a>` : "");
}
function show(scrollTop = true) {
  const k = current();
  pages.forEach(p => { p.hidden = p.id !== "p-" + k; });
  markCurrent();
  pager(k);
  document.title = (k === "home" ? "" : byId[k].dataset[lang()] + " · ") + "EyeLab Wiki";
  if (scrollTop) window.scrollTo(0, 0);
  closeMenu();
}
window.addEventListener("hashchange", () => show(true));

/* mobile drawer */
const menuBtn = $("#menu-btn");
function closeMenu() { if (window.matchMedia("(max-width:980px)").matches) { nav.hidden = true; menuBtn.setAttribute("aria-expanded", "false"); } }
menuBtn.addEventListener("click", () => {
  const open = nav.hidden;
  nav.hidden = !open;
  menuBtn.setAttribute("aria-expanded", String(open));
  if (open) { const s = $("#q"); try { s.focus({ preventScroll: true }); } catch (e) {} }
});
function syncDrawer() { if (!window.matchMedia("(max-width:980px)").matches) nav.hidden = false; else nav.hidden = true; }
window.addEventListener("resize", () => { if (!window.matchMedia("(max-width:980px)").matches) nav.hidden = false; });

/* search over the visible-language text of every page */
const q = $("#q"), hits = $("#hits");
function textOf(p) {
  const clone = p.cloneNode(true);
  $$(lang() === "it" ? ".l-en" : ".l-it", clone).forEach(e => e.remove());
  return clone.textContent.replace(/\s+/g, " ").toLowerCase();
}
q.addEventListener("input", () => {
  const term = q.value.trim().toLowerCase();
  if (term.length < 2) { hits.hidden = true; hits.innerHTML = ""; return; }
  const found = pages.map(p => [p, textOf(p).split(term).length - 1]).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  hits.hidden = false;
  hits.innerHTML = found.length
    ? `<div>${T("Pages containing", "Pagine che contengono")} “${q.value.trim().replace(/[<>&]/g, "")}”</div><ul>` +
      found.map(([p, n]) => `<li><a href="#${p.id.slice(2)}">${p.dataset[lang()]} <span class="small">· ${n}</span></a></li>`).join("") + "</ul>"
    : `<div>${T("No page contains this term.", "Nessuna pagina contiene questo termine.")}</div>`;
});

/* ---------- language ---------- */
function setLang(l) {
  root.setAttribute("data-lang", l);
  root.lang = l;
  try { localStorage.setItem("eyelab-wiki-lang", l); } catch (e) {}
  $$(".seg button[data-lang]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
  numbers();
  buildNav();
  show(false);
  drawAll();
  if (q.value) q.dispatchEvent(new Event("input"));
}
$$(".seg button[data-lang]").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));

/* decimal separator in shared numeric tables and .nf spans */
function numbers() {
  $$("table.data td, table.data th, .nf").forEach(el => {
    if (el.dataset.raw == null) el.dataset.raw = el.innerHTML;
    el.innerHTML = lang() === "it" ? el.dataset.raw.replace(/(\d)\.(\d)/g, "$1,$2") : el.dataset.raw;
  });
}

/* ---------- maths ---------- */
function maths() {
  const ok = typeof window.katex !== "undefined";
  $$("[data-tex]").forEach(el => {
    const tex = el.getAttribute("data-tex");
    const display = el.classList.contains("tex");
    if (ok) {
      try { window.katex.render(tex, el, { displayMode: display, output: "mathml", throwOnError: false }); return; } catch (e) {}
    }
    if (display) { el.textContent = tex; el.classList.add("raw"); }
  });
}

/* ---------- figures ---------- */
const NS = "http://www.w3.org/2000/svg";
const mk = (p, tag, attrs, txt) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (txt != null) e.textContent = txt; p.appendChild(e); return e; };
const tip = $("#tip");
function hover(el, fn) {
  el.addEventListener("pointerenter", () => { tip.textContent = fn(); tip.hidden = false; });
  el.addEventListener("pointermove", e => { tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; });
  el.addEventListener("pointerleave", () => { tip.hidden = true; });
}
const clear = s => { while (s.firstChild) s.removeChild(s.firstChild); };

/* Campaign 1 data (Surface webcam, 15–16 Oct 2026). Columns: test, d, angle, RMSE, sd, p95, sd, max node bias, RMS bias, RMS jitter, scale permil */
const COND = [["B1",400,0,0.377,0.042,0.691,0.147,0.786,0.367,0.057,-3.982],["B1",400,30,0.507,0.03,0.785,0.115,0.884,0.507,0.014,-3.454],["B1",400,50,1.02,0.085,1.496,0.105,1.72,1.019,0.016,-4.693],["B1",600,0,0.288,0.046,0.516,0.073,0.601,0.288,0.02,-2.53],["B1",600,30,0.485,0.076,0.767,0.125,0.818,0.482,0.052,-1.924],["B1",600,50,1.093,0.175,1.665,0.183,1.767,1.091,0.073,-2.918],["B1",900,0,0.886,0.582,1.546,0.841,1.223,0.799,0.363,-2.114],["B1",900,30,0.718,0.091,1.14,0.182,1.176,0.703,0.141,-2.535],["B1",900,50,1.009,0.307,1.514,0.523,1.507,0.98,0.219,-2.184]];
const ROT = {"400,0":0.030,"400,30":0.037,"400,50":0.019,"600,0":0.094,"600,30":0.180,"600,50":0.098,"900,0":4.335,"900,30":0.416,"900,50":0.201};
const NODES = [
 ["600 mm · 0°","600 mm · 0°",[[-79.971,60.056,0.076,-0.083],[-39.985,60.056,-0.046,-0.066],[0,60.056,-0.152,-0.049],[39.985,60.056,-0.242,-0.032],[79.971,60.056,-0.316,-0.015],[-79.971,20.019,0.05,-0.034],[-39.985,20.019,-0.094,-0.025],[0,20.019,-0.223,-0.016],[39.985,20.019,-0.336,-0.007],[79.971,20.019,-0.434,0.002],[-79.971,-20.019,0.025,0.062],[-39.985,-20.019,-0.142,0.063],[0,-20.019,-0.294,0.064],[39.985,-20.019,-0.431,0.065],[79.971,-20.019,-0.551,0.066],[-79.971,-60.056,0,0.203],[-39.985,-60.056,-0.19,0.196],[0,-60.056,-0.366,0.19],[39.985,-60.056,-0.525,0.183],[79.971,-60.056,-0.669,0.176]]],
 ["600 mm · 50°, side + (rep. 1)","600 mm · 50°, lato + (rip. 1)",[[-79.971,60.056,-0.742,0.061],[-39.985,60.056,-0.771,0.081],[0,60.056,-0.782,0.102],[39.985,60.056,-0.776,0.123],[79.971,60.056,-0.752,0.143],[-79.971,20.019,-0.679,-0.034],[-39.985,20.019,-0.803,-0.022],[0,20.019,-0.91,-0.01],[39.985,20.019,-1,0.002],[79.971,20.019,-1.073,0.014],[-79.971,-20.019,-0.615,0.064],[-39.985,-20.019,-0.835,0.068],[0,-20.019,-1.038,0.071],[39.985,-20.019,-1.224,0.074],[79.971,-20.019,-1.392,0.078],[-79.971,-60.056,-0.553,0.353],[-39.985,-60.056,-0.867,0.348],[0,-60.056,-1.165,0.343],[39.985,-60.056,-1.446,0.338],[79.971,-60.056,-1.71,0.332]]],
 ["600 mm · 50°, side − (rep. 2)","600 mm · 50°, lato − (rip. 2)",[[-79.971,60.056,1.015,-0.093],[-39.985,60.056,0.974,-0.089],[0,60.056,0.847,-0.085],[39.985,60.056,0.631,-0.082],[79.971,60.056,0.329,-0.078],[-79.971,20.019,1.126,0.201],[-39.985,20.019,1.165,0.248],[0,20.019,1.115,0.296],[39.985,20.019,0.977,0.344],[79.971,20.019,0.752,0.391],[-79.971,-20.019,1.239,0.338],[-39.985,-20.019,1.356,0.43],[0,-20.019,1.384,0.521],[39.985,-20.019,1.325,0.613],[79.971,-20.019,1.178,0.704],[-79.971,-60.056,1.351,0.317],[-39.985,-60.056,1.548,0.453],[0,-60.056,1.655,0.589],[39.985,-60.056,1.674,0.725],[79.971,-60.056,1.605,0.861]]]];
const WHATIF = {"400,0":{A:[.378,.672],B:[.365,.652],D:[.205,.424],E:[.205,.436],C:[.081,.120]},"400,30":{A:[.527,.796],B:[.457,.744],D:[.328,.471],E:[.246,.484],C:[.076,.132]},"400,50":{A:[1.034,1.500],B:[.981,1.451],D:[.905,1.256],E:[.722,1.350],C:[.165,.296]},"600,0":{A:[.296,.556],B:[.269,.519],D:[.167,.352],E:[.179,.362],C:[.073,.110]},"600,30":{A:[.567,.906],B:[.510,.811],D:[.488,.733],E:[.393,.751],C:[.107,.210]},"600,50":{A:[1.185,1.802],B:[1.134,1.710],D:[1.101,1.685],E:[.914,1.787],C:[.188,.365]},"900,0":{A:[.287,.506],B:[.266,.452],D:[.188,.313],E:[.191,.320],C:[.058,.101]},"900,30":{A:[.655,1.133],B:[.611,1.039],D:[.604,.956],E:[.511,.997],C:[.115,.212]},"900,50":{A:[1.180,1.806],B:[1.140,1.729],D:[1.156,1.712],E:[.954,1.796],C:[.197,.381]}};
const U = 0.156;
const cls = v => v < 1 - U ? "g" : v <= 1 + U ? "w" : "b";
const verdict = v => ({ g: '<span class="pill good">ok</span>', w: `<span class="pill warn">${T("marginal", "limite")}</span>`, b: '<span class="pill bad">no</span>' })[cls(v)];
const td = (v, c = "") => `<td${c ? ` class="${c}"` : ""}>${v}</td>`;
function tbl(id, head, rows, extraHead) {
  const el = document.getElementById(id); if (!el) return;
  el.innerHTML = "<thead>" + (extraHead || "") + "<tr>" + head.map(h => `<th${h.startsWith("|") ? ' class="sep"' : ""}>${h.replace("|", "")}</th>`).join("") + "</tr></thead><tbody>" + rows.map(r => "<tr>" + r.join("") + "</tr>").join("") + "</tbody>";
}

function tables() {
  tbl("t-b1", [T("Distance", "Distanza"), T("Angle", "Angolo"), "RMSE mm", "p95 mm", T("|Max node bias", "|Bias max nodo"), T("RMS bias", "RMS bias"), "RMS jitter", T("Scale ‰", "Scala ‰"), T("|Rot. jitter p95", "|Jitter rot. p95"), T("Verdict", "Giudizio")],
    COND.map(c => [td(c[1] + " mm"), td(c[2] + "°"), td(fmt(c[3]) + " ± " + fmt(c[4])), td(fmt(c[5]) + " ± " + fmt(c[6]), cls(c[5])), td(fmt(c[7]), "sep " + cls(c[7])), td(fmt(c[8])), td(fmt(c[9])), td(fmt(c[10], 1)), td(fmt(ROT[c[1] + "," + c[2]]) + "°", "sep" + (ROT[c[1] + "," + c[2]] > 1 ? " b" : "")), `<td>${verdict(c[5])}</td>`]));
  tbl("t-dz", [T("Condition", "Condizione"), "EyeLab", T("Marker PnP", "PnP marker")],
    [["400 · 0°",1.97,2.48],["400 · 30°",1.17,1.07],["400 · 50°",1.25,1.36],["600 · 0°",2.12,2.63],["600 · 30°",1.02,2.32],["600 · 50°",0.51,1.06],["900 · 0°",2.88,0.76],["900 · 30°",1.99,2.40],["900 · 50°",1.34,2.03]].map(r => [td(r[0]), td("+" + fmt(r[1])), td("+" + fmt(r[2]))]));
  tbl("t-size", [T("Distance", "Distanza"), T("Edge px (0°)", "Lato px (0°)"), "0°", "30°", "50°", T("≈ px per side", "≈ px per lato")],
    [[400,54,-14.9,-14.2,-19.2],[600,36,-19.7,-22.0,-24.4],[900,24,-30.1,-27.9,-23.0]].map(r => [td(r[0] + " mm"), td("≈" + r[1]), td(fmt(r[2], 1)), td(fmt(r[3], 1)), td(fmt(r[4], 1)), td(fmt(-r[1] * (r[2] + r[3] + r[4]) / 3000, 2))]));
  tbl("t-dn", ["Marker", T("Position (mm)", "Posizione (mm)"), "30° · 400", "600", "900", "|50° · 400", "600", "900"],
    [[30,"(−118, +70)",-0.88,-0.96,-1.13,-0.82,-0.69,-0.89],[32,"(−118, −70)",-0.29,-0.38,-0.47,-0.43,-0.40,-0.51],[31,"(+118, +70)",-0.07,0.04,-0.05,-0.30,-0.27,-0.11]].map(r => [td(r[0]), td(r[1]), ...r.slice(2).map((v, i) => td(fmt(v), i == 3 ? "sep" : ""))]));
  const V = [["A", T("A · as recorded", "A · come registrato")], ["B", T("B · corner correction", "B · correzione spigoli")], ["D", T("D · markers re-measured", "D · marker rimisurati")], ["E", T("E · D + out-of-plane", "E · D + fuori piano")], ["C", T("C · saddle pose (bound)", "C · posa dai punti sella (limite)")]];
  tbl("t-whatif", [T("Condition", "Condizione"), ...V.map(v => "|" + v[1])],
    Object.entries(WHATIF).map(([k, o]) => { const [d, a] = k.split(","); return [td(d + " · " + a + "°"), ...V.map(([key]) => td(fmt(o[key][0]) + " / " + fmt(o[key][1]), "sep " + cls(o[key][1])))]; }));
  tbl("t-flip", [T("Session", "Sessione"), "EyeLab", "SQPnP + LM", T("Best IPPE", "IPPE migliore")],
    [[T("900 · 0° · rep. 1", "900 · 0° · rip. 1"),6.63,0.65,2.74],[T("900 · 0° · rep. 3", "900 · 0° · rip. 3"),6.73,0.55,3.67],[T("600 · 0° · rep. 1 (ref.)", "600 · 0° · rip. 1 (rif.)"),0.07,0.05,0.37]].map(r => [td(r[0]), td(fmt(r[1]) + "°", r[1] > 1 ? "b" : ""), td(fmt(r[2]) + "°"), td(fmt(r[3]) + "°")]));
  const rp = T("rep.", "rip.");
  tbl("t-b1b", [T("Session", "Sessione"), T("Median", "Mediana"), "p95", "|0–60 mm", "60–120", "120–180", "180–260", T("|Rot. p95", "|Rot. p95"), T("Transl. p95", "Trasl. p95"), T("|Points with truth", "|Punti con verità"), T("H residual px", "Residuo H px")],
    [["0° · " + rp + " 1",1.16,1.74,.47,1.01,1.32,1.24,1.74,4.89,100,0.12],["0° · " + rp + " 2",4.44,6.88,.98,2.70,4.50,6.15,1.76,2.41,100,0.10],["0° · " + rp + " 3",2.49,3.57,.67,1.66,2.55,3.21,.96,2.77,100,0.11],["30° · " + rp + " 1",5.26,8.27,1.34,3.40,5.33,7.32,3.19,20.26,100,0.24],["30° · " + rp + " 2",8.30,13.32,2.43,5.39,8.30,11.70,3.78,19.52,100,0.25],["30° · " + rp + " 3",1.03,1.72,.54,.77,1.17,1.32,.80,1.27,100,0.23]]
      .map(r => [td(r[0]), td(fmt(r[1]), cls(r[1])), td(fmt(r[2]), cls(r[2])), ...r.slice(3, 7).map((v, i) => td(fmt(v), (i == 0 ? "sep " : "") + cls(v))), td(fmt(r[7]) + "°", "sep"), td(fmt(r[8]) + " mm"), td(r[9] + " %", "sep"), td(fmt(r[10]))]),
    `<tr><th></th><th colspan="2">${T("Error, mm", "Errore, mm")}</th><th colspan="4" class="sep">${T("Median error by distance from marker 30", "Errore mediano per distanza dal marker 30")}</th><th colspan="2" class="sep">${T("EyeLab pose spread", "Dispersione posa EyeLab")}</th><th colspan="2" class="sep">${T("Ground truth", "Verità a terra")}</th></tr>`);
  tbl("t-gt", [T("Check", "Controllo"), T("Points", "Punti"), T("Corrected", "Corretti"), T("Failed", "Falliti"), T("95 % bound", "Limite 95 %")],
    [[T("B1 · random sample, automatic truth", "B1 · campione casuale, verità automatica"), "300", T("1 (missing)", "1 (assente)"), "1 / 300", fmt(1.6, 1) + " %"],
     [T("B1 · detected points only", "B1 · solo punti trovati"), "299", "0", "0 / 299", fmt(1.0, 1) + " %"],
     [T("B1b · shown points, automatic truth", "B1b · punti mostrati, verità automatica"), "121", "65", "65 / 121", "—"],
     [T("B1b · random sample, reviewed truth", "B1b · campione casuale, verità rivista"), "120", "0", "0 / 120", fmt(2.5, 1) + " %"]]
      .map(r => r.map((v, i) => td(v, i == 3 && String(v).startsWith("65") ? "b" : ""))));
}

function plateA() {
  const s = document.getElementById("plateA"); if (!s) return; clear(s);
  const W = 297, H = 210, cx = W / 2, cy = H / 2, q = 15;
  mk(s, "rect", { x: 0, y: 0, width: W, height: H, fill: "var(--paper)", stroke: "var(--muted)", "stroke-width": .8 });
  for (let j = 0; j < 4; j++) for (let i = 0; i < 5; i++) { const x = cx + (i - 2) * 40, y = cy + (j - 1.5) * 40;
    mk(s, "rect", { x: x - q, y: y - q, width: q, height: q, fill: "var(--ink)" }); mk(s, "rect", { x: x, y: y, width: q, height: q, fill: "var(--ink)" });
    mk(s, "circle", { cx: x, cy: y, r: 2.2, fill: "none", stroke: "var(--s3)", "stroke-width": 1 }); }
  [[30, cx - 118, cy - 70], [31, cx + 118, cy - 70], [32, cx - 118, cy + 70]].forEach(([id, x, y]) => {
    mk(s, "rect", { x: x - 10, y: y - 10, width: 20, height: 20, fill: "none", stroke: "var(--s1)", "stroke-width": 1.6 });
    mk(s, "rect", { x: x - 6, y: y - 6, width: 12, height: 12, fill: "var(--ink)" });
    mk(s, "text", { x: x, y: y + (y < cy ? -13 : 19), "text-anchor": "middle", "font-size": 8, fill: "var(--s1)" }, id); });
  mk(s, "line", { x1: cx - 100, x2: cx + 100, y1: 196, y2: 196, stroke: "var(--s2)", "stroke-width": 1.4 });
  mk(s, "text", { x: cx, y: 206, "text-anchor": "middle", "font-size": 7.5 }, T("X bar 200", "barra X 200"));
  mk(s, "line", { x1: 14, x2: 14, y1: cy - 75, y2: cy + 75, stroke: "var(--s2)", "stroke-width": 1.4 });
  mk(s, "text", { x: 6, y: cy, "text-anchor": "middle", "font-size": 7.5, transform: `rotate(-90 6 ${cy})` }, T("Y bar 150", "barra Y 150"));
  const dim = (x1, y1, x2, y2, t, ty) => { mk(s, "line", { x1, y1, x2, y2, stroke: "var(--muted)", "stroke-width": .6, "stroke-dasharray": "2 1.5" }); mk(s, "text", { x: (x1 + x2) / 2, y: ty, "text-anchor": "middle", "font-size": 7 }, t); };
  dim(cx - 118, -6, cx + 118, -6, T("236 (marker centres 30–31)", "236 (centri marker 30–31)"), -8);
  dim(cx - 80, cy + 73, cx - 40, cy + 73, T("pitch 40", "passo 40"), cy + 81);
  mk(s, "text", { x: cx + 118, y: cy - 48, "text-anchor": "middle", "font-size": 7 }, "□ 20");
  mk(s, "text", { x: W - 4, y: H + 11, "text-anchor": "end", "font-size": 7 }, "297 × 210");
}
function geom() {
  const s = document.getElementById("geom"); if (!s) return; clear(s);
  const cxp = 280, cyp = 110, th = 30 * Math.PI / 180, L = 70;
  mk(s, "rect", { x: 22, y: 96, width: 34, height: 28, rx: 3, fill: "none", stroke: "var(--fg)", "stroke-width": 1.5 });
  mk(s, "path", { d: "M56,104 L66,98 L66,122 L56,116 Z", fill: "var(--fg)" });
  mk(s, "text", { x: 39, y: 142, "text-anchor": "middle", "font-size": 10 }, "camera");
  const ux = Math.sin(th), uy = Math.cos(th);
  mk(s, "line", { x1: cxp - L * ux, y1: cyp - L * uy, x2: cxp + L * ux, y2: cyp + L * uy, stroke: "var(--s1)", "stroke-width": 5, "stroke-linecap": "round" });
  mk(s, "text", { x: cxp + L * ux + 6, y: cyp + L * uy + 4, "font-size": 10, fill: "var(--s1)" }, "plate");
  mk(s, "line", { x1: 66, y1: 110, x2: cxp, y2: cyp, stroke: "var(--fg2)", "stroke-width": 1, "stroke-dasharray": "5 4" });
  mk(s, "text", { x: (66 + cxp) / 2, y: 102, "text-anchor": "middle", "font-size": 12, fill: "var(--fg)" }, "d");
  const nx = -Math.cos(th), ny = Math.sin(th);
  mk(s, "line", { x1: cxp, y1: cyp, x2: cxp + nx * 90, y2: cyp + ny * 90, stroke: "var(--s2)", "stroke-width": 1.4 });
  mk(s, "text", { x: cxp + nx * 96 - 4, y: cyp + ny * 96 + 10, "text-anchor": "end", "font-size": 10, fill: "var(--s2)" }, T("normal", "normale"));
  const r = 48, a0 = Math.PI, a1 = Math.PI - th;
  mk(s, "path", { d: `M${cxp + r * Math.cos(a0)},${cyp + r * Math.sin(a0)} A${r},${r} 0 0 0 ${cxp + r * Math.cos(a1)},${cyp + r * Math.sin(a1)}`, fill: "none", stroke: "var(--fg)", "stroke-width": 1 });
  mk(s, "text", { x: cxp - r - 10, y: cyp + 20, "text-anchor": "middle", "font-size": 12, fill: "var(--fg)" }, "θ");
  mk(s, "text", { x: 180, y: 206, "text-anchor": "middle", "font-size": 10 }, "d = 400 · 600 · 900 mm   θ = 0° · 30° · 50°");
}
function chartAngle() {
  const s = document.getElementById("c-angle"); if (!s) return; clear(s);
  const L = 48, R = 420, Tp = 14, B = 212, ymax = 2;
  const x = a => L + (R - L) * (a / 50), y = v => B - (B - Tp) * v / ymax;
  for (let v = 0; v <= 2; v += 0.5) { mk(s, "line", { x1: L, x2: R, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }); mk(s, "text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 11 }, fmt(v, 1)); }
  [0, 30, 50].forEach(a => mk(s, "text", { x: x(a), y: B + 18, "text-anchor": "middle", "font-size": 11 }, a + "°"));
  mk(s, "text", { x: (L + R) / 2, y: B + 36, "text-anchor": "middle", "font-size": 11 }, T("angle to the plate normal", "angolo rispetto alla normale"));
  mk(s, "text", { x: 12, y: (Tp + B) / 2, "text-anchor": "middle", "font-size": 11, transform: `rotate(-90 12 ${(Tp + B) / 2})` }, "p95 mm");
  mk(s, "line", { x1: L, x2: R, y1: y(1), y2: y(1), stroke: "var(--bad)", "stroke-width": 1, "stroke-dasharray": "4 3" });
  [[400, "var(--s1)"], [600, "var(--s2)"], [900, "var(--s3)"]].forEach(([d, c]) => {
    const pts = COND.filter(r => r[1] == d);
    mk(s, "polyline", { points: pts.map(r => x(r[2]) + "," + y(r[5])).join(" "), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round" });
    pts.forEach(r => { const g = mk(s, "g", {}); mk(g, "circle", { cx: x(r[2]), cy: y(r[5]), r: 4.5, fill: c, stroke: "var(--paper)", "stroke-width": 2 }); mk(g, "circle", { cx: x(r[2]), cy: y(r[5]), r: 12, fill: "transparent" }); hover(g, () => `${d} mm · ${r[2]}°  p95 ${fmt(r[5])} ± ${fmt(r[6])} mm`); });
    const last = pts[2]; mk(s, "text", { x: x(50) + 6, y: y(last[5]) + (d == 600 ? -6 : d == 900 ? 12 : 4), "font-size": 11, fill: c }, d);
  });
}
function chartSplit() {
  const s = document.getElementById("c-split"); if (!s) return; clear(s);
  const L = 40, R = 430, Tp = 14, B = 200, ymax = 1.2, n = COND.length, bw = (R - L) / n;
  const y = v => B - (B - Tp) * v / ymax;
  for (let v = 0; v <= 1.2001; v += 0.4) { mk(s, "line", { x1: L, x2: R, y1: y(v), y2: y(v), stroke: "var(--rule)", "stroke-width": 1 }); mk(s, "text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 11 }, fmt(v, 1)); }
  COND.forEach((c, i) => { const x0 = L + i * bw + bw * 0.18, w = bw * 0.3;
    [[c[8], "var(--s1)", T("bias", "bias")], [c[9], "var(--s2)", "jitter"]].forEach(([v, col, lab], j) => { const h = B - y(v), xx = x0 + j * (w + 2);
      const g = mk(s, "g", {}); mk(g, "path", { d: `M${xx},${B} v${-Math.max(h - 3, 0)} q0,-3 3,-3 h${w - 6} q3,0 3,3 v${Math.max(h - 3, 0)} z`, fill: col }); mk(g, "rect", { x: xx - 2, y: Tp, width: w + 4, height: B - Tp, fill: "transparent" }); hover(g, () => `${c[1]} mm · ${c[2]}°  RMS ${lab} ${fmt(v, 3)} mm`); });
    mk(s, "text", { x: L + i * bw + bw / 2, y: B + 15, "text-anchor": "middle", "font-size": 10 }, c[2] + "°"); });
  [400, 600, 900].forEach((d, k) => { mk(s, "text", { x: L + (k * 3 + 1.5) * bw, y: B + 32, "text-anchor": "middle", "font-size": 11, fill: "var(--fg)" }, d + " mm"); if (k) mk(s, "line", { x1: L + k * 3 * bw, x2: L + k * 3 * bw, y1: B, y2: B + 36, stroke: "var(--rule)" }); });
  mk(s, "line", { x1: L, x2: R, y1: B, y2: B, stroke: "var(--muted)", "stroke-width": 1 });
}
function quivers() {
  const host = document.getElementById("quivers"); if (!host) return; host.innerHTML = "";
  const K = 40;
  NODES.forEach(([en, it, pts], idx) => {
    const fig = document.createElement("div"); fig.className = "fig"; fig.innerHTML = `<h3>${T(en, it)}</h3>`;
    const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", "-150 -100 300 210"); s.setAttribute("role", "img"); s.setAttribute("aria-label", T("bias field ", "campo di bias ") + T(en, it)); fig.appendChild(s); host.appendChild(fig);
    mk(s, "rect", { x: -140, y: -90, width: 280, height: 180, fill: "none", stroke: "var(--rule)", "stroke-dasharray": "3 3" });
    [[30, -118, 70], [31, 118, 70], [32, -118, -70]].forEach(([id, u, v]) => { mk(s, "rect", { x: u - 10, y: -v - 10, width: 20, height: 20, fill: "var(--fg2)", opacity: .8 }); mk(s, "text", { x: u, y: -v + (v > 0 ? -14 : 24), "text-anchor": "middle", "font-size": 9 }, id); });
    mk(s, "defs", {}).innerHTML = '<marker id="ah' + idx + '" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="var(--s1)"/></marker>';
    pts.forEach(([u, v, du, dv]) => { const g = mk(s, "g", {}); mk(g, "circle", { cx: u, cy: -v, r: 2.2, fill: "var(--fg)" });
      mk(g, "line", { x1: u, y1: -v, x2: u + du * K, y2: -(v + dv * K), stroke: "var(--s1)", "stroke-width": 1.8, "marker-end": `url(#ah${idx})` });
      mk(g, "circle", { cx: u, cy: -v, r: 9, fill: "transparent" }); hover(g, () => `${T("node", "nodo")} (${fmt(u, 0)}, ${fmt(v, 0)})  bias ${fmt(Math.hypot(du, dv))} mm`); });
    mk(s, "line", { x1: -140, x2: -100, y1: 102, y2: 102, stroke: "var(--s1)", "stroke-width": 1.8 }); mk(s, "text", { x: -95, y: 105, "font-size": 9 }, "1 mm (×40)");
    mk(s, "text", { x: 140, y: 105, "text-anchor": "end", "font-size": 9 }, T("plate face, mm", "faccia del plate, mm"));
  });
}
function lockDiagram() {
  const s = document.getElementById("d-lock"); if (!s) return; clear(s);
  mk(s, "defs", {}).innerHTML = '<marker id="la" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--fg2)"/></marker>';
  const box = (x, y, w, name, sub, col) => { mk(s, "rect", { x, y, width: w, height: 58, rx: 4, fill: "var(--paper)", stroke: col, "stroke-width": 2 }); mk(s, "text", { x: x + w / 2, y: y + 25, "text-anchor": "middle", "font-size": 13, fill: "var(--fg)", "font-weight": 600 }, name); mk(s, "text", { x: x + w / 2, y: y + 44, "text-anchor": "middle", "font-size": 10 }, sub); };
  box(20, 86, 170, "SEARCHING", T("no pose drawn", "nessuna posa"), "var(--muted)");
  box(265, 86, 170, "LOCKED", T("measure + Kalman", "misura + Kalman"), "var(--good)");
  box(510, 86, 170, "COASTING", T("predict only ≤ 0.3 s", "solo predizione ≤ 0,3 s").replace("0.3", lang() === "it" ? "0,3" : "0.3"), "var(--warn)");
  const arrow = (d, label, lx, ly) => { mk(s, "path", { d, fill: "none", stroke: "var(--fg2)", "stroke-width": 1.4, "marker-end": "url(#la)" }); mk(s, "text", { x: lx, y: ly, "text-anchor": "middle", "font-size": 10 }, label); };
  arrow("M190,104 L263,104", T("≥3 or 2×5 consistent", "≥3 o 2×5 coerenti"), 227, 96);
  arrow("M435,104 L508,104", T("no valid pose", "nessuna posa valida"), 472, 96);
  arrow("M508,128 L437,128", T("pose passes gates", "posa supera i gate"), 472, 144);
  arrow("M595,86 C595,30 105,30 105,84", T("coast window expired → reset filter", "finestra scaduta → reset del filtro"), 350, 40);
  mk(s, "text", { x: 350, y: 182, "text-anchor": "middle", "font-size": 10 }, T("acquisition: finite, depth > 0, RMS ≤ 15 px, not ambiguous", "aggancio: finito, profondità > 0, RMS ≤ 15 px, non ambiguo"));
  mk(s, "text", { x: 350, y: 200, "text-anchor": "middle", "font-size": 10 }, T("tracking: adaptive RMS gate, motion gate max(floor, v·Δt), ≥ 1 marker", "tracciamento: gate RMS adattivo, gate di movimento max(soglia, v·Δt), ≥ 1 marker"));
}
function drawAll() { tables(); plateA(); geom(); chartAngle(); chartSplit(); quivers(); lockDiagram(); }

/* ---------- start ---------- */
$$(".seg button[data-lang]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.lang === lang())));
syncDrawer();
numbers();
maths();
buildNav();
drawAll();
show(false);
})();
