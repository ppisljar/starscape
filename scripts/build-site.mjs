/* Assembles the GitHub Pages site into `site-dist/`:
 *   site-dist/index.html      the landing page (copied from site/)
 *   site-dist/docs/*.html     README/DESIGN/SCALE rendered to styled HTML (this script)
 *   site-dist/playground/     the vite build of demo/   (npm run build:playground)
 *   site-dist/api/            the typedoc API reference  (npm run build:api)
 *   site-dist/.nojekyll       so Pages serves typedoc's _-prefixed asset dirs
 *
 * This script writes only index.html + docs/ + .nojekyll; it never wipes site-dist, so it
 * composes with the vite and typedoc steps regardless of order. Run via `npm run build:site`.
 */
import { readFileSync, writeFileSync, mkdirSync, cpSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'site-dist');
const docsOut = join(out, 'docs');
mkdirSync(docsOut, { recursive: true });

writeFileSync(join(out, '.nojekyll'), '');
cpSync(join(root, 'site', 'index.html'), join(out, 'index.html'));

marked.setOptions({ gfm: true });

const DOCS = [
  { md: 'README.md', html: 'readme.html', title: 'Guide', blurb: 'What it is and how to use it.' },
  { md: 'DESIGN.md', html: 'design.html', title: 'Design', blurb: 'The contract everything else follows.' },
  { md: 'SCALE.md', html: 'scale.html', title: 'Performance', blurb: 'Measured numbers and the methodology behind them.' },
];

const CSS = `
:root{--bg:#05060d;--panel:rgba(18,22,44,.5);--border:rgba(120,140,255,.18);--ink:#eaf0ff;--dim:#9aa6cc;--faint:#6b7499;--accent:#7aa2ff;--accent2:#c77dff;--accent3:#4dd6c1;--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;--sans:"Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.65;-webkit-font-smoothing:antialiased}
body::before{content:"";position:fixed;inset:0;z-index:-1;background:radial-gradient(1100px 650px at 80% -10%,rgba(122,90,248,.18),transparent 60%),radial-gradient(900px 600px at 0% 0%,rgba(77,214,193,.1),transparent 55%);pointer-events:none}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header.nav{position:sticky;top:0;z-index:10;backdrop-filter:blur(10px);background:linear-gradient(180deg,rgba(5,6,13,.85),rgba(5,6,13,.4));border-bottom:1px solid var(--border)}
.nav .in{max-width:1180px;margin:0 auto;padding:0 24px;height:60px;display:flex;align-items:center;gap:22px}
.brand{font-weight:700;font-size:17px;display:flex;gap:8px;align-items:center}
.brand .dot{width:10px;height:10px;border-radius:50%;background:radial-gradient(circle at 30% 30%,#fff,var(--accent) 60%,transparent);box-shadow:0 0 22px rgba(122,162,255,.5)}
.nav nav{margin-left:auto;display:flex;gap:20px;font-size:14px}
.nav nav a{color:var(--dim)}.nav nav a:hover{color:var(--ink);text-decoration:none}.nav nav a.on{color:var(--ink)}
.layout{max-width:1180px;margin:0 auto;padding:30px 24px 80px;display:grid;grid-template-columns:230px 1fr;gap:40px}
@media(max-width:860px){.layout{grid-template-columns:1fr}.side{position:static!important;height:auto!important}}
.side{position:sticky;top:90px;height:max-content;align-self:start}
.side h4{font-family:var(--mono);font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--faint);margin:0 0 10px}
.side a{display:block;padding:8px 12px;border-radius:9px;color:var(--dim);font-size:14.5px;margin-bottom:2px;border:1px solid transparent}
.side a:hover{color:var(--ink);background:var(--panel);text-decoration:none}
.side a.on{color:var(--ink);background:var(--panel);border-color:var(--border)}
.side .sep{height:1px;background:var(--border);margin:14px 0}
.doc{min-width:0;max-width:820px}
.doc h1{font-size:clamp(28px,5vw,42px);letter-spacing:-.8px;margin:0 0 18px;background:linear-gradient(180deg,#fff,#b9c6ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.doc h2{font-size:24px;margin:42px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border);letter-spacing:-.3px}
.doc h3{font-size:18px;margin:28px 0 8px;color:#cdd6f4}
.doc h4{font-size:15px;margin:20px 0 6px;color:var(--dim)}
.doc p,.doc li{color:#c6cfe8}
.doc a{color:var(--accent)}
.doc code{font-family:var(--mono);font-size:.88em;background:rgba(122,162,255,.12);border:1px solid var(--border);border-radius:6px;padding:1px 6px;color:#dbe4ff}
.doc pre{background:#080a16;border:1px solid var(--border);border-radius:12px;padding:20px;overflow-x:auto}
.doc pre code{background:none;border:none;padding:0;font-size:13px;line-height:1.7;color:#cdd6f4}
.doc blockquote{margin:18px 0;padding:12px 18px;border-left:3px solid var(--accent);background:var(--panel);border-radius:0 10px 10px 0;color:var(--dim)}
.doc table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:13px;margin:18px 0;display:block;overflow-x:auto}
.doc th,.doc td{padding:9px 13px;border:1px solid var(--border);text-align:right;white-space:nowrap}
.doc th{color:var(--accent3);background:rgba(77,214,193,.06);text-transform:uppercase;font-size:11px;letter-spacing:.5px}
.doc td:last-child,.doc th:last-child{text-align:left}
.doc hr{border:none;border-top:1px solid var(--border);margin:32px 0}
.doc ul,.doc ol{padding-left:22px}
.doc img{max-width:100%}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:8px}
.card{display:block;background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px;transition:transform .16s,border-color .16s}
.card:hover{transform:translateY(-3px);border-color:rgba(122,162,255,.5);text-decoration:none}
.card h3{margin:0 0 6px;font-size:17px;color:var(--ink)}.card p{margin:0;color:var(--dim);font-size:14px}
`;

function navLinks(active) {
  const docs = DOCS.map(
    (d) => `<a href="${d.html}" class="${active === d.html ? 'on' : ''}">${d.title}</a>`,
  ).join('');
  return `<a href="../">Home</a><a href="../playground/">Playground</a>${docs}<a href="../api/">API</a><a href="https://github.com/ppisljar/starscape">GitHub ↗</a>`;
}

function sideLinks(active) {
  const docs = DOCS.map(
    (d) =>
      `<a href="${d.html}" class="${active === d.html ? 'on' : ''}">${d.title}</a>`,
  ).join('');
  return `<h4>Documentation</h4>${docs}<div class="sep"></div><h4>Explore</h4><a href="../playground/">▶ Playground</a><a href="../api/">API reference</a><a href="https://github.com/ppisljar/starscape">GitHub ↗</a>`;
}

function fixLinks(html) {
  return html
    .replace(/href="(?:\.\/)?README\.md"/g, 'href="readme.html"')
    .replace(/href="(?:\.\/)?DESIGN\.md"/g, 'href="design.html"')
    .replace(/href="(?:\.\/)?SCALE\.md"/g, 'href="scale.html"');
}

function shell({ title, active, body }) {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>${title} — starscape</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ctext y='26' font-size='26'%3E%E2%9C%A8%3C/text%3E%3C/svg%3E"/>
<style>${CSS}</style></head><body>
<header class="nav"><div class="in"><a class="brand" href="../" style="color:inherit"><span class="dot"></span> starscape</a><nav>${navLinks(active)}</nav></div></header>
<div class="layout"><aside class="side">${sideLinks(active)}</aside><main class="doc">${body}</main></div>
</body></html>`;
}

// per-doc pages
for (const d of DOCS) {
  const md = readFileSync(join(root, d.md), 'utf8');
  const body = fixLinks(marked.parse(md));
  writeFileSync(join(docsOut, d.html), shell({ title: d.title, active: d.html, body }));
}

// docs index
const indexBody = `<h1>Documentation</h1>
<p>starscape is a WebGPU graph-visualization library for very large graphs rendered as a flyable 3D field. Start with the guide, read the design contract for the full architecture, and check the performance page for measured numbers.</p>
<div class="cards">${DOCS.map((d) => `<a class="card" href="${d.html}"><h3>${d.title}</h3><p>${d.blurb}</p></a>`).join('')}
<a class="card" href="../api/"><h3>API reference →</h3><p>Every exported function, type and constant, generated from source.</p></a>
<a class="card" href="../playground/"><h3>Playground →</h3><p>Explore every control live on synthetic graphs up to a million nodes.</p></a></div>`;
writeFileSync(join(docsOut, 'index.html'), shell({ title: 'Docs', active: '', body: indexBody }));

console.log('site: wrote index.html, docs/ (%d pages + index), .nojekyll', DOCS.length);
