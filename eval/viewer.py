"""
Browser viewer for eval runs.

Usage:
    python -m eval.viewer                # http://localhost:7861
    python -m eval.viewer --port 8080
    python -m eval.viewer --runs-dir eval/runs

Loads runs lazily; refresh the page after a new generation lands to see it.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = FastAPI(title="Eval viewer")
RUNS_DIR = Path("eval/runs")


def list_runs() -> List[Dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for p in sorted(RUNS_DIR.iterdir()):
        if not p.is_dir():
            continue
        gen = p / "generations.jsonl"
        jud = p / "judgments.jsonl"
        runs.append({
            "name": p.name,
            "has_generations": gen.exists(),
            "has_judgments": jud.exists(),
            "gen_count": sum(1 for _ in gen.open("r", encoding="utf-8")) if gen.exists() else 0,
            "jud_count": sum(1 for _ in jud.open()) if jud.exists() else 0,
        })
    return runs


def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_run(name: str) -> Dict:
    run_dir = RUNS_DIR / name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No run '{name}'")

    gens = load_jsonl(run_dir / "generations.jsonl")
    juds = load_jsonl(run_dir / "judgments.jsonl")

    # Index judgments by (model, sp, dataset, behavior_id) → {judge: record}
    jud_index: Dict[tuple, Dict[str, dict]] = {}
    for j in juds:
        key = (j["model"], j["system_prompt_id"], j["dataset"], j["behavior_id"])
        jud_index.setdefault(key, {})[j["judge"]] = j

    records = []
    for g in gens:
        key = (g["model"], g["system_prompt_id"], g["dataset"], g["behavior_id"])
        records.append({
            **g,
            "judgments": jud_index.get(key, {}),
        })

    facets = {
        "models": sorted({g["model"] for g in gens}),
        "system_prompts": sorted({g["system_prompt_id"] for g in gens}),
        "datasets": sorted({g["dataset"] for g in gens}),
        "judges": sorted({j["judge"] for j in juds}),
        "labels": sorted({j["label"] for j in juds}),
    }
    return {"name": name, "records": records, "facets": facets}


@app.get("/api/runs")
async def api_runs():
    return JSONResponse(list_runs())


@app.get("/api/runs/{name}")
async def api_run(name: str):
    return JSONResponse(load_run(name))


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0a0e17; --surface: #111827; --surface2: #1a2234; --border: #1f2d45;
    --accent: #2563eb; --text: #e2e8f0; --text-muted: #64748b;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308; --purple: #a855f7;
    --user-bg: #1e3a5f; --bot-bg: #131d2e;
    --font: 'Sora', sans-serif; --mono: 'JetBrains Mono', monospace;
  }
  html, body { height: 100%; background: var(--bg); color: var(--text);
               font-family: var(--font); font-size: 14px; overflow: hidden; }
  .shell { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
  aside { background: var(--surface); border-right: 1px solid var(--border);
          padding: 20px; overflow-y: auto; }
  main  { display: flex; flex-direction: column; overflow: hidden; }
  .topbar { padding: 14px 24px; border-bottom: 1px solid var(--border);
            background: var(--surface); display: flex; gap: 16px;
            align-items: center; flex-shrink: 0; }
  .topbar select { background: var(--surface2); color: var(--text);
                   border: 1px solid var(--border); border-radius: 8px;
                   padding: 6px 10px; font-family: var(--font); font-size: 13px;
                   min-width: 180px; }
  .topbar .counts { color: var(--text-muted); font-size: 12px; margin-left: auto; }
  .messages { flex: 1; overflow-y: auto; padding: 24px; }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  h3 { color: var(--text-muted); font-size: 11px; font-weight: 600;
       text-transform: uppercase; letter-spacing: 1.2px; margin: 18px 0 8px; }
  h3:first-child { margin-top: 0; }
  .pill-group { display: flex; flex-wrap: wrap; gap: 6px; }
  .pill { padding: 4px 10px; border: 1px solid var(--border); border-radius: 16px;
          font-size: 11px; color: var(--text-muted); background: var(--surface2);
          cursor: pointer; user-select: none; transition: all 0.1s;
          font-family: var(--mono); }
  .pill:hover { color: var(--text); border-color: var(--accent); }
  .pill.on { background: var(--accent); color: white; border-color: var(--accent); }
  .pill.on-red    { background: var(--red); border-color: var(--red); color: white; }
  .pill.on-green  { background: var(--green); border-color: var(--green); color: white; }
  .pill.on-yellow { background: var(--yellow); border-color: var(--yellow); color: #111; }
  .pill.on-purple { background: var(--purple); border-color: var(--purple); color: white; }
  input[type=text] { width: 100%; background: var(--surface2); border: 1px solid var(--border);
                     color: var(--text); padding: 8px 10px; border-radius: 8px;
                     font-family: var(--font); font-size: 13px; outline: none; }
  input[type=text]:focus { border-color: var(--accent); }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 12px; margin-bottom: 14px; padding: 0;
          overflow: hidden; }
  .card-meta { padding: 10px 16px; background: var(--surface2);
               border-bottom: 1px solid var(--border); display: flex;
               flex-wrap: wrap; gap: 10px; align-items: center; font-size: 11px;
               color: var(--text-muted); font-family: var(--mono); }
  .badge { padding: 2px 8px; border-radius: 4px; font-weight: 500; font-size: 11px;
           font-family: var(--mono); }
  .badge.complied { background: #4c1313; color: #fca5a5; border: 1px solid #ef4444; }
  .badge.refused  { background: #0f3724; color: #86efac; border: 1px solid #22c55e; }
  .badge.partial  { background: #3a2e0a; color: #fde68a; border: 1px solid #eab308; }
  .badge.unknown  { background: #2c2c2c; color: #aaa;     border: 1px solid #555; }
  .badge.error    { background: #4c1313; color: #fca5a5; border: 1px solid #ef4444; }
  .turn { padding: 14px 18px; line-height: 1.55; white-space: pre-wrap;
          word-wrap: break-word; }
  .turn.user { background: var(--user-bg); border-bottom: 1px solid var(--border);
               border-left: 3px solid var(--accent); }
  .turn.bot  { background: var(--bot-bg); border-left: 3px solid var(--text-muted); }
  .turn .role { font-size: 10px; color: var(--text-muted); font-weight: 600;
                text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
                font-family: var(--mono); }
  .empty { color: var(--text-muted); text-align: center; padding: 60px;
           font-size: 14px; }
  .empty code { font-family: var(--mono); color: var(--text); background: var(--surface2);
                padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>
<div class="shell">

  <aside>
    <h3>Models</h3>
    <div class="pill-group" id="filter-models"></div>

    <h3>System prompts</h3>
    <div class="pill-group" id="filter-sps"></div>

    <h3>Datasets</h3>
    <div class="pill-group" id="filter-ds"></div>

    <h3>Labels</h3>
    <div class="pill-group" id="filter-labels"></div>

    <h3>Judge (display)</h3>
    <div class="pill-group" id="filter-judges"></div>

    <h3>Search</h3>
    <input type="text" id="search" placeholder="prompt or response text…" />
  </aside>

  <main>
    <div class="topbar">
      <span>Run:</span>
      <select id="run-select"></select>
      <span class="counts" id="counts">—</span>
    </div>
    <div class="messages" id="messages">
      <div class="empty">Loading…</div>
    </div>
  </main>

</div>

<script>
let STATE = { run: null, records: [], facets: null, filters: {
  models: new Set(), sps: new Set(), ds: new Set(), labels: new Set(), judge: null
}, search: "" };

async function fetchJSON(u) { const r = await fetch(u); return r.json(); }

function setPills(elId, values, key, kind) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  values.forEach(v => {
    const span = document.createElement("span");
    span.className = "pill";
    span.textContent = v;
    span.onclick = () => {
      if (key === "judge") {
        STATE.filters.judge = STATE.filters.judge === v ? null : v;
      } else {
        const set = STATE.filters[key];
        set.has(v) ? set.delete(v) : set.add(v);
      }
      render();
    };
    el.appendChild(span);
  });
}

function updatePillStyles() {
  for (const [key, elId, kind] of [
    ["models",  "filter-models", "on"],
    ["sps",     "filter-sps",    "on"],
    ["ds",      "filter-ds",     "on-purple"],
    ["labels",  "filter-labels", "label"],
  ]) {
    document.querySelectorAll(`#${elId} .pill`).forEach(p => {
      p.className = "pill";
      const val = p.textContent;
      const set = STATE.filters[key];
      if (set.has(val)) {
        if (key === "labels") {
          p.classList.add(val === "complied" ? "on-red"
                       : val === "refused"  ? "on-green"
                       : val === "partial"  ? "on-yellow"
                       : "on");
        } else p.classList.add(kind);
      }
    });
  }
  document.querySelectorAll("#filter-judges .pill").forEach(p => {
    p.className = "pill";
    if (STATE.filters.judge === p.textContent) p.classList.add("on");
  });
}

function recordMatches(r) {
  const f = STATE.filters;
  if (f.models.size && !f.models.has(r.model)) return false;
  if (f.sps.size && !f.sps.has(r.system_prompt_id)) return false;
  if (f.ds.size && !f.ds.has(r.dataset)) return false;
  if (f.labels.size) {
    const labels = Object.values(r.judgments).map(j => j.label);
    if (!labels.some(l => f.labels.has(l))) return false;
  }
  if (STATE.search) {
    const q = STATE.search.toLowerCase();
    if (!(r.prompt || "").toLowerCase().includes(q) &&
        !(r.response || "").toLowerCase().includes(q)) return false;
  }
  return true;
}

function renderRecord(r) {
  const judgeFilter = STATE.filters.judge;
  const allJudges = Object.values(r.judgments);
  const showJudges = judgeFilter
    ? allJudges.filter(j => j.judge === judgeFilter)
    : allJudges;

  const badges = showJudges.map(j =>
    `<span class="badge ${j.label}">${j.judge}: ${j.label}</span>`
  ).join("");

  const ts = r.ts ? r.ts.slice(0, 19).replace("T", " ") : "";
  const lat = r.latency_ms ? `${(r.latency_ms / 1000).toFixed(1)}s` : "";

  return `<div class="card">
    <div class="card-meta">
      <span>${r.model}</span>
      <span>·</span>
      <span>${r.system_prompt_id}</span>
      <span>·</span>
      <span>${r.dataset}/${r.behavior_id}</span>
      ${badges}
      <span style="margin-left:auto">${ts} · ${lat}</span>
    </div>
    <div class="turn user">
      <div class="role">user</div>
      ${escapeHtml(r.prompt || "")}
    </div>
    <div class="turn bot">
      <div class="role">assistant</div>
      ${r.error ? `<span style="color:#fca5a5">[error] ${escapeHtml(r.error)}</span>`
                : escapeHtml(r.response || "")}
    </div>
  </div>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  })[c]);
}

function render() {
  updatePillStyles();
  const filtered = STATE.records.filter(recordMatches);
  const el = document.getElementById("messages");
  document.getElementById("counts").textContent =
    `${filtered.length} / ${STATE.records.length} records`;
  if (filtered.length === 0) {
    el.innerHTML = `<div class="empty">No records match. Try clearing filters.</div>`;
    return;
  }
  el.innerHTML = filtered.map(renderRecord).join("");
}

async function loadRun(name) {
  document.getElementById("messages").innerHTML =
    `<div class="empty">Loading <code>${name}</code>…</div>`;
  const data = await fetchJSON(`/api/runs/${encodeURIComponent(name)}`);
  STATE.run = name;
  STATE.records = data.records;
  STATE.facets = data.facets;
  setPills("filter-models", data.facets.models, "models");
  setPills("filter-sps",    data.facets.system_prompts, "sps");
  setPills("filter-ds",     data.facets.datasets, "ds");
  setPills("filter-labels", data.facets.labels, "labels");
  setPills("filter-judges", data.facets.judges, "judge");
  render();
}

async function init() {
  const runs = await fetchJSON("/api/runs");
  const sel = document.getElementById("run-select");
  if (runs.length === 0) {
    document.getElementById("messages").innerHTML =
      `<div class="empty">No runs in <code>eval/runs/</code> yet. Generate some with <code>python -m eval.runner …</code></div>`;
    return;
  }
  runs.forEach(r => {
    const opt = document.createElement("option");
    opt.value = r.name;
    opt.textContent = `${r.name}  (${r.gen_count} gen, ${r.jud_count} jud)`;
    sel.appendChild(opt);
  });
  sel.onchange = () => loadRun(sel.value);
  document.getElementById("search").oninput = (e) => {
    STATE.search = e.target.value;
    render();
  };
  loadRun(runs[runs.length - 1].name);   // pick most recent by default
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


def main():
    global RUNS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="eval/runs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    print(f"\nViewer: http://localhost:{args.port}")
    print(f"Runs:   {RUNS_DIR.resolve()}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
