"""Interactive manual-review page for flagged subtask segments, grouped by problem type.

Builds ``<out>/review_app.html`` (+ ``<out>/review_app/dense/*.jpg`` frame strips). Serve
``<out>`` over HTTP (e.g. ``python -m http.server -d outputs/subtask``) and open the page.
Decisions are kept in the browser (localStorage) and exported as ``overrides.csv`` /
``canonical_aliases.json``; drop those files into ``<out>`` and rerun vocab.py.

Example:
  python scripts/subtask/review_app.py --dataset data/pick_place_2_3_openwam_v3 --out outputs/subtask
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HEAD_CAMERA, WRIST_CAMERA, episode_table, label_path, read_frames, save_jpeg, video_path  # noqa: E402

MODELS = ("gpt-6.1-sol", "gpt-6-astra")
CATEGORIES = [  # key, title, flags (a segment shows under every category it matches)
    ("empty", "空抓 / 抓取失败", {"no_object", "failed"}),
    ("object", "物体名称不一致", {"disagree_object"}),
    ("bin", "是否放进篮子存疑", {"not_in_bin", "disagree_success"}),
    ("hand", "手别不符 / 双手重叠 / 过短 / 多物体", {"hand_mismatch", "overlap", "short", "multiple_objects"}),
    ("conf", "低置信度", {"low_conf"}),
    ("release", "未检测到松手", {"open_at_end"}),
    ("still", "静止段 (Stay still)", set()),
]
AUTO_EXCLUDE = {"no_object", "failed", "not_in_bin", "multiple_objects", "manual_exclude", "short"}
SUGGESTED_ALIASES = {  # reviewer suggestions prefilled in the name tab (editable there)
    "black toy dinosaur": "black toy", "black toy gun": "black toy", "black toy scorpion": "black toy",
    "black toy gorilla": "black toy", "black toy lizard": "black toy", "black toy cat": "black toy",
    "black toy dragon": "black toy", "white syringe": "white pipette", "gray pipette": "white pipette",
    "white utensil": "white pipette", "black object": "black toy",
}
KEYFRAME_ORDER = {"PICK": [("head", "start"), ("head", "grasp"), ("wrist", "grasp"), ("wrist", "lift"),
                           ("head", "release"), ("head", "after")],
                  "STILL": [("head", "start"), ("head", "mid"), ("head", "end")]}


def dense_strip(job):
    dataset, out, ep_row, items = job
    import cv2

    ep = int(ep_row["episode_index"])
    head_req, wrist_req = defaultdict(list), defaultdict(list)
    for it in items:
        it["head_frames"] = np.linspace(it["start"], it["end"] - 1, 10).astype(int).tolist()
        head_req[HEAD_CAMERA] += it["head_frames"]
        if it["type"] == "PICK":
            a, b = max(it["start"], it["grasp"] - 15), min(it["end"] - 1, max(it["release"], it["grasp"] + 30) + 15)
            it["wrist_frames"] = np.linspace(a, b, 5).astype(int).tolist()
            wrist_req[WRIST_CAMERA[it["hand"]]] += it["wrist_frames"]
    decoded = {cam: read_frames(video_path(dataset, ep_row, cam), f, 320) for cam, f in {**head_req, **wrist_req}.items()}
    paths = {}
    for it in items:
        def tile(img, f, w=256):
            t = cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (62, 18), (0, 0, 0), -1)
            cv2.putText(t, str(f), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            return t
        head = [tile(decoded[HEAD_CAMERA][f], f) for f in it["head_frames"]]
        rows = [np.concatenate(head[:5], 1), np.concatenate(head[5:], 1)]
        if it["type"] == "PICK":
            cam = WRIST_CAMERA[it["hand"]]
            wr = [tile(decoded[cam][f], f) for f in it["wrist_frames"]]
            h = rows[0].shape[0]
            wr = [cv2.resize(w, (256, h)) for w in wr]
            rows.append(np.concatenate(wr, 1))
        name = f"review_app/dense/ep{ep:03d}_{it['start']:05d}.jpg"
        save_jpeg(out / name, np.concatenate(rows, 0), quality=80)
        paths[(ep, it["seg"])] = name
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/subtask"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)
    out = args.out

    rules = pd.read_parquet(out / "segments_rules.parquet")
    labels = pd.read_parquet(out / "segment_labels.parquet").set_index(["episode_index", "seg_id"])
    kf = pd.read_parquet(out / "keyframes.parquet")
    kf_idx = kf.set_index(["episode_index", "seg_id", "camera", "tag"]).path
    eps = episode_table(args.dataset).set_index("episode_index", drop=False)
    vocab = json.loads((out / "vocab_map.json").read_text())["result"]["map"] if (out / "vocab_map.json").exists() else {}

    review_flags = set().union(*(c[2] for c in CATEGORIES))
    segments, by_ep = [], defaultdict(list)
    for r in rules.itertuples():
        lab = labels.loc[(r.episode_index, r.seg_id)]
        flags = set(f for f in lab["flags"].split(",") if f)
        if not (flags & review_flags) and r.type != "STILL":
            continue
        ev = json.loads(r.events) if r.events else {}
        wrist = "lwrist" if r.hand == "left" else "rwrist"
        imgs = []
        for cam, tag in KEYFRAME_ORDER[r.type]:
            c = wrist if cam == "wrist" else "head"
            key = (r.episode_index, r.seg_id, c, tag)
            if key in kf_idx.index:
                imgs.append({"src": kf_idx[key], "cap": f"{'手腕' if cam == 'wrist' else '头部'} {tag}"})
        models = {}
        for m in MODELS:
            p = label_path(out, m, r)
            models[m] = json.loads(p.read_text())["result"] if p.exists() else {}
        cats = [k for k, _, fl in CATEGORIES if (flags & fl) or (k == "still" and r.type == "STILL")]
        seg = {
            "id": f"{r.episode_index}:{ev.get('grasp', r.start)}", "ep": int(r.episode_index), "seg": int(r.seg_id),
            "type": r.type, "hand": r.hand, "start": int(r.start), "end": int(r.end), "frame": int(ev.get("grasp", r.start)),
            "grasp": int(ev.get("grasp", r.start)), "release": int(ev.get("release", r.end - 1)),
            "flags": sorted(flags), "cats": cats, "object": lab["object"], "object2": lab["object_secondary"],
            "auto_exclude": bool(flags & AUTO_EXCLUDE) or (r.type == "PICK" and lab["object"] in (None, "NONE")),
            "imgs": imgs, "models": models,
        }
        segments.append(seg)
        by_ep[int(r.episode_index)].append(seg)

    jobs = [(args.dataset, out, eps.loc[ep], [dict(s) for s in items]) for ep, items in by_ep.items()]
    dense = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, d in enumerate(pool.map(dense_strip, jobs), 1):
            dense.update(d)
            print(f"\rdense strips {i}/{len(jobs)} episodes", end="", flush=True)
    print()
    for s in segments:
        s["dense"] = dense[(s["ep"], s["seg"])]

    # object names: counts, raw names mapped to each, example wrist images
    pick = labels.reset_index()
    pick = pick[pick.type == "PICK"]
    counts = Counter(pick.object) + Counter(pick.object_secondary)
    counts.pop("NONE", None)
    counts.pop(None, None)
    raw_by_canon = defaultdict(set)
    for raw, c in vocab.items():
        raw_by_canon[c].add(raw)
    examples = defaultdict(list)
    for r in pick.itertuples():
        if r.object and r.object == r.object_secondary and len(examples[r.object]) < 3:
            rule = rules[(rules.episode_index == r.episode_index) & (rules.seg_id == r.seg_id)].iloc[0]
            wrist = "lwrist" if rule.hand == "left" else "rwrist"
            key = (r.episode_index, r.seg_id, wrist, "lift")
            if key in kf_idx.index:
                examples[r.object].append(kf_idx[key])
    for r in pick.itertuples():  # names that only ever appear in disagreements
        for name in (r.object, r.object_secondary):
            if name and name != "NONE" and not examples[name]:
                rule = rules[(rules.episode_index == r.episode_index) & (rules.seg_id == r.seg_id)].iloc[0]
                key = (r.episode_index, r.seg_id, "lwrist" if rule.hand == "left" else "rwrist", "lift")
                if key in kf_idx.index:
                    examples[name].append(kf_idx[key])
    names = [{"name": n, "count": c, "raw": sorted(raw_by_canon.get(n, [])), "examples": examples[n][:3],
              "suggest": SUGGESTED_ALIASES.get(n, "")} for n, c in counts.most_common()]

    data = {"segments": segments, "categories": [{"key": k, "title": t} for k, t, _ in CATEGORIES],
            "names": names, "models": list(MODELS)}
    html = TEMPLATE.replace("/*DATA*/", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (out / "review_app.html").write_text(html)
    print(f"{len(segments)} segments, {len(names)} names -> {out / 'review_app.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subtask Review</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--fg:#1f2430;--muted:#6b7280;--line:#e3e6ea;--accent:#2563eb;--keep:#15803d;--excl:#b91c1c;--warn:#b45309;--chip:#eef2ff}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1d2026;--fg:#e6e8eb;--muted:#9aa1ab;--line:#2c3038;--accent:#60a5fa;--keep:#4ade80;--excl:#f87171;--warn:#fbbf24;--chip:#25304a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:17px;margin:0 8px 0 0;display:inline-block}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
.tab{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:999px;padding:4px 12px;cursor:pointer;font-size:13px}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.tab small{opacity:.8;margin-left:4px}
button.act{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer;font-size:13px}
button.act:hover{border-color:var(--accent)}
.progress{height:6px;background:var(--line);border-radius:3px;overflow:hidden;width:220px;display:inline-block;vertical-align:middle}
.progress>i{display:block;height:100%;background:var(--keep)}
main{padding:12px 16px;max-width:1500px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:12px}
.card.keep{border-left-color:var(--keep)}.card.exclude{border-left-color:var(--excl)}
.head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.head b{font-size:15px}.muted{color:var(--muted)}
.chip{background:var(--chip);border-radius:4px;padding:1px 6px;font-size:12px}
.chip.auto-ex{background:#fde2e2;color:#991b1b}.chip.auto-keep{background:#dcfce7;color:#166534}
@media (prefers-color-scheme:dark){.chip.auto-ex{background:#4c1d1d;color:#fecaca}.chip.auto-keep{background:#173d27;color:#bbf7d0}}
.imgs{display:flex;gap:4px;overflow-x:auto;margin:8px 0}
.imgs figure{margin:0;flex:0 0 auto;text-align:center}
.imgs img{height:150px;border-radius:4px;cursor:zoom-in;display:block}
.imgs figcaption{font-size:11px;color:var(--muted)}
.models{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px}
.model{border:1px solid var(--line);border-radius:6px;padding:6px 8px}
.model .obj{font-weight:600}
.model .use{float:right}
.decide{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.decide label{display:flex;gap:4px;align-items:center}
.decide input[type=text]{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:3px 6px;width:210px}
.decide select{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:2px 4px}
.seg-btn{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:3px 10px;cursor:pointer}
.seg-btn.keep.on{background:var(--keep);color:#fff;border-color:var(--keep)}
.seg-btn.exclude.on{background:var(--excl);color:#fff;border-color:var(--excl)}
details summary{cursor:pointer;color:var(--accent);font-size:13px}
details img{max-width:100%;margin-top:6px;border-radius:4px}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:10;cursor:zoom-out}
#lightbox img{max-width:96vw;max-height:94vh}
table.names{border-collapse:collapse;width:100%;background:var(--card)}
table.names td,table.names th{border:1px solid var(--line);padding:5px 6px;vertical-align:middle;text-align:left}
table.names img{height:70px;border-radius:3px;margin-right:3px;cursor:zoom-in}
.hint{font-size:12px;color:var(--muted)}
.kbd{border:1px solid var(--line);border-radius:3px;padding:0 4px;font-size:11px}
@media (max-width:760px){.models{grid-template-columns:1fr}.imgs img{height:110px}}
</style></head><body>
<header>
  <h1>子任务人工复核</h1><span class="muted" id="summary"></span>
  <span class="progress"><i id="prog"></i></span>
  <div class="bar" id="tabs"></div>
  <div class="bar">
    <label><input type="checkbox" id="onlyTodo"> 只看未复核</label>
    <input type="text" id="search" placeholder="按 episode / 物体筛选，如 73 或 tape" style="border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:3px 8px;width:260px">
    <button class="act" id="exportCsv">导出 overrides.csv</button>
    <button class="act" id="exportAlias">导出 canonical_aliases.json</button>
    <button class="act" id="importBtn">导入复核记录</button><input type="file" id="importFile" accept=".json,.csv" hidden>
    <button class="act" id="backup">备份全部记录(JSON)</button>
    <span class="hint">记录自动保存在本浏览器；导出的文件放到 outputs/subtask/ 后重跑 vocab.py 生效</span>
  </div>
</header>
<main id="main"></main>
<div id="lightbox"><img alt=""></div>
<script>
const DATA = /*DATA*/;
const KEY = "subtask-review-v2";
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
state.seg = state.seg || {}; state.alias = state.alias || {};
for (const n of DATA.names) if (n.suggest && !(n.name in state.alias)) state.alias[n.name] = n.suggest;
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }

const names = DATA.names.map(n => n.name);
const datalist = document.createElement("datalist"); datalist.id = "objnames";
datalist.innerHTML = names.map(n => `<option value="${n}">`).join(""); document.body.appendChild(datalist);

let current = DATA.categories[0].key;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const decided = s => (state.seg[s.id] || {}).decision;

function renderTabs() {
  const tabs = document.getElementById("tabs");
  const cats = [...DATA.categories, {key: "names", title: "物体名称合并"}];
  tabs.innerHTML = cats.map(c => {
    const segs = c.key === "names" ? DATA.names : DATA.segments.filter(s => s.cats.includes(c.key));
    const done = c.key === "names" ? Object.values(state.alias).filter(Boolean).length : segs.filter(decided).length;
    return `<button class="tab ${c.key === current ? "on" : ""}" data-k="${c.key}">${c.title}<small>${done}/${segs.length}</small></button>`;
  }).join("");
  tabs.querySelectorAll(".tab").forEach(b => b.onclick = () => { current = b.dataset.k; render(); window.scrollTo(0, 0); });
  const total = DATA.segments.length, done = DATA.segments.filter(decided).length;
  document.getElementById("summary").textContent = `已复核 ${done} / ${total} 段`;
  document.getElementById("prog").style.width = (100 * done / total) + "%";
}

function modelBox(s, m) {
  const r = s.models[m] || {};
  if (s.type === "STILL") return `<div class="model"><div class="muted">${m}</div>idle=${r.robot_idle} · 有人=${r.human_present} · 有人整理=${r.human_rearranging}<br>${esc(r.note || "")}</div>`;
  return `<div class="model"><button class="act use" data-use="${m}">采用</button><div class="muted">${m}</div>
    <div class="obj">${esc(r.object)}</div>
    抓起=${r.grasp_success} · 入篮=${r.placed_in_bin} · 手=${esc(r.hand_used)} · 置信=${r.confidence} · 人干预=${r.human_intervention}
    ${r.multiple_objects ? " · <b>多物体</b>" : ""}<br><span class="muted">${esc(r.anomaly || "")}</span></div>`;
}

function card(s) {
  const d = state.seg[s.id] || {};
  const auto = s.auto_exclude ? `<span class="chip auto-ex">自动结果：剔除</span>` : `<span class="chip auto-keep">自动结果：保留</span>`;
  const imgs = s.imgs.map(i => `<figure><img loading="lazy" src="${i.src}"><figcaption>${i.cap}</figcaption></figure>`).join("");
  const dur = ((s.end - s.start) / 30).toFixed(1);
  const pickCtl = s.type === "PICK" ? `
    <label>物体 <input type="text" list="objnames" data-f="object" value="${esc(d.object ?? (s.object === "NONE" ? "" : s.object))}"></label>
    <label>抓起 <select data-f="lifted"><option value="">按模型</option><option value="true">是</option><option value="false">否</option></select></label>
    <label>入篮 <select data-f="placed"><option value="">按模型</option><option value="true">是</option><option value="false">否</option></select></label>` : "";
  return `<div class="card ${d.decision || ""}" data-id="${s.id}">
    <div class="head"><b>ep${s.ep} · #${s.seg} · ${s.type}${s.hand ? " · " + (s.hand === "left" ? "左手" : "右手") : ""}</b>
      <span class="muted">帧 [${s.start}, ${s.end}) · ${dur}s${s.type === "PICK" ? ` · grasp ${s.grasp} · release ${s.release}` : ""}</span>
      ${s.flags.map(f => `<span class="chip">${f}</span>`).join("")} ${auto}</div>
    <div class="imgs">${imgs}</div>
    <details><summary>展开密集帧（头部 10 帧${s.type === "PICK" ? " + 手腕 5 帧" : ""}）</summary><img loading="lazy" src="${s.dense}"></details>
    <div class="models">${DATA.models.map(m => modelBox(s, m)).join("")}</div>
    <div class="decide">
      <button class="seg-btn keep ${d.decision === "keep" ? "on" : ""}" data-dec="keep">保留 <span class="kbd">K</span></button>
      <button class="seg-btn exclude ${d.decision === "exclude" ? "on" : ""}" data-dec="exclude">剔除 <span class="kbd">X</span></button>
      ${pickCtl}
      <label>备注 <input type="text" data-f="note" value="${esc(d.note || "")}" style="width:240px"></label>
      ${d.decision ? `<button class="act" data-clear>撤销</button>` : ""}
    </div></div>`;
}

function namesView() {
  const rows = DATA.names.map(n => `<tr><td><b>${esc(n.name)}</b><div class="hint">${esc(n.raw.join(", "))}</div></td>
    <td>${n.count}</td><td>${n.examples.map(e => `<img loading="lazy" src="${e}">`).join("")}</td>
    <td><input type="text" list="objnames" data-alias="${esc(n.name)}" value="${esc(state.alias[n.name] || "")}" placeholder="保持不变"
      style="border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:5px;padding:3px 6px;width:220px">
      ${n.suggest ? `<div class="hint">建议：${esc(n.suggest)}</div>` : ""}</td></tr>`).join("");
  return `<p class="hint">把外观相同、模型叫法不同的物体合并到同一个名称（填目标名称；留空=不变）。计数=两个模型给出该名称的次数，图片为手腕抬起时的画面。</p>
    <table class="names"><tr><th>名称 / 原始叫法</th><th>次数</th><th>示例</th><th>合并为</th></tr>${rows}</table>`;
}

function render() {
  renderTabs();
  const main = document.getElementById("main");
  if (current === "names") {
    main.innerHTML = namesView();
    main.querySelectorAll("[data-alias]").forEach(i => i.onchange = () => { state.alias[i.dataset.alias] = i.value.trim(); save(); renderTabs(); });
  } else {
    const q = document.getElementById("search").value.trim().toLowerCase();
    const todo = document.getElementById("onlyTodo").checked;
    const segs = DATA.segments.filter(s => s.cats.includes(current) && (!todo || !decided(s)) &&
      (!q || String(s.ep) === q || [s.object, s.object2].some(o => (o || "").toLowerCase().includes(q))));
    main.innerHTML = segs.length ? segs.map(card).join("") : `<p class="muted">没有需要显示的片段。</p>`;
    main.querySelectorAll(".card").forEach(bind);
  }
  main.querySelectorAll("img").forEach(img => img.addEventListener("click", () => {
    const lb = document.getElementById("lightbox"); lb.querySelector("img").src = img.src; lb.style.display = "flex"; }));
  main.querySelectorAll("select[data-f]").forEach(sel => {
    const d = state.seg[sel.closest(".card").dataset.id] || {}; sel.value = d[sel.dataset.f] || ""; });
}

function bind(el) {
  const id = el.dataset.id, s = DATA.segments.find(x => x.id === id);
  const get = () => state.seg[id] || (state.seg[id] = {});
  const setDec = dec => {
    const d = get(); d.decision = dec;
    if (dec === "keep" && s.type === "PICK" && d.object === undefined) d.object = el.querySelector('[data-f="object"]').value.trim();
    save(); const y = window.scrollY; render(); window.scrollTo(0, y);
  };
  el.querySelectorAll("[data-dec]").forEach(b => b.onclick = () => setDec(b.dataset.dec));
  el.querySelectorAll("[data-f]").forEach(i => i.onchange = () => { get()[i.dataset.f] = i.value.trim(); save(); });
  el.querySelectorAll("[data-use]").forEach(b => b.onclick = () => {
    const r = s.models[b.dataset.use] || {}; const d = get();
    const o = (r.object || "").toLowerCase().trim();
    d.object = ["none", "no object", "nothing", ""].includes(o) ? "NONE" : o; d.lifted = String(!!r.grasp_success); d.placed = String(!!r.placed_in_bin);
    const ok = r.grasp_success && r.placed_in_bin && d.object !== "NONE";
    setDec(ok ? "keep" : "exclude"); });
  const clr = el.querySelector("[data-clear]"); if (clr) clr.onclick = () => { delete state.seg[id]; save(); const y = window.scrollY; render(); window.scrollTo(0, y); };
  el.addEventListener("mouseenter", () => { hovered = el; });
}
let hovered = null;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !hovered) return;
  const k = e.key.toLowerCase();
  if (k === "k" || k === "x") hovered.querySelector(`[data-dec="${k === "k" ? "keep" : "exclude"}"]`).click();
});
document.getElementById("lightbox").onclick = e => e.currentTarget.style.display = "none";
document.getElementById("onlyTodo").onchange = render;
document.getElementById("search").oninput = render;

function download(name, text, type) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type}));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const csvCell = v => /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
document.getElementById("exportCsv").onclick = () => {
  const lines = ["episode_index,frame,object,lifted,placed,exclude,note"];
  for (const s of DATA.segments) {
    const d = state.seg[s.id]; if (!d || !d.decision) continue;
    const keep = d.decision === "keep";
    let obj = s.type === "PICK" ? (d.object || "") : "", lifted = d.lifted || "", placed = d.placed || "";
    if (["none", "no object"].includes(obj.toLowerCase())) obj = "NONE";
    if (keep && s.type === "PICK") { if (!obj || obj === "NONE") { alert(`ep${s.ep} #${s.seg} 标为保留但没有物体名称`); return; } lifted = lifted || "true"; placed = placed || "true"; }
    if (!keep && s.type === "PICK" && !obj) obj = "";
    lines.push([s.ep, s.frame, obj, lifted, placed, keep ? "false" : "true", d.note || ""].map(v => csvCell(String(v))).join(","));
  }
  download("overrides.csv", lines.join("\n") + "\n", "text/csv");
};
document.getElementById("exportAlias").onclick = () => {
  const out = {}; for (const [k, v] of Object.entries(state.alias)) if (v && v !== k) out[k] = v;
  download("canonical_aliases.json", JSON.stringify(out, null, 1), "application/json");
};
document.getElementById("backup").onclick = () => download("review_state.json", JSON.stringify(state, null, 1), "application/json");
document.getElementById("importBtn").onclick = () => document.getElementById("importFile").click();
document.getElementById("importFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return; const text = await f.text();
  if (f.name.endsWith(".json")) { const s = JSON.parse(text); state.seg = {...state.seg, ...(s.seg || {})}; state.alias = {...state.alias, ...(s.alias || {})}; }
  else {
    const rows = text.trim().split(/\r?\n/).slice(1);
    for (const line of rows) {
      const c = line.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(x => x.replace(/,$/, "").replace(/^"|"$/g, "").replace(/""/g, '"'));
      const id = `${c[0]}:${c[1]}`; if (!DATA.segments.find(s => s.id === id)) continue;
      state.seg[id] = {object: c[2], lifted: c[3], placed: c[4], decision: c[5] === "true" ? "exclude" : "keep", note: c[6]};
    }
  }
  save(); render(); e.target.value = "";
};
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
