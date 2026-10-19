"""Local trajectory viewer: episode list, three synchronized camera videos, subtask timeline.

Serves the dataset videos directly (with HTTP Range support so the browser can seek) and
the subtask annotation (``vocab.py`` -> ``subtasks.parquet``).

Example:
  python scripts/subtask/viewer.py --dataset data/pick_place_2_3_openwam_v3 \\
      --annotation outputs/subtask/subtasks.parquet --port 8770
  # then open http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FPS, episode_table, video_path  # noqa: E402

CAMERAS = [
    ("head", "observation.images.left_head", "头部"),
    ("left", "observation.images.left_wrist", "左手腕"),
    ("right", "observation.images.right_wrist", "右手腕"),
]


def build_data(dataset: Path, annotation: Path) -> tuple[dict, dict]:
    eps = episode_table(dataset)
    segs = pd.read_parquet(annotation)
    videos, episodes = {}, []
    for r in eps.itertuples(index=False):
        row = dict(zip(eps.columns, r))
        ep = int(row["episode_index"])
        for key, cam, _ in CAMERAS:
            videos[(key, ep)] = video_path(dataset, row, cam)
        items = []
        for s in segs[segs.episode_index == ep].sort_values("start").itertuples():
            ev = json.loads(s.events) if isinstance(s.events, str) and s.events else {}
            items.append({
                "key": int(ev.get("grasp", s.start)),  # edit key: a frame inside the rule segment
                "start": int(s.start), "end": int(s.end), "type": s.type, "hand": s.hand,
                "object": s.object if isinstance(s.object, str) else None,
                "prompt": s.prompt if s.type == "PICK" else ("Stay still." if s.type == "STILL" else None),
                "exclude": bool(s.exclude),
                "flags": [f for f in (s.flags or "").split(",") if f],
                "events": ev,
            })
        episodes.append({
            "ep": ep, "frames": int(row["length"]), "segments": items,
            "picks": sum(1 for x in items if x["type"] == "PICK" and not x["exclude"]),
            "excluded": sum(1 for x in items if x["exclude"] and x["type"] == "PICK"),
        })
    objects = sorted({x["object"] for e in episodes for x in e["segments"] if x["object"] and x["object"] != "NONE"})
    data = {"fps": FPS, "cameras": [{"key": k, "label": lab} for k, _, lab in CAMERAS], "episodes": episodes,
            "objects": objects,
            "main_prompt": "Pick all items into the bin", "annotation": str(annotation)}
    return data, videos


class EditStore:
    """Reviewer edits made in the viewer, persisted as JSON and exported for the pipeline:
    overrides.csv (vocab.py) and boundary_edits.csv (segment.py)."""

    def __init__(self, out: Path):
        self.out, self.lock = out, threading.Lock()
        path = out / "manual_edits.json"
        self.data = json.loads(path.read_text()) if path.exists() else {"segments": {}, "boundaries": []}

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps(self.data, ensure_ascii=False).encode()

    def apply(self, req: dict) -> None:
        with self.lock:
            if req["kind"] == "segment":
                key = f"{int(req['ep'])}:{int(req['frame'])}"
                fields = {k: req.get(k) for k in ("type", "object", "hand", "decision", "note")}
                if not any(fields[k] for k in ("object", "hand", "decision", "note")):
                    self.data["segments"].pop(key, None)
                else:
                    self.data["segments"][key] = {"ep": int(req["ep"]), "frame": int(req["frame"]), **fields}
            elif req["kind"] == "boundary":
                b = self.data["boundaries"]
                if req.get("delete"):
                    b[:] = [x for x in b if not (x["ep"] == int(req["ep"]) and x["near"] == int(req["near"]))]
                else:
                    b[:] = [x for x in b if not (x["ep"] == int(req["ep"]) and x["near"] == int(req["near"]))]
                    b.append({"ep": int(req["ep"]), "near": int(req["near"]), "to": int(req["to"])})
            else:
                raise ValueError("unknown edit kind")
            self._write()

    def _write(self) -> None:
        def atomic(name, text):
            tmp = self.out / (name + ".tmp")
            tmp.write_text(text)
            os.replace(tmp, self.out / name)

        atomic("manual_edits.json", json.dumps(self.data, ensure_ascii=False, indent=1))
        rows = []
        for e in sorted(self.data["segments"].values(), key=lambda x: (x["ep"], x["frame"])):
            keep, excl = e.get("decision") == "keep", e.get("decision") == "exclude"
            pick = e.get("type") == "PICK"
            rows.append([e["ep"], e["frame"], e.get("object") or "", e.get("hand") or "",
                         "true" if keep and pick else "", "true" if keep and pick else "",
                         "true" if excl else ("false" if keep else ""), e.get("note") or ""])
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["episode_index", "frame", "object", "hand", "lifted", "placed", "exclude", "note"])
        w.writerows(rows)
        atomic("overrides.csv", buf.getvalue())
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["episode_index", "near_frame", "new_frame"])
        w.writerows([[b["ep"], b["near"], b["to"]] for b in self.data["boundaries"]])
        atomic("boundary_edits.csv", buf.getvalue())


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, page: bytes, data: bytes, videos: dict, edits: EditStore, **kw):
        self.page, self.data, self.videos, self.edits = page, data, videos, edits
        super().__init__(*a, **kw)

    def do_POST(self):  # noqa: N802
        if self.path != "/api/edit":
            return self._send(b"not found", "text/plain", 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.edits.apply(req)
        except (ValueError, KeyError, TypeError) as e:
            return self._send(json.dumps({"error": str(e)}).encode(), "application/json", 400)
        self._send(self.edits.snapshot(), "application/json")

    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(self.page, "text/html; charset=utf-8")
        if path == "/data.json":
            return self._send(self.data, "application/json")
        if path == "/api/edits":
            return self._send(self.edits.snapshot(), "application/json")
        m = re.fullmatch(r"/video/(\w+)/(\d+)\.mp4", path)
        if m and (m.group(1), int(m.group(2))) in self.videos:
            return self._video(self.videos[(m.group(1), int(m.group(2)))])
        self._send(b"not found", "text/plain", 404)

    def _video(self, file: Path):
        size = file.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:  # suffix range
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(file, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancels range requests while seeking


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, default=Path("outputs/subtask/subtasks.parquet"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args(argv)
    data, videos = build_data(args.dataset, args.annotation)
    page = (Path(__file__).resolve().parent / "viewer.html").read_bytes()
    handler = partial(Handler, page=page, data=json.dumps(data, ensure_ascii=False).encode(), videos=videos,
                      edits=EditStore(args.annotation.parent))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{len(data['episodes'])} episodes -> http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
