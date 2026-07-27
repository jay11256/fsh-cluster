#!/usr/bin/env python3
"""
video_pred_viewer.py
--------------------
Local web UI that plays a FishFormer-scored recording while scrolling a
swimlane timeline of predicted behavior spans (and ground-truth events) in
sync with the playhead.

Usage
-----
    python video_pred_viewer.py --json /path/to/span_dumps/<label>__<recording>.json

Then open the printed URL (default http://127.0.0.1:8765).

Input
-----
The JSON is the only required input -- it is the span_dumps output of
bhargav/sandboxes/fishtal/dump_former_spans.py (see former.py for the model
that produces it):

    {
      "label": ..., "recording": ..., "epoch": ..., "stride": ..., "duration": ...,
      "gt_events": [[time, class_name], ...],
      "spans": {class_name: [[start, end, score], ...], ...},
      "metrics": {...}
    }

The video itself is not named in the JSON -- only "recording" is -- so the
video is located from the recording id: predictions are timestamped against
the *whole* recording (duration matches the full video length, not any one
4s clip), so this looks up the single whole-recording video under
raw_data/processed_ofure/pairs/<recording>/*.mp4. The per-recording 4s clips
under bhargav/data/ofure/<recording>/clips are a different, non-contiguous
timeline (used for feature extraction) and are not addressable this way.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

PAIRS_ROOT = Path("/fs/vulcan-projects/fsh_track/raw_data/processed_ofure/pairs")
OFURE_ROOT = Path("/fs/vulcan-projects/fsh_track/bhargav/data/ofure")

TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _load_dump(json_path: Path) -> dict:
    with json_path.open() as f:
        d = json.load(f)
    for key in ("recording", "spans", "duration"):
        if key not in d:
            raise ValueError(f"{json_path} is missing required field '{key}'")
    return d


def _resolve_video(recording: str) -> Path:
    """Find the single whole-recording video for `recording`.

    Predictions are timestamped from t=0 to `duration` against the full
    recording, so the video must be the one whole-video mp4 under
    raw_data/processed_ofure/pairs/<recording>/, not the segmented 4s clips
    under bhargav/data/ofure/<recording>/clips (a different, gapped timeline).
    """
    pairs_dir = PAIRS_ROOT / recording
    if pairs_dir.is_dir():
        mp4s = sorted(pairs_dir.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0]
        if len(mp4s) > 1:
            raise FileNotFoundError(
                f"Expected exactly one whole-recording video under {pairs_dir}, "
                f"found {len(mp4s)}: {[p.name for p in mp4s]}"
            )

    hint = ""
    clips_dir = OFURE_ROOT / recording / "clips"
    if clips_dir.is_dir():
        n = len(list(clips_dir.glob("*.mp4")))
        hint = (
            f" Found {n} segmented clips under {clips_dir}, but this viewer needs "
            f"the single whole-recording video, since predictions are timestamped "
            f"against the full recording, not the segmented-clip timeline."
        )
    raise FileNotFoundError(
        f"No whole-recording video found for recording '{recording}' under "
        f"{pairs_dir}.{hint}"
    )


def _build_payload(d: dict, video_name: str, view_window: float) -> dict:
    labels = list(d["spans"].keys())
    colors = {label: TAB10[i % len(TAB10)] for i, label in enumerate(labels)}

    spans = {
        label: [[float(s), float(e), float(sc)] for s, e, sc in d["spans"].get(label, [])]
        for label in labels
    }
    gt_events = [[float(t), str(c)] for t, c in d.get("gt_events", [])]

    return {
        "video_name": video_name,
        "label": d.get("label"),
        "recording": d.get("recording"),
        "epoch": d.get("epoch"),
        "metrics": d.get("metrics", {}),
        "duration": float(d["duration"]),
        "labels": labels,
        "colors": colors,
        "spans": spans,
        "gt_events": gt_events,
        "view_window": view_window,
    }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Video Prediction Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a222c;
      --text: #e7ecf1;
      --muted: #9aa7b5;
      --accent: #3d9cf0;
      --line: #2a3542;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; height: 100%;
      background: var(--bg); color: var(--text);
      font: 14px/1.4 "Segoe UI", system-ui, sans-serif;
    }
    .app {
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;
      gap: 0;
    }
    header {
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
      padding: 10px 16px; border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    header h1 {
      margin: 0; font-size: 15px; font-weight: 600; letter-spacing: 0.02em;
    }
    header .meta { color: var(--muted); font-size: 12px; }
    .controls {
      display: flex; align-items: center; gap: 10px; margin-left: auto; flex-wrap: wrap;
    }
    label.ctrl {
      display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px;
    }
    input[type="range"] { width: 120px; }
    input[type="number"] {
      width: 64px; background: var(--bg); color: var(--text);
      border: 1px solid var(--line); border-radius: 4px; padding: 3px 6px;
    }
    button {
      background: #243040; color: var(--text); border: 1px solid var(--line);
      border-radius: 5px; padding: 5px 10px; cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    .main {
      display: grid;
      grid-template-rows: minmax(220px, 48vh) minmax(220px, 1fr);
      min-height: 0;
    }
    .video-wrap {
      background: #000; display: flex; align-items: center; justify-content: center;
      min-height: 0; border-bottom: 1px solid var(--line);
    }
    video {
      width: 100%; height: 100%; max-height: 48vh; object-fit: contain; background: #000;
    }
    #chart { width: 100%; height: 100%; min-height: 220px; }
    footer {
      display: flex; gap: 16px; flex-wrap: wrap; align-items: center;
      padding: 8px 16px; border-top: 1px solid var(--line); background: var(--panel);
      color: var(--muted); font-size: 12px;
    }
    .scores {
      display: flex; gap: 10px; flex-wrap: wrap; margin-left: auto;
    }
    .score-pill {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 2px 8px; border-radius: 999px; background: #121820;
      border: 1px solid var(--line);
    }
    .swatch { width: 8px; height: 8px; border-radius: 50%; }
    .score-val { font-variant-numeric: tabular-nums; color: var(--text); }
    .hint { opacity: 0.85; }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1 id="title">Video Prediction Viewer</h1>
      <div class="meta" id="meta"></div>
      <div class="controls">
        <label class="ctrl">Window
          <input id="viewWindow" type="number" min="5" step="5" />
          <span>s</span>
        </label>
        <label class="ctrl">Min score
          <input id="minScore" type="range" min="0" max="1" step="0.01" value="0" />
          <span id="minScoreVal">0.00</span>
        </label>
        <label class="ctrl">
          <input id="follow" type="checkbox" checked /> Follow playhead
        </label>
        <label class="ctrl">
          <input id="showGt" type="checkbox" checked /> Ground truth
        </label>
        <button id="fitAll" type="button">Show full timeline</button>
      </div>
    </header>
    <div class="main">
      <div class="video-wrap">
        <video id="video" controls preload="metadata"></video>
      </div>
      <div id="chart"></div>
    </div>
    <footer>
      <span class="hint">Click the chart to seek · scrubbing the video scrolls the timeline</span>
      <span id="clock">t = 0.00 s</span>
      <div class="scores" id="scores"></div>
    </footer>
  </div>
  <script>
    const state = {
      data: null,
      viewWindow: 60,
      minScore: 0,
      follow: true,
      showGt: true,
      chartReady: false,
      lastXRange: null,
    };

    const video = document.getElementById("video");

    function duration() {
      return Math.max(state.data.duration || 0, video.duration || 0);
    }

    function xRangeFor(t) {
      const dur = duration();
      if (!state.follow) return [0, dur];
      const half = state.viewWindow / 2;
      let x0 = Math.max(0, t - half);
      let x1 = Math.min(dur, t + half);
      if (x1 - x0 < state.viewWindow) {
        if (x0 === 0) x1 = Math.min(dur, state.viewWindow);
        else x0 = Math.max(0, x1 - state.viewWindow);
      }
      return [x0, x1];
    }

    // Highest-scoring span active at time t, per label. null entries mean "none".
    function activeSpans(t) {
      return state.data.labels.map(label => {
        const spans = state.data.spans[label];
        let best = null;
        for (const [s, e, score] of spans) {
          if (t >= s && t <= e && (best === null || score > best)) best = score;
        }
        return best;
      });
    }

    function nearestGt(t) {
      const events = state.data.gt_events;
      if (!events.length) return null;
      let best = null, bestDist = Infinity;
      for (const [et, cls] of events) {
        const dist = Math.abs(et - t);
        if (dist < bestDist) { bestDist = dist; best = [et, cls]; }
      }
      return best ? { time: best[0], cls: best[1], delta: best[0] - t } : null;
    }

    function updateScores(t) {
      const vals = activeSpans(t);
      const el = document.getElementById("scores");
      el.innerHTML = state.data.labels.map((label, i) => {
        const v = vals[i];
        const active = v !== null;
        const color = state.data.colors[label];
        return `<span class="score-pill" style="${active ? "border-color:" + color : ""}">
          <span class="swatch" style="background:${color}"></span>
          ${label}
          <span class="score-val">${active ? v.toFixed(3) : "—"}</span>
        </span>`;
      }).join("");

      const gt = nearestGt(t);
      const clockText = gt
        ? `t = ${t.toFixed(2)} s  ·  nearest GT: ${gt.cls} @ ${gt.time.toFixed(2)}s (Δ ${gt.delta >= 0 ? "+" : ""}${gt.delta.toFixed(2)}s)`
        : `t = ${t.toFixed(2)} s`;
      document.getElementById("clock").textContent = clockText;
    }

    function filteredSpans(label) {
      return state.data.spans[label].filter(s => s[2] >= state.minScore);
    }

    function buildTraces() {
      const traces = [];
      for (const label of state.data.labels) {
        const spans = filteredSpans(label).slice().sort((a, b) => a[2] - b[2]);
        const color = state.data.colors[label];
        traces.push({
          type: "bar",
          orientation: "h",
          base: spans.map(s => s[0]),
          x: spans.map(s => s[1] - s[0]),
          y: spans.map(() => label),
          width: 0.6,
          customdata: spans.map(s => [s[0], s[1], s[2]]),
          marker: {
            color: color,
            opacity: spans.map(s => 0.25 + 0.75 * s[2]),
            line: { width: 0 },
          },
          name: label,
          hovertemplate:
            `${label}<br>%{customdata[0]:.2f}s – %{customdata[1]:.2f}s` +
            `<br>score: %{customdata[2]:.3f}<extra></extra>`,
        });
      }
      if (state.showGt && state.data.gt_events.length) {
        const events = state.data.gt_events;
        traces.push({
          type: "scatter",
          mode: "markers",
          x: events.map(e => e[0]),
          y: events.map(e => e[1]),
          marker: {
            symbol: "triangle-down",
            size: 11,
            color: "#ffffff",
            line: { width: 1.5, color: "#000000" },
          },
          name: "Ground truth",
          hovertemplate: `GT: %{y}<br>%{x:.2f}s<extra></extra>`,
        });
      }
      return traces;
    }

    function baseLayout(t) {
      const [x0, x1] = xRangeFor(t);
      state.lastXRange = [x0, x1];
      return {
        paper_bgcolor: "#0f1419",
        plot_bgcolor: "#121820",
        barmode: "overlay",
        margin: { l: 110, r: 20, t: 18, b: 42 },
        xaxis: {
          title: { text: "Time (s)", font: { size: 12, color: "#9aa7b5" } },
          range: [x0, x1],
          color: "#9aa7b5",
          gridcolor: "#2a3542",
          zeroline: false,
        },
        yaxis: {
          type: "category",
          categoryorder: "array",
          categoryarray: state.data.labels.slice().reverse(),
          color: "#9aa7b5",
          gridcolor: "#2a3542",
          zeroline: false,
        },
        legend: {
          orientation: "h",
          y: 1.12,
          font: { color: "#e7ecf1", size: 11 },
        },
        shapes: [{
          type: "line",
          x0: t, x1: t, y0: 0, y1: 1,
          xref: "x", yref: "paper",
          line: { color: "#ffdd57", width: 2 },
        }],
        hovermode: "closest",
      };
    }

    function rebuild(t) {
      Plotly.react("chart", buildTraces(), baseLayout(t), {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
      });
      state.chartReady = true;
      updateScores(t);
    }

    function syncPlayhead(t) {
      if (!state.chartReady) return;
      const patch = {
        "shapes[0].x0": t,
        "shapes[0].x1": t,
      };
      if (state.follow) {
        const [x0, x1] = xRangeFor(t);
        const prev = state.lastXRange;
        if (!prev || Math.abs(prev[0] - x0) > 0.05 || Math.abs(prev[1] - x1) > 0.05) {
          patch["xaxis.range"] = [x0, x1];
          state.lastXRange = [x0, x1];
        }
      }
      Plotly.relayout("chart", patch);
      updateScores(t);
    }

    async function boot() {
      const resp = await fetch("/api/data");
      state.data = await resp.json();
      state.viewWindow = state.data.view_window || 60;
      document.getElementById("viewWindow").value = state.viewWindow;
      document.getElementById("title").textContent =
        `${state.data.label || ""} · ${state.data.recording || state.data.video_name}`;
      const m = state.data.metrics || {};
      const metaBits = [
        `epoch=${state.data.epoch}`,
        `${state.data.labels.length} classes`,
        `${state.data.gt_events.length} GT events`,
      ];
      if (m.avg_map !== undefined) metaBits.push(`avg_mAP=${Number(m.avg_map).toFixed(3)}`);
      if (m.point_recall !== undefined) metaBits.push(`recall=${Number(m.point_recall).toFixed(3)}`);
      if (m.point_precision !== undefined) metaBits.push(`precision=${Number(m.point_precision).toFixed(3)}`);
      document.getElementById("meta").textContent = metaBits.join(" · ");
      video.src = "/video";
      rebuild(0);

      video.addEventListener("seeked", () => syncPlayhead(video.currentTime));
      video.addEventListener("loadedmetadata", () => syncPlayhead(video.currentTime || 0));

      document.getElementById("chart").on("plotly_click", (ev) => {
        if (!ev || !ev.points || !ev.points.length) return;
        const t = ev.points[0].x;
        if (typeof t === "number" && isFinite(t)) {
          const maxT = video.duration || duration();
          video.currentTime = Math.max(0, Math.min(maxT, t));
          syncPlayhead(video.currentTime);
        }
      });

      document.getElementById("viewWindow").addEventListener("change", (e) => {
        state.viewWindow = Math.max(5, Number(e.target.value) || 60);
        rebuild(video.currentTime || 0);
      });
      document.getElementById("minScore").addEventListener("input", (e) => {
        state.minScore = Number(e.target.value) || 0;
        document.getElementById("minScoreVal").textContent = state.minScore.toFixed(2);
        rebuild(video.currentTime || 0);
      });
      document.getElementById("follow").addEventListener("change", (e) => {
        state.follow = e.target.checked;
        rebuild(video.currentTime || 0);
      });
      document.getElementById("showGt").addEventListener("change", (e) => {
        state.showGt = e.target.checked;
        rebuild(video.currentTime || 0);
      });
      document.getElementById("fitAll").addEventListener("click", () => {
        state.follow = false;
        document.getElementById("follow").checked = false;
        rebuild(video.currentTime || 0);
      });

      let lastSync = 0;
      function tick(now) {
        if (!video.paused && !video.ended && now - lastSync > 50) {
          syncPlayhead(video.currentTime);
          lastSync = now;
        }
        requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    boot().catch((err) => {
      document.body.innerHTML = `<pre style="padding:20px;color:#f88">Failed to load: ${err}</pre>`;
    });
  </script>
</body>
</html>
"""


def _range_file_response(path: Path, request: Request) -> Response:
    file_size = path.stat().st_size
    content_type = "video/mp4"
    range_header = request.headers.get("range")

    if not range_header:
        def full_iter():
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            full_iter(),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )

    units, _, rng = range_header.partition("=")
    if units.strip() != "bytes":
        raise HTTPException(status_code=416, detail="Invalid range unit")
    start_s, _, end_s = rng.strip().partition("-")
    try:
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError as exc:
        raise HTTPException(status_code=416, detail="Invalid range") from exc

    start = max(0, start)
    end = min(file_size - 1, end)
    if start > end:
        raise HTTPException(status_code=416, detail="Invalid range bounds")

    length = end - start + 1

    def ranged_iter():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        ranged_iter(),
        status_code=206,
        media_type=content_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


def create_app(video_path: Path, payload: dict) -> FastAPI:
    app = FastAPI(title="Video Prediction Viewer")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/data")
    def api_data() -> JSONResponse:
        return JSONResponse(payload)

    @app.get("/video")
    def video(request: Request) -> Response:
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video not found")
        return _range_file_response(video_path, request)

    return app


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synced video + predicted-span timeline viewer for FishFormer "
        "span_dumps (local web UI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--json",
        required=True,
        help="Path to a span_dumps/<label>__<recording>.json produced by "
        "dump_former_spans.py. This is the only data input -- the video is "
        "located automatically from its 'recording' field.",
    )
    p.add_argument(
        "--view-window",
        type=float,
        default=60.0,
        help="Seconds of timeline visible while following the playhead",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address")
    p.add_argument("--port", type=int, default=8765, help="Bind port")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    json_path = Path(args.json).expanduser().resolve()

    if not json_path.is_file():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        return 1

    d = _load_dump(json_path)

    try:
        video_path = _resolve_video(d["recording"])
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = _build_payload(d, video_name=video_path.name, view_window=args.view_window)
    app = create_app(video_path=video_path, payload=payload)
    url = f"http://{args.host}:{args.port}/"

    n_spans = sum(len(v) for v in payload["spans"].values())
    print("=" * 60)
    print("Video Prediction Viewer")
    print(f"  json  : {json_path}")
    print(f"  video : {video_path}")
    print(f"  label : {payload['label']}  recording: {payload['recording']}  epoch: {payload['epoch']}")
    print(f"  spans : {n_spans} across {len(payload['labels'])} classes")
    print(f"  gt    : {len(payload['gt_events'])} events")
    print(f"  open  : {url}")
    print("=" * 60)
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def _stop(signum, frame):  # noqa: ARG001
        server.should_exit = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
