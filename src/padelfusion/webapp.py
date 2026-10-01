"""Local web app: the whole workflow through buttons in the browser.

Runs on this computer only (bound to 127.0.0.1). Videos, runs and labels
live in a data folder (default <project root>/padelfusion-data, or
$PADELFUSION_DATA). Uses only the standard library HTTP server, so no extra
dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from . import video as vid
from .config import PipelineConfig
from .court import COURT_LINES, CourtModel, detect_court, fit_corners_from_lines
from .detection import yolo_available
from .distortion import Distortion, estimate_k

WEB_DIR = Path(__file__).parent / "web"
PROJECT_ROOT = Path(__file__).parent.parent.parent
RUN_ID = re.compile(r"^[a-z0-9-]{6,40}$")
_lock = threading.RLock()  # guards run.json read-modify-write


def data_dir() -> Path:
    d = Path(os.environ.get("PADELFUSION_DATA", PROJECT_ROOT / "padelfusion-data"))
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


# -- run storage ----------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise KeyError(run_id)
    d = data_dir() / "runs" / run_id
    if not d.is_dir():
        raise KeyError(run_id)
    return d


def load_run(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "run.json").read_text())


def save_run(run: dict) -> None:
    with _lock:
        p = data_dir() / "runs" / run["id"] / "run.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(run, indent=1))
        tmp.replace(p)


def update_run(run_id: str, **fields) -> dict:
    with _lock:
        run = load_run(run_id)
        run.update(fields)
        save_run(run)
        return run


def list_runs() -> list[dict]:
    runs = []
    for d in (data_dir() / "runs").iterdir():
        f = d / "run.json"
        if f.exists():
            try:
                runs.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                continue
    return sorted(runs, key=lambda r: r["created"], reverse=True)


def new_run(name: str, kind: str = "upload") -> dict:
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    (data_dir() / "runs" / run_id).mkdir(parents=True)
    run = {"id": run_id, "name": name, "kind": kind, "created": time.time(), "state": "new",
           "stage": "", "progress": 0.0, "log": [], "error": None, "video_file": None}
    save_run(run)
    return run


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(s: dict) -> None:
    settings_path().write_text(json.dumps(s, indent=1))
    try:
        os.chmod(settings_path(), 0o600)
    except OSError:
        pass


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or load_settings().get("anthropic_api_key")


# -- video helpers ----------------------------------------------------------------
def video_path(run: dict) -> Path:
    return run_dir(run["id"]) / run["video_file"]


def probe_run(run: dict) -> dict:
    info = vid.probe(video_path(run))
    errors, warnings = vid.check_capture(info, PipelineConfig().capture)
    frame = vid.read_frame(video_path(run), 0.0)
    k = estimate_k(frame, PipelineConfig().court.distortion_max_residual_px)
    frame = Distortion(k).rectify(frame)
    cv2.imwrite(str(run_dir(run["id"]) / "frame0.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return update_run(run["id"], state="ready", info=info.to_dict(), capture_errors=errors,
                      capture_warnings=warnings, distortion_k=k)


def court_lines(corners) -> list[list[list[float]]]:
    m = CourtModel(np.asarray(corners, dtype=float))
    out = []
    for a, b in COURT_LINES.values():
        ts = np.linspace(0, 1, 12)
        out.append(m.court_to_image(np.outer(1 - ts, a) + np.outer(ts, b)).round(1).tolist())
    net = m.court_to_image([[0, 10], [10, 10]]).round(1).tolist()
    out.append(net)
    return out


# -- background jobs ----------------------------------------------------------------
def _job(run_id: str, fn) -> None:
    def progress(stage: str, frac: float) -> None:
        with _lock:
            log = load_run(run_id).get("log", [])
            if not log or log[-1]["stage"] != stage:
                log.append({"t": time.time(), "stage": stage})
            update_run(run_id, stage=stage, progress=round(frac, 3), log=log)

    def target():
        try:
            fn(progress)
        except Exception as e:  # surfaced in the UI
            update_run(run_id, state="error", error=str(e) or e.__class__.__name__,
                       traceback=traceback.format_exc()[-3000:])

    threading.Thread(target=target, daemon=True).start()


def start_analysis(run_id: str, opts: dict) -> None:
    from .pipeline import analyze

    run = load_run(run_id)
    cfg = PipelineConfig()
    cfg.detection.backend = opts.get("detector", "auto")
    me = tuple(opts["me"]) if opts.get("me") else None
    corners = opts.get("corners")
    use_llm = bool(opts.get("llm")) and api_key() is not None
    client = None
    if use_llm:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key())
    shutil.rmtree(run_dir(run_id) / "out", ignore_errors=True)
    update_run(run_id, state="running", progress=0.0, stage="Starting", log=[], error=None,
               options={k: opts.get(k) for k in ("me", "corners", "detector", "llm", "clips")})

    def work(progress):
        res = analyze(video_path(run), run_dir(run_id) / "out", cfg, corners=corners, me_px=me,
                      use_llm=use_llm, cut_clips=bool(opts.get("clips")),
                      skip_capture_check=bool(opts.get("skip_capture_check")),
                      llm_client=client, progress=progress)
        validation = None
        if (run_dir(run_id) / "labels.json").exists():
            validation = run_validation(run_id)
        update_run(run_id, state="done", progress=1.0, stage="Done",
                   court=res.timeline.meta["court"], summary_status=res.summary.status,
                   validation=validation)

    _job(run_id, work)


def start_preview(run_id: str) -> None:
    """Re-encode to H.264 so the browser can play it (e.g. iPhone HEVC .mov)."""
    run = load_run(run_id)
    update_run(run_id, preview_state="running")

    def work(progress):
        import subprocess

        out = run_dir(run_id) / "preview.mp4"
        subprocess.run([vid.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i",
                        str(video_path(run)), "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                        "-vf", "scale=-2:720", "-c:a", "aac", "-movflags", "+faststart", str(out)],
                       check=True)
        update_run(run_id, preview_state="done", preview_file="preview.mp4")

    def guarded(progress):
        try:
            work(progress)
        except Exception as e:
            update_run(run_id, preview_state="error", preview_error=str(e))

    threading.Thread(target=guarded, args=(None,), daemon=True).start()


def run_validation(run_id: str) -> list[dict]:
    from .validate import validate_dir

    checks = validate_dir(run_dir(run_id) / "out", run_dir(run_id) / "labels.json")
    result = [c.__dict__ for c in checks]
    update_run(run_id, validation=result)
    return result


def run_summary(run_id: str) -> dict:
    """Headline stats for the results page, straight from stats.json."""
    p = run_dir(run_id) / "out" / "stats.json"
    if not p.exists():
        return {}
    stats = json.loads(p.read_text())["stats"]
    return {"stats": [{k: s[k] for k in ("key", "label", "player", "display", "confidence")}
                      for s in stats if s["key"] != "heatmap"]}


# -- HTTP ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "padelfusion"

    def log_message(self, fmt, *args):  # quiet
        pass

    # helpers
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, msg):
        self._json({"error": msg}, status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n))

    def _file(self, path: Path, ctype: str | None = None):
        if not path.is_file():
            return self._error(404, "not found")
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            else:
                start = max(0, size - int(b))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    # routing
    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    def do_DELETE(self):
        self._route("DELETE")

    def _route(self, method):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        try:
            if method == "GET" and not parts:
                return self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if parts and parts[0] == "files" and method == "GET" and len(parts) >= 3:
                base = run_dir(parts[1]).resolve()
                target = (base / "/".join(parts[2:])).resolve()
                if base not in target.parents:
                    return self._error(403, "forbidden")
                return self._file(target)
            if parts[:1] == ["api"]:
                return self._api(method, parts[1:], parse_qs(url.query))
            return self._error(404, "not found")
        except KeyError:
            return self._error(404, "unknown run")
        except Exception as e:
            return self._error(500, f"{e.__class__.__name__}: {e}")

    def _api(self, method, p, query):
        if p == ["env"] and method == "GET":
            return self._json({"yolo": yolo_available(), "anthropic_key": api_key() is not None,
                               "data_dir": str(data_dir())})
        if p == ["settings"] and method == "PUT":
            body = self._body()
            s = load_settings()
            key = (body.get("anthropic_api_key") or "").strip()
            if key:
                s["anthropic_api_key"] = key
            elif "anthropic_api_key" in body:
                s.pop("anthropic_api_key", None)
            save_settings(s)
            return self._json({"anthropic_key": api_key() is not None})
        if p == ["runs"] and method == "GET":
            return self._json(list_runs())
        if p == ["upload"] and method == "PUT":
            name = Path(query.get("filename", ["video.mp4"])[0]).name
            ext = Path(name).suffix.lower() or ".mp4"
            if not re.match(r"^\.[a-z0-9]{1,5}$", ext):
                return self._error(400, "bad file name")
            run = new_run(name)
            dest = run_dir(run["id"]) / f"input{ext}"
            left = int(self.headers.get("Content-Length") or 0)
            with open(dest, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            update_run(run["id"], video_file=dest.name)
            try:
                run = probe_run(load_run(run["id"]))
            except Exception as e:
                run = update_run(run["id"], state="error", error=f"could not read this video: {e}")
            return self._json(run)
        if len(p) >= 2 and p[0] == "runs":
            run_id = p[1]
            rest = p[2:]
            if not rest and method == "GET":
                run = load_run(run_id)
                run["has_labels"] = (run_dir(run_id) / "labels.json").exists()
                if run.get("state") == "done":
                    run["results"] = run_summary(run_id)
                return self._json(run)
            if not rest and method == "DELETE":
                shutil.rmtree(run_dir(run_id))
                return self._json({"ok": True})
            if rest == ["court", "detect"] and method == "POST":
                frame = cv2.imread(str(run_dir(run_id) / "frame0.jpg"))
                m = detect_court(frame, PipelineConfig().court.auto_min_score)
                if m is None:
                    return self._json({"found": False})
                return self._json({"found": True, "corners": m.corners_img.round(1).tolist(),
                                   "score": m.score, "lines": court_lines(m.corners_img)})
            if rest == ["court", "preview"] and method == "POST":
                return self._json({"lines": court_lines(self._body()["corners"])})
            if rest == ["court", "from_lines"] and method == "POST":
                return self._json(fit_corners_from_lines(self._body()["lines"]))
            if rest == ["analyze"] and method == "POST":
                start_analysis(run_id, self._body())
                return self._json(load_run(run_id))
            if rest == ["preview"] and method == "POST":
                start_preview(run_id)
                return self._json(load_run(run_id))
            if rest == ["labels"] and method == "GET":
                f = run_dir(run_id) / "labels.json"
                return self._json(json.loads(f.read_text()) if f.exists() else {})
            if rest == ["labels"] and method == "PUT":
                labels = self._body()
                (run_dir(run_id) / "labels.json").write_text(json.dumps(labels, indent=1))
                return self._json({"ok": True})
            if rest == ["validate"] and method == "POST":
                if not (run_dir(run_id) / "out" / "timeline.json").exists():
                    return self._error(400, "analyze the video first")
                return self._json(run_validation(run_id))
        return self._error(404, "unknown endpoint")


def serve(port: int = 8765, open_browser: bool = True) -> None:
    # Runs interrupted by a restart can't resume; mark them.
    for r in list_runs():
        if r.get("state") == "running":
            update_run(r["id"], state="error", error="interrupted: the app was closed during analysis")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Padel analyzer running at {url}  (data: {data_dir()})")
    print("Keep this window open while you use the app. Close it to stop.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
