"""Command-line interface.

  padelfusion analyze VIDEO --out DIR [--corners ...] [--me X,Y]
  padelfusion frame VIDEO frame.png      # first frame with a pixel grid, for corners / self-tag
  padelfusion validate RUN_DIR labels.json
  padelfusion demo DIR                   # synthetic match -> analyze -> validate
  padelfusion ui                         # the same, through buttons in your browser
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

from . import video as vid
from .config import PipelineConfig
from .court import parse_corners


def _parse_point(text: str) -> tuple[float, float]:
    x, y = (float(v) for v in text.split(","))
    return x, y


def cmd_analyze(a) -> int:
    from .pipeline import CaptureSpecError, CourtNotFoundError, TrackingInitError, analyze

    cfg = PipelineConfig()
    cfg.detection.backend = a.detector
    cfg.tracking.analysis_fps = a.analysis_fps
    corners = parse_corners(a.corners) if a.corners else None
    me = _parse_point(a.me) if a.me else None
    try:
        res = analyze(a.video, a.out, cfg, corners=corners, me_px=me, use_llm=not a.no_llm,
                      cut_clips=a.clips, skip_capture_check=a.skip_capture_check)
    except (CaptureSpecError, CourtNotFoundError, TrackingInitError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"report:   {res.report_path}")
    print(f"timeline: {res.out_dir / 'timeline.json'} (+ CSVs)")
    for s in res.stats.stats:
        if s.key == "heatmap":
            continue
        print(f"  {s.label:<32} {s.player or 'match':<6} {s.display:<24} [{s.confidence}]")
    print(f"summary:  {res.summary.status} {res.summary.detail}")
    return 0


def cmd_frame(a) -> int:
    frame = vid.read_frame(a.video, a.t)
    h, w = frame.shape[:2]
    step = 100
    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(frame, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(frame, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(a.out, frame)
    print(f"wrote {a.out} ({w}x{h}). Read off the court corners (near-left, near-right, far-right, "
          f"far-left) for --corners and a point on yourself for --me.")
    return 0


def cmd_validate(a) -> int:
    from .validate import format_checks, validate_dir

    checks = validate_dir(a.run_dir, a.labels)
    print(format_checks(checks))
    if a.json:
        Path(a.json).write_text(json.dumps([c.__dict__ for c in checks], indent=1))
    return 0 if all(c.passed for c in checks) else 1


def cmd_demo(a) -> int:
    from .pipeline import analyze
    from .synth import SynthSpec, generate
    from .validate import format_checks, validate_dir

    out = Path(a.out)
    spec = SynthSpec(duration_s=a.duration)
    paths = generate(out / "input", spec)
    meta = json.loads(paths["meta"].read_text())
    cfg = PipelineConfig()
    cfg.detection.backend = "background"
    res = analyze(paths["video"], out / "run", cfg, me_px=tuple(meta["me_px"]),
                  use_llm=not a.no_llm, skip_capture_check=True)
    print(f"court: {res.timeline.meta['court']['source']} (score {res.timeline.meta['court']['score']})")
    print(f"report: {res.report_path}")
    checks = validate_dir(out / "run", paths["labels"])
    print(format_checks(checks))
    return 0 if all(c.passed for c in checks) else 1


def cmd_ui(a) -> int:
    from .webapp import serve

    serve(port=a.port, open_browser=not a.no_browser)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="padelfusion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="analyze a match video")
    an.add_argument("video")
    an.add_argument("--out", required=True, help="output directory")
    an.add_argument("--corners", help="manual court corners 'x,y;x,y;x,y;x,y' "
                                      "(near-left, near-right, far-right, far-left)")
    an.add_argument("--me", help="pixel 'x,y' on yourself in the first frame")
    an.add_argument("--detector", default="auto", choices=["auto", "yolo", "background"])
    an.add_argument("--analysis-fps", type=float, default=15.0)
    an.add_argument("--no-llm", action="store_true", help="skip the Claude coaching summary")
    an.add_argument("--clips", action="store_true", help="cut one MP4 clip per rally")
    an.add_argument("--skip-capture-check", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    fr = sub.add_parser("frame", help="export a frame with a pixel grid")
    fr.add_argument("video")
    fr.add_argument("out")
    fr.add_argument("--t", type=float, default=0.0)
    fr.set_defaults(fn=cmd_frame)

    va = sub.add_parser("validate", help="compare a run with hand labels")
    va.add_argument("run_dir")
    va.add_argument("labels")
    va.add_argument("--json", help="also write results as JSON")
    va.set_defaults(fn=cmd_validate)

    de = sub.add_parser("demo", help="synthetic end-to-end run")
    de.add_argument("out")
    de.add_argument("--duration", type=float, default=90.0)
    de.add_argument("--no-llm", action="store_true")
    de.set_defaults(fn=cmd_demo)

    ui = sub.add_parser("ui", help="open the point-and-click app in your browser")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    ui.set_defaults(fn=cmd_ui)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
