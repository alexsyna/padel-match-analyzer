"""The local web app's HTTP API: upload, court detection, labels, file serving."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from padelfusion.synth import SynthSpec, generate


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setenv("PADELFUSION_DATA", str(data))
    from padelfusion import webapp

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    video = generate(tmp_path_factory.mktemp("v"), SynthSpec(width=1280, height=720, duration_s=6.0))["video"]
    yield f"http://127.0.0.1:{httpd.server_port}", video
    httpd.shutdown()
    mp.undo()


def call(base, path, method="GET", body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req) as r:
        return r.status, r.read()


def test_page_and_env(server):
    base, _ = server
    status, html = call(base, "/")
    assert status == 200 and b"Padel Analyzer" in html
    env = json.loads(call(base, "/api/env")[1])
    assert "yolo" in env and "anthropic_key" in env


def test_upload_rectifies_fisheye_and_still_detects_court(server, tmp_path_factory):
    base, _ = server
    video = generate(tmp_path_factory.mktemp("v_fisheye"),
                     SynthSpec(width=1280, height=720, duration_s=3.0), distortion_k=-0.2)["video"]
    status, body = call(base, "/api/upload?filename=fisheye.mp4", "PUT", raw=video.read_bytes())
    run = json.loads(body)
    assert run["state"] == "ready"
    assert run["distortion_k"] != 0.0

    court = json.loads(call(base, f"/api/runs/{run['id']}/court/detect", "POST", body={})[1])
    assert court["found"]


def test_court_from_lines_endpoint(server):
    base, video = server
    status, body = call(base, "/api/upload?filename=lines.mp4", "PUT", raw=video.read_bytes())
    run = json.loads(body)
    from padelfusion.synth import image_corners

    c = image_corners(1280, 720)
    req = {"lines": {
        "near_baseline": [c[0].tolist(), c[1].tolist()],
        "right_sideline": [c[1].tolist(), c[2].tolist()],
        "far_baseline": [c[2].tolist(), c[3].tolist()],
        "left_sideline": [c[3].tolist(), c[0].tolist()],
    }}
    result = json.loads(call(base, f"/api/runs/{run['id']}/court/from_lines", "POST", body=req)[1])
    assert result["missing"] == {}
    assert all(result["corners"].values())


def test_upload_detect_labels_and_files(server):
    base, video = server
    status, body = call(base, "/api/upload?filename=match.mp4", "PUT", raw=video.read_bytes())
    run = json.loads(body)
    assert run["state"] == "ready" and run["info"]["has_audio"]
    assert any("1080p" in e for e in run["capture_errors"])

    court = json.loads(call(base, f"/api/runs/{run['id']}/court/detect", "POST", body={})[1])
    assert court["found"] and len(court["corners"]) == 4 and court["lines"]

    labels = {"shots": [1.0, 2.0], "rallies": [{"start": 1.0, "end": 2.0}]}
    call(base, f"/api/runs/{run['id']}/labels", "PUT", body=labels)
    assert json.loads(call(base, f"/api/runs/{run['id']}/labels")[1]) == labels

    status, jpg = call(base, f"/files/{run['id']}/frame0.jpg")
    assert status == 200 and jpg[:2] == b"\xff\xd8"
    with pytest.raises(urllib.error.HTTPError) as e:
        call(base, f"/files/{run['id']}/..%2F..%2Fsettings.json")
    assert e.value.code in (403, 404)

    runs = json.loads(call(base, "/api/runs")[1])
    assert any(r["id"] == run["id"] for r in runs)
    call(base, f"/api/runs/{run['id']}", "DELETE")
    assert not any(r["id"] == run["id"] for r in json.loads(call(base, "/api/runs")[1]))
