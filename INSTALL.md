# Installation

Two ways to install: the point-and-click launcher (no terminal, recommended for everyday use) or a manual command-line install (for development, or if the launcher doesn't fit your setup).

## Requirements

| | |
|---|---|
| **Python** | 3.10 or newer ([python.org/downloads](https://www.python.org/downloads/)) |
| **OS** | macOS, Windows, or Linux |
| **Disk space** | ~500 MB for dependencies, plus space for your videos and run outputs (a 10-minute clip and its analysis typically run a few hundred MB) |
| **GPU** | Not required. YOLO detection and the rest of the pipeline run on CPU. |
| **ffmpeg** | Not required separately — bundled via the `imageio-ffmpeg` dependency |

No account, API key, or internet access is required for the core analysis. An Anthropic API key is only needed if you want Claude-written coaching summaries (optional, see below).

## Option 1: Point-and-click launcher (recommended)

1. Install Python 3.10+ if you don't have it. On Windows, tick **"Add python.exe to PATH"** during setup.
2. Download this repository (GitHub: *Code → Download ZIP*, then unzip) or `git clone` it.
3. Double-click the launcher for your system, in the project's top folder:
   - **macOS**: `start-mac.command` — the first time, macOS blocks unsigned scripts; right-click it → **Open** → **Open** to allow it.
   - **Windows**: `start-windows.bat`
   - **Linux**: `start-linux.sh` (or `bash start-linux.sh` from a terminal)

The first launch creates a `.venv` folder next to the launcher and installs everything into it — a few minutes, longer with the YOLO detector (`ultralytics`). Every launch after that starts in a few seconds. Your browser opens automatically at `http://127.0.0.1:8765`. Keep the launcher's terminal window open while you use the app; closing it stops the server.

To skip the YOLO detector (smaller, faster install, but less reliable detection on real footage — see the README's capture spec), set `PADELFUSION_EXTRAS=llm` before running the launcher.

## Option 2: Command line

```bash
git clone <this repo>
cd padelfusion-claude-peaceful-hawking-jullzg
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # core: numpy, scipy, opencv, matplotlib, bundled ffmpeg
pip install -e '.[yolo]'         # recommended: YOLO person detector for real footage
pip install -e '.[llm]'          # optional: Claude coaching summaries
pip install -e '.[dev]'          # optional: pytest, for running the test suite
```

Then either open the browser app:

```bash
padelfusion ui
```

or use the CLI directly — see the [README](README.md#usage) for the full `analyze` / `validate` / `demo` command reference.

## YOLO weights

The YOLO detector needs model weights (`yolov8s.pt`, ~22 MB). A copy ships at the project root, so no download happens on first use. If it's ever missing, `ultralytics` downloads it automatically the first time YOLO detection runs (needs internet that one time).

## Claude coaching summaries (optional)

To get an AI-written coaching summary in the report:

1. Install the `llm` extra (`pip install -e '.[llm]'`, or leave the launcher's default extras as-is).
2. Get an API key from [console.anthropic.com](https://console.anthropic.com/).
3. Paste it on the app's home page (stored locally in your data folder, never sent anywhere except to Anthropic), or set the `ANTHROPIC_API_KEY` environment variable before launching.

Every number Claude writes is checked against the computed stats; if a summary doesn't check out, it's withheld rather than shown with a wrong figure.

## Verify it works

```bash
pip install -e '.[dev]'
pytest
```

65 tests should pass, covering court mapping, detection, tracking, shot/rally detection, stats confidence tiers, and an end-to-end synthetic-match run.

To check the full pipeline end to end without your own footage:

```bash
padelfusion demo out/demo --no-llm
```

This renders a synthetic match with known ground truth, analyzes it, and validates the result — a good first thing to run after installing.

## Where your data lives

Videos, runs, and reports are kept in `padelfusion-data/` inside the project folder. Set the `PADELFUSION_DATA` environment variable to use a different location instead.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python 3.10 or newer is needed" | Install from [python.org/downloads](https://www.python.org/downloads/); on Windows make sure "Add to PATH" was ticked, then try again |
| macOS blocks `start-mac.command` ("unidentified developer") | Right-click the file → **Open** → **Open** in the dialog. This is only needed once. |
| `pip install -e '.[yolo]'` fails | The launcher automatically retries without the YOLO extra. For a manual install, run `pip install -e '.[llm]'` instead and use `--detector background` on the CLI (lower accuracy on real footage, see the README's known limits) |
| Port 8765 already in use | Another instance is probably already running — check for an existing launcher window, or set a different port with `padelfusion ui --port 8766` |
| Re-installing after a dependency change | Delete the `.venv` folder and re-run the launcher, or re-run `pip install -e .` manually |

## Updating

Pull the latest changes (or download a fresh ZIP) and re-run the launcher — it reinstalls dependencies automatically when `pyproject.toml` changes. For a manual install, re-run the `pip install -e .` commands above.

## Uninstalling

Delete the project folder. Nothing is installed outside it — no system-wide packages, no background services, no files written elsewhere on your computer.
