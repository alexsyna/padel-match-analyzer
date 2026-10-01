"""Self-contained HTML report: heatmaps, stat cards with confidence labels,
rally list with clip links, and the coaching summary — tabbed, with stats
filterable by player and warnings/method notes tucked behind tooltips so the
page reads cleanly at a glance."""

from __future__ import annotations

import base64
import html
import io
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .court import COURT_LENGTH, COURT_WIDTH, NET_Y, SERVICE_LINE_FROM_NET  # noqa: E402
from .llm import Summary  # noqa: E402
from .stats import TIERS, StatsResult  # noqa: E402
from .timeline import Timeline  # noqa: E402

STAT_ICONS = {
    "net_time_pct": "🥅",
    "net_at_shots_pct": "📍",
    "distance_m": "🏃",
    "partner_distance_m": "🤝",
    "shot_count": "🎾",
    "rally_count": "🔁",
    "rally_mean_s": "⏱️",
    "rally_median_s": "⏱️",
    "rally_longest_s": "⏱️",
    "shots_per_rally": "📊",
    "shots_attributed": "🎯",
}
DEFAULT_STAT_ICON = "📈"


def _heatmap_png(grid: list[list[float]], title: str) -> str:
    g = np.array(grid)
    fig, ax = plt.subplots(figsize=(2.6, 4.6), dpi=110)
    ax.imshow(g, origin="lower", extent=[0, COURT_WIDTH, 0, COURT_LENGTH], cmap="magma",
              interpolation="bilinear", aspect="equal")
    line = dict(color="white", lw=0.9, alpha=0.9)
    ax.plot([0, 10, 10, 0, 0], [0, 0, 20, 20, 0], **line)
    ax.plot([0, 10], [NET_Y, NET_Y], color="#7fd1ff", lw=1.6)
    for y in (NET_Y - SERVICE_LINE_FROM_NET, NET_Y + SERVICE_LINE_FROM_NET):
        ax.plot([0, 10], [y, y], **line)
    ax.plot([5, 5], [NET_Y - SERVICE_LINE_FROM_NET, NET_Y + SERVICE_LINE_FROM_NET], **line)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(5, -0.9, "camera side", ha="center", va="top", fontsize=6, color="#666")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _badge(conf: str) -> str:
    return f'<span class="badge {conf.lower()}">{conf}</span>'


def _tip(text: str) -> str:
    """A small (i) icon whose hover/focus tooltip shows `text`."""
    return f'<span class="tip" tabindex="0" data-tip="{html.escape(text)}">i</span>'


CSS = """
:root{--bg:#fff;--fg:#1b1b1f;--muted:#666;--line:#e3e3e8;--card:#f7f7f9;
--high:#1f7a3f;--medium:#9a6a00;--low:#b3261e}
@media (prefers-color-scheme: dark){:root{--bg:#141418;--fg:#ececf1;--muted:#a0a0aa;
--line:#2c2c34;--card:#1c1c22;--high:#5fd08a;--medium:#e3b341;--low:#ff8a80}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:20px 0 8px}
.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.tablewrap{overflow-x:auto}
.badge{font-size:12px;font-weight:600;padding:1px 8px;border-radius:10px;border:1px solid currentColor}
.badge.high{color:var(--high)}.badge.medium{color:var(--medium)}.badge.low{color:var(--low)}
.heatmaps{display:flex;flex-wrap:wrap;gap:12px}
.heatmaps figure{margin:0;background:var(--card);padding:8px;border-radius:8px}
.heatmaps img{display:block;max-width:100%}
.card{background:var(--card);padding:12px 16px;border-radius:8px}

.metarow{display:flex;align-items:center;gap:8px;flex-wrap:wrap}

/* tooltip (i) icon, used for stat method/notes */
.tip{position:relative;cursor:help;display:inline-block;color:var(--muted);border:1px solid var(--muted);
  border-radius:50%;width:15px;height:15px;text-align:center;font-size:10px;line-height:14px;
  font-style:italic;user-select:none}
.tip:hover::after,.tip:focus::after{content:attr(data-tip);position:absolute;left:50%;
  transform:translateX(-50%);bottom:135%;background:var(--fg);color:var(--bg);padding:7px 10px;
  border-radius:6px;font-size:12px;line-height:1.4;white-space:pre-wrap;width:230px;z-index:10;
  box-shadow:0 2px 10px rgba(0,0,0,.25)}
.tip:hover::before,.tip:focus::before{content:"";position:absolute;left:50%;transform:translateX(-50%);
  bottom:122%;border:5px solid transparent;border-top-color:var(--fg);z-index:10}

/* warnings disclosure */
details.warnbox{display:inline-block;position:relative}
details.warnbox summary{cursor:pointer;list-style:none;color:var(--low);font-weight:600;
  border:1px solid var(--low);border-radius:6px;padding:2px 8px;font-size:13px}
details.warnbox summary::-webkit-details-marker{display:none}
details.warnbox .warnpanel{position:absolute;top:130%;left:0;background:var(--card);
  border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:260px;max-width:360px;
  z-index:20;box-shadow:0 2px 10px rgba(0,0,0,.2)}
details.warnbox .warnpanel p{margin:4px 0;font-size:13px}

/* tabs */
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--line);margin:20px 0 4px}
.tab-btn{background:none;border:none;padding:9px 16px;font:inherit;font-size:14px;cursor:pointer;
  color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}
.tab-btn.active{color:var(--fg);border-color:var(--fg);font-weight:600}
.tabpanel{display:none}
.tabpanel.active{display:block}

/* player picker + stat cards */
.playerpick{display:flex;align-items:center;gap:8px;margin:14px 0}
.playerpick select{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid var(--line);
  background:var(--bg);color:var(--fg)}

/* heatmap + stats side by side: avoids a lone tall/narrow heatmap leaving a
   wide empty gap next to it, and the scroll past that gap to reach stats */
.statsgrid{display:flex;gap:28px;align-items:flex-start;flex-wrap:wrap}
.heatmapcol{flex:0 0 auto}
.heatmapcol h2{margin-top:0}
.statscol{flex:1 1 380px;min-width:280px}
.statscol h2{margin-top:0}
.statcards{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.statcard{background:var(--card);border-radius:8px;padding:12px 14px}
.statcard .row{display:flex;align-items:center;justify-content:space-between;gap:6px}
.statcard .icon{font-size:19px}
.statcard .label{font-size:12.5px;color:var(--muted);margin:6px 0 2px}
.statcard .value{font-size:1.25rem;font-weight:600}
.statcard .valrow{display:flex;align-items:baseline;justify-content:space-between;gap:8px}
@media (max-width:620px){.statsgrid{gap:20px}.heatmapcol{flex-basis:100%}}
"""

TAB_SCRIPT = """
document.querySelectorAll('.tab-btn').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.tab-btn').forEach(function(x){x.classList.remove('active')});
    document.querySelectorAll('.tabpanel').forEach(function(x){x.classList.remove('active')});
    b.classList.add('active');
    document.getElementById('tab-' + b.dataset.tab).classList.add('active');
  });
});
function applyPlayerFilter(){
  var sel = document.getElementById('playersel');
  if(!sel) return;
  var v = sel.value;
  document.querySelectorAll('[data-player]').forEach(function(c){
    c.style.display = (c.dataset.player === 'match' || c.dataset.player === v) ? '' : 'none';
  });
}
var playersel = document.getElementById('playersel');
if(playersel){ playersel.addEventListener('change', applyPlayerFilter); applyPlayerFilter(); }
"""


def render_report(tl: Timeline, result: StatsResult, summary: Summary, out_path: str | Path,
                  video_path: str | Path, clips: dict[int, str] | None = None) -> Path:
    # Goal: zero stats shown without a confidence label.
    for s in result.stats:
        if s.confidence not in TIERS:
            raise ValueError(f"refusing to render stat {s.key!r} without a confidence label")

    out_path = Path(out_path)
    rel_video = os.path.relpath(Path(video_path).resolve(), out_path.parent.resolve())
    meta = tl.meta
    esc = html.escape
    players = sorted(tl.players)
    me_pid = next((p for p in players if tl.players[p].get("is_user")), None)
    me_team = tl.players[me_pid]["team"] if me_pid else None

    def player_label(pid: str) -> str:
        info = tl.players[pid]
        if pid == me_pid:
            return f"{pid} (you)"
        if info["team"] == me_team:
            return f"{pid} (partner)"
        return f"{pid} (opponent)"

    default_pid = me_pid or (players[0] if players else None)

    parts = [f"<!doctype html><html><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Padel Match Report</title><style>{CSS}</style></head><body><main>"]
    parts.append("<h1>Padel match report</h1>")

    warnings = meta.get("warnings", [])
    warn_btn = ""
    if warnings:
        items = "".join(f"<p>⚠ {esc(w)}</p>" for w in warnings)
        warn_btn = (f"<details class='warnbox'><summary>⚠ {len(warnings)}</summary>"
                    f"<div class='warnpanel'>{items}</div></details>")
    parts.append("<p class='metarow'><span class='muted'>"
                 f"{esc(Path(str(meta['video'])).name)} · "
                 f"{meta['duration_s'] / 60:.1f} min · court mapping: {esc(meta['court']['source'])}"
                 + (f" (line score {meta['court']['score']})" if meta['court'].get('score') is not None else "")
                 + f" · detector: {esc(meta['detector'])}</span>{warn_btn}</p>")

    # -- tabs --------------------------------------------------------------
    parts.append("<div class='tabs'>"
                 "<button class='tab-btn active' data-tab='stats'>Heatmap &amp; stats</button>"
                 "<button class='tab-btn' data-tab='rallies'>Rallies</button>"
                 "</div>")

    # == Tab: Heatmap & stats ==============================================
    parts.append("<div id='tab-stats' class='tabpanel active'>")

    parts.append("<h2>Coaching summary</h2><div class='card'>")
    if summary.status == "ok" and summary.text:
        body = esc(summary.text).replace("\n", "<br>")
        parts.append(f"<p>{body}</p><p class='muted'>Written by Claude from the stats below. "
                     "Every figure in it was checked against the stats engine.</p>")
    else:
        parts.append(f"<p class='muted'>No summary ({esc(summary.status)}): {esc(summary.detail)}</p>")
    parts.append("</div>")

    if players:
        parts.append("<div class='playerpick'><label for='playersel' class='muted'>Player</label>"
                     "<select id='playersel'>")
        for pid in players:
            sel = " selected" if pid == default_pid else ""
            parts.append(f"<option value='{esc(pid)}'{sel}>{esc(player_label(pid))}</option>")
        parts.append("</select></div>")

    # Heatmap and stats side by side: a tall, narrow heatmap wastes a whole
    # row of width on its own, and pushing stats below it forces a scroll
    # past empty space to reach them. Putting them in one row uses that
    # width and keeps both visible together.
    parts.append("<div class='statsgrid'>")

    parts.append("<div class='heatmapcol'><h2>Heatmap</h2><div class='heatmaps'>")
    for pid in sorted(result.heatmaps):
        st = result.by_key("heatmap", pid)
        role = "you" if tl.players[pid].get("is_user") else tl.players[pid]["team"] + " team"
        img = _heatmap_png(result.heatmaps[pid]["grid"], f"{pid} ({role})")
        parts.append(f"<figure data-player='{esc(pid)}'><img alt='{pid} heatmap' "
                     f"src='data:image/png;base64,{img}'>"
                     f"<figcaption>{_badge(st.confidence)} <span class='muted'>{esc(st.display)}</span>"
                     f"</figcaption></figure>")
    parts.append("</div></div>")

    # Stat cards: per-player ones scoped to the dropdown, match-level always shown
    parts.append("<div class='statscol'><h2>Stats</h2><div class='statcards'>")
    for s in result.stats:
        if s.key == "heatmap":
            continue
        icon = STAT_ICONS.get(s.key, DEFAULT_STAT_ICON)
        tip_text = s.method + ("\n" + "; ".join(s.notes) if s.notes else "")
        data_player = s.player if s.player else "match"
        parts.append(f"<div class='statcard' data-player='{esc(data_player)}'>"
                     f"<div class='row'><span class='icon'>{icon}</span>{_badge(s.confidence)}</div>"
                     f"<div class='label'>{esc(s.label)}</div>"
                     f"<div class='valrow'><span class='value'>{esc(s.display)}</span>{_tip(tip_text)}</div>"
                     f"</div>")
    parts.append("</div></div>")  # /statcards /statscol
    parts.append("</div>")  # /statsgrid
    parts.append("<p class='muted' style='margin-top:14px'>Confidence: High = positional, checked by "
                 "construction; Medium = audio-based or jitter-sensitive; Low = experimental. Validate "
                 "against a hand count before trusting any number.</p>")
    parts.append("</div>")  # /tab-stats

    # == Tab: Rallies =======================================================
    parts.append("<div id='tab-rallies' class='tabpanel'>")
    parts.append("<h2>Rallies</h2><div class='tablewrap'><table><tr><th>#</th><th>Start</th>"
                 "<th>Duration</th><th>Shots</th><th>Clip</th></tr>")
    for r in tl.rallies:
        if clips and r["rally_id"] in clips:
            href = clips[r["rally_id"]]
        else:
            href = f"{rel_video}#t={r['clip_start']:.1f},{r['clip_end']:.1f}"
        parts.append(f"<tr><td>{r['rally_id']}</td><td>{_ts(r['start'])}</td><td>{r['duration']:.1f} s</td>"
                     f"<td>{r['n_shots']}</td><td><a href='{esc(href)}'>watch</a></td></tr>")
    parts.append("</table></div>")
    parts.append("</div>")  # /tab-rallies

    parts.append(f"<script>{TAB_SCRIPT}</script>")
    parts.append("</main></body></html>")
    out_path.write_text("".join(parts))
    return out_path


def _ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"
