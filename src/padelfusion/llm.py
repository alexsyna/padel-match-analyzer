"""Coaching summary written by Claude.

Claude interprets the stats but never creates numbers. It only sees the
computed stats. Every number in its text is then checked against the
figures the stats engine produced, and a summary that fails the check
is withheld.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .stats import StatsResult

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a padel coach writing a short training summary for an amateur player \
("P1", the person who filmed the match) from stats computed by an analysis pipeline.

Rules:
- Use only the figures in the provided JSON, written exactly as they appear in each stat's \
"display" field. Do not calculate new numbers: no sums, differences, ratios, averages or \
rounding. If a comparison matters, describe it in words ("more", "about the same").
- Refer to players only as P1, P2, P3, P4. P2 is P1's partner; P3 and P4 are the opponents.
- When you use a Medium or Low confidence stat, say so. Treat Low confidence stats as tentative.
- Do not mention shot types, winners, errors, technique quality or level ratings: the pipeline \
does not measure them.
- 120 to 220 words. Plain prose, then 2-3 concrete training suggestions as a short list."""

_NUM = re.compile(r"(?<![A-Za-z])(\d+(?:[.,]\d+)?)")


@dataclass
class Summary:
    text: str | None
    status: str  # ok | withheld | skipped | error
    detail: str = ""


def llm_payload(result: StatsResult) -> dict:
    return {
        "stats": [
            {"stat": s.label, "player": s.player or "match", "display": s.display,
             "confidence": s.confidence, "notes": s.notes}
            for s in result.stats
        ],
        "player_roles": {"P1": "user", "P2": "user's partner", "P3": "opponent", "P4": "opponent"},
    }


def allowed_numbers(result: StatsResult) -> set[float]:
    """Every number that appears in a stat's display string or notes."""
    nums: set[float] = set()
    for s in result.stats:
        for text in [s.display, *s.notes]:
            for m in _NUM.finditer(text):
                nums.add(float(m.group(1).replace(",", ".")))
    return nums


def untraceable_numbers(text: str, allowed: set[float]) -> list[str]:
    """Numbers in `text` that do not trace to the stats engine.

    Player IDs (P1..P4) and list markers ("1." at line start) are ignored.
    """
    cleaned = re.sub(r"\bP[1-4]\b", "", text)
    cleaned = re.sub(r"(?m)^\s*\d+[.)]\s", "", cleaned)
    bad = []
    for m in _NUM.finditer(cleaned):
        v = float(m.group(1).replace(",", "."))
        if not any(abs(v - a) < 1e-9 for a in allowed):
            bad.append(m.group(1))
    return bad


def _call_claude(client, messages: list[dict]) -> str:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=messages,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in response.content if b.type == "text").strip()


def summarize(result: StatsResult, client=None, max_attempts: int = 2) -> Summary:
    try:
        import anthropic
    except ImportError:
        return Summary(None, "skipped", "anthropic package not installed (pip install padelfusion[llm])")
    if client is None:
        try:
            client = anthropic.Anthropic()
        except Exception as e:  # missing credentials
            return Summary(None, "skipped", f"no Anthropic credentials: {e}")

    allowed = allowed_numbers(result)
    payload = json.dumps(llm_payload(result), indent=1)
    messages = [{"role": "user", "content": f"Match stats:\n{payload}\n\nWrite the summary."}]
    bad: list[str] = []
    for _ in range(max_attempts):
        try:
            text = _call_claude(client, messages)
        except anthropic.APIConnectionError as e:
            return Summary(None, "error", f"could not reach the API: {e}")
        except anthropic.APIStatusError as e:
            return Summary(None, "error", f"API error {e.status_code}: {e.message}")
        except RuntimeError as e:
            return Summary(None, "error", str(e))
        bad = untraceable_numbers(text, allowed)
        if not bad:
            return Summary(text, "ok")
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                "These numbers do not appear in the stats and must not be used: "
                + ", ".join(sorted(set(bad)))
                + ". Rewrite the summary using only figures exactly as given in the display fields.")},
        ]
    return Summary(None, "withheld", "summary contained figures not produced by the stats engine: "
                   + ", ".join(sorted(set(bad))))
