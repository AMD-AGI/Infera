"""The display model pass: a cheap extraction of task / current run / plan from the MAIN session's newest
history and the task book, used only when the ledger lacks them (design §8.5). Output goes to o11y_facts'
own state (`state/facts/display_extract.json`), never to the ledger; facts mark it `extracted`."""

from __future__ import annotations

import json
import time
from pathlib import Path

from o11y_core import api, store
from o11y_sources import model
from o11y_sources.config import prompt_text
from o11y_sources.cursor import CursorStore, write_json_private
from o11y_sources.slices import collect_slices

from .llm_schemas import DISPLAY_OUT

PROMPT = Path(__file__).resolve().parent / "prompts" / "display.md"


def _task_book_excerpt(cfg: dict, limit: int = 6000) -> str:
    p = cfg["sources"].get("task_book")
    try:
        text = Path(p).read_text(encoding="utf-8", errors="replace") if p else ""
    except OSError:
        text = ""
    return text[:limit] if text else "(none)"


def display_pass(
    root: Path, cfg: dict, state_dir: Path, fleet, deadline: float, need: bool
) -> dict:
    dc = cfg["llm"]["display"]
    report = {"llm": None}
    mains = [s for s in fleet if s.role == "main"]
    if not need or not mains:
        return report
    cursors = CursorStore(state_dir)
    slices = collect_slices(
        cursors, mains[:1], "llm-display", int(dc["max_slice_bytes"]), int(dc["lookback_bytes"])
    )
    if not slices:
        return report
    sl = slices[0]
    tail = [
        {k: v for k, v in r.items() if k not in ("writer", "v", "id")}
        for r in api.read_ledger(root)[0][-15:]
    ]
    prompt = "\n".join(
        [
            prompt_text(dc["prompt"], PROMPT),
            "\n## Run ledger tail\n" + "\n".join(json.dumps(r, ensure_ascii=False) for r in tail),
            "\n## Task book excerpt\n" + _task_book_excerpt(cfg),
            f"\n## Main session {sl.session.name or sl.session.session_id} (newest part)\n{sl.text}",
        ]
    )
    cwd = state_dir / "model_cwd"  # neutral dir: no project CLAUDE.md
    cwd.mkdir(parents=True, exist_ok=True, mode=0o700)
    res = model.run(
        prompt,
        DISPLAY_OUT,
        model=dc["model"],
        effort=dc.get("effort"),
        claude_bin=cfg["claude_bin"],
        timeout_s=max(30, min(float(dc["timeout_s"]), deadline - time.time())),
        disable_thinking=cfg["llm"].get("disable_thinking", True),
        cwd=str(cwd),
    )
    report["llm"] = {
        "ok": res.ok,
        "cost_usd": res.cost_usd,
        "duration_s": round(res.duration_s, 1),
        "model": res.model,
        "error": res.error,
        "prompt_chars": len(prompt),
    }
    if not res.ok:
        return report
    out = dict(res.data)
    out["evidence"] = sl.evidence(out.pop("evidence_offsets", []))
    out["generated_at"] = store.now()
    out["session_id"] = sl.session.session_id
    write_json_private(state_dir / "display_extract.json", out)
    cursors.put(f"llm-display:{sl.path}", sl.cursor)
    return report
