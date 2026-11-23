"""Analysis configuration (config.analysis.v1: models, effort, prompts, lookback, data locations) plus the
code constants of review F4. The task book is the trigger's setting. An invalid document keeps the last valid
one in effect; `problems_new` is True the first time a given invalid document is seen (report once)."""

from __future__ import annotations

import os
from pathlib import Path

from o11y_sources import otel as otel_src
from o11y_sources import registry

from .door import api
from .door import fleet as fleet_mod

PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent  # plugins/agent-sys-o11y/

CONSTANTS = {
    "claude_bin": "claude",
    "timeout_s": 240,
    "max_slice_bytes": {"log": 200_000, "extract": 120_000},
    "ledger_tail": 40,
    "token_step": 1_000_000,
    "perm_wait_s": 600,
    "idle_warn_s": 1200,
    "api_error_k": 3,
    "crash_grace_s": 60,
    "forge_stale_s": 1800,
    "facts_stale_after_s": 900,
}


def prompt_text(spec: str | None, default_file: Path) -> str:
    """A prompt path relative to the plugin, an absolute path, or inline text; else the job's default."""
    if spec and "\n" in spec:
        return spec
    for cand in [Path(os.path.expanduser(spec)), PLUGIN_DIR / spec] if spec else []:
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
    return default_file.read_text(encoding="utf-8")


def fleet(root: Path, cfg: dict) -> tuple[list, str | None]:
    """(fleet sessions from the trigger's registry, registry problem or None)."""
    rows, problem = api.read_registry(root)
    return registry.fleet_sessions(rows, cfg["sources"]["claude_config_dir"]), problem


def load(root: Path, state_dir: Path) -> tuple[dict, list[str], bool]:
    """(config, problems, problems_are_new) for one job; the task book path is relative to the main's launch dir."""
    cfg, problems, new = api.load_config(root, "analysis", state_dir / "config")
    trig, _, _ = api.load_config(root, "trigger", state_dir / "config")
    src = cfg.get("sources") or {}
    book = trig.get("task_book")
    if book and not os.path.isabs(os.path.expanduser(book)):
        rows, _ = api.read_registry(root)
        launch = next(
            (r.get("launch_dir") or r.get("cwd") for r in rows if r.get("role") == "main"), None
        )
        book = os.path.join(launch, book) if launch else None
    out = {
        **CONSTANTS,
        "log": dict(cfg.get("log") or {}),
        "extract": dict(cfg.get("extract") or {}),
        "sources": {
            "pool_dir": src.get("machine_pool_dir"),
            "e2e_flow_globs": list(src.get("e2e_flow_globs") or []),
            "otel_dir": os.path.expanduser(src["otel_dir"])
            if src.get("otel_dir")
            else otel_src.default_otel_dir(),
            "task_book": os.path.expanduser(book) if book else None,
            "claude_config_dir": str(fleet_mod.claude_config_dir()),
        },
    }
    return out, problems, new
