"""Analysis config: protocol `config.analysis.v1` (schema defaults + `<fleet>/config/analysis.json`, loaded by
o11y_core), flattened into the shape the analyzer uses. An invalid config file falls back to the defaults and
is reported as `o11y.protocol.invalid_file`."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from o11y_core import api, fleet

PKG_DIR = Path(__file__).resolve().parent
# plugins/agent-sys-o11y/ (config prompt paths are relative to it)
PLUGIN_DIR = PKG_DIR.parent.parent
OTEL_DIR_ENV_VAR = "AGENT_SYS_O11Y_OTEL_DIR"


def default_otel_dir(environ: Mapping[str, str]) -> Path:
    """Design §4: `$AGENT_SYS_O11Y_OTEL_DIR`, else `/var/tmp/$USER/infera_agent_sys/o11y/otel`."""
    explicit = environ.get(OTEL_DIR_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    user = environ.get("USER") or str(os.getuid())
    return Path("/var/tmp") / user / "infera_agent_sys" / "o11y" / "otel"


def _prompt(spec: str, default_name: str) -> str:
    """A prompt path relative to the plugin, an absolute path, or inline text."""
    if "\n" in spec:
        return spec
    for cand in (
        Path(os.path.expanduser(spec)),
        PLUGIN_DIR / spec,
        PKG_DIR / spec,
        PKG_DIR / "prompts" / Path(spec).name,
    ):
        if cand.is_file():
            return cand.read_text(encoding="utf-8")
    return (PKG_DIR / "prompts" / default_name).read_text(encoding="utf-8")


def _task_book(root: Path, spec: str | None, launch_dir: str | None) -> str | None:
    """`sources.task_book`, else config.trigger `task_book`; relative paths are relative to the launch dir."""
    if not spec:
        trig, _ = api.load_config(root, "trigger")
        spec = trig.get("task_book")
    if not spec:
        return None
    p = Path(os.path.expanduser(spec))
    if not p.is_absolute() and launch_dir:
        p = Path(launch_dir) / p
    return str(p)


def load(
    root: Path, environ: Mapping[str, str] | None = None, launch_dir: str | None = None
) -> tuple[dict, list[str]]:
    environ = os.environ if environ is None else environ
    c, problems = api.load_config(Path(root), "analysis")
    det, src = c["detectors"], c["sources"]
    otel_dir = src.get("otel_dir")
    cfg = {
        "claude_bin": c["claude_bin"],
        "llm": {
            "log": {**c["log"], "prompt_text": _prompt(c["log"]["prompt"], "log.md")},
            "display": {
                **c["display"],
                "prompt_text": _prompt(c["display"]["prompt"], "display.md"),
            },
            "disable_thinking": True,
        },
        "thresholds": {
            "token_step": int(det["token_step"]),
            "perm_wait_s": float(det["permission_wait_warn_s"]),
            "idle_warn_s": float(det["idle_warn_s"]),
            "api_error_k": int(det["api_error_repeat"]),
            "crash_grace_s": float(det["crash_grace_s"]),
            "forge_stale_s": float(det["campaign_stale_s"]),
        },
        "sources": {
            # null = per session ($CLAUDE_CONFIG_DIR of the analyzer's env, passed by the trigger)
            "claude_config_dir": os.path.expanduser(src["claude_config_dir"])
            if src.get("claude_config_dir")
            else str(fleet.claude_config_dir(environ)),
            "pool_dir": src["machine_pool_dir"],
            "campaign_globs": list(src["campaign_globs"]),
            "e2e_flow_globs": list(src.get("e2e_flow_globs") or []),
            "otel_dir": os.path.expanduser(otel_dir)
            if otel_dir
            else str(default_otel_dir(environ)),
            "task_book": _task_book(Path(root), src.get("task_book"), launch_dir),
        },
        "facts_stale_after_s": float(c["facts_stale_after_s"]),
        "budget_s": float(c.get("budget_s", 300)),
    }
    return cfg, problems
