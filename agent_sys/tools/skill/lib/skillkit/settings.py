"""Per-user settings of the infera-agent-sys skills, and the one decision of
when the entry skills engage.

`trigger`:
  auto      — on a request in the domain (LLM serving, perf, kernels, …), decided
              by `matches()`; the same function drives the hint hook.
  explicit  — only when the user names agent_sys. The installed entry skills are
              re-rendered with a description that says so (`render_gated`), so
              Claude does not even load them for a generic request.
`ask_remote`: offer to configure the knowledge repo when it is missing.

Standard library only: `hooks/prompt_hint.py` imports this module directly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

TRIGGERS = ("auto", "explicit")
ALIASES = {
    "on": "auto",
    "off": "explicit",
    "disable": "explicit",
    "disabled": "explicit",
    "enable": "auto",
}
DEFAULTS = {"trigger": "auto", "ask_remote": True}
#: The entry skills, rendered per mode; the others are linked as they are.
GATED = ("infera-agent-sys", "agent-sys-prior-work")

#: Naming agent_sys: agent_sys, agent-sys, agentsys, "agent sys", infera-agent(-sys), the
#: slash command. Not a path component (`.../agent_sys/x.py` is not a request).
EXPLICIT = re.compile(
    r"(?<![/A-Za-z0-9_])(?:agent[\s_-]?sys|infera[\s_-]?agent(?:[\s_-]?sys)?)(?![A-Za-z0-9_/])",
    re.I,
)

#: STRONG terms point at this domain on their own; WEAK ones (generic perf
#: words) need a second hit, so "optimise my web page" stays quiet.
STRONG_EN = [
    "agent_sys",
    "agent-sys",
    "infera",
    "inferencex",
    "inference x",
    "sglang",
    "vllm",
    "trt-llm",
    "tensorrt-llm",
    "llm serving",
    "model serving",
    "ttft",
    "tpot",
    "goodput",
    "rocprof",
    "torch profiler",
    "tracelens",
    "kernelforge",
    "kernel forge",
    "geak",
    "magpie",
    "triton",
    "aiter",
    "hipblaslt",
    "ck tile",
    "composable kernel",
    "attention backend",
    "flash attention",
    "moe",
    "fused moe",
    "pd disaggregation",
    "disaggregation",
    "disagg",
    "prefill",
    "mooncake",
    "kv cache",
    "speculative decoding",
    "mxfp4",
    "cuda graph",
    "hip graph",
    "rocm",
    "mi300x",
    "mi325x",
    "mi355x",
    "gfx942",
    "gfx950",
    "tensor parallel",
    "expert parallel",
    "deepseek",
    "qwen",
    "kimi",
    "llama",
    "task package",
]
WEAK_EN = [
    "atom",
    "mla",
    "agentic",
    "mtp",
    "eagle",
    "glm",
    "handoff",
    "itl",
    "deploy",
    "deployment",
    "serving",
    "throughput",
    "latency",
    "benchmark",
    "bench",
    "profiling",
    "profiler",
    "trace",
    "kernel",
    "gemm",
    "e2e",
    "end-to-end",
    "decode",
    "rdma",
    "quantization",
    "fp8",
    "fp4",
    "llm",
    "inference",
    "gpu",
    "optimize",
    "optimise",
    "speedup",
]
STRONG_ZH = [
    "投机解码",
    "kv缓存",
    "专家并行",
    "张量并行",
    "推理加速",
    "pd分离",
    "大模型",
    "算子",
    "预填充",
    "首字",
    "加速比",
    "投机采样",
    "推理服务",
]
WEAK_ZH = [
    "部署",
    "推理",
    "性能",
    "优化",
    "调优",
    "内核",
    "压测",
    "基准",
    "吞吐",
    "延迟",
    "剖析",
    "性能分析",
    "分离",
    "解码",
    "量化",
    "多机",
    "集群",
    "显卡",
]


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "infera-agent-sys"


def path() -> Path:
    override = os.environ.get("INFERA_AGENT_SYS_SETTINGS")
    return Path(override) if override else config_dir() / "settings.json"


def load() -> dict:
    p = path()
    try:
        data = json.loads(p.read_text()) if p.is_file() else {}
    except ValueError:
        data = {}
    return {**DEFAULTS, **data}


def save(**changes: object) -> dict:
    if "trigger" in changes:
        changes["trigger"] = ALIASES.get(str(changes["trigger"]), changes["trigger"])
    data = {**load(), **changes}
    if data["trigger"] not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS} (or {sorted(ALIASES)})")
    path().parent.mkdir(parents=True, exist_ok=True)
    path().write_text(json.dumps(data, indent=2) + "\n")
    return data


#: Generic performance words: two of these alone are "make my web page faster".
GENERIC = frozenset(
    "optimize optimise latency throughput benchmark bench speedup llm inference gpu deploy deployment "
    "serving 性能 优化 调优 部署 延迟 吞吐 基准 推理 加速比 压测 显卡".split()
)
#: Only the head of a huge pasted prompt is examined (the hook has a time budget).
SCAN_CHARS = 20000


def matches(prompt: str) -> list[str]:
    """Domain terms in `prompt`, or [] when it is not in the domain.

    One STRONG term is enough; otherwise two WEAK terms, at least one of them
    not a generic performance word. Model and hardware names may carry a
    version (`qwen3.6`, `glm5.2`, `mi300x`)."""
    low = prompt[:SCAN_CHARS].lower()

    def en(words: list[str]) -> list[str]:
        return [k for k in words if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z])", low)]

    def zh(words: list[str]) -> list[str]:
        hits = [k for k in words if k in low]
        return [k for k in hits if not any(k != o and k in o for o in hits)]  # 性能 inside 性能分析

    strong = en(STRONG_EN) + zh(STRONG_ZH)
    weak = [w for w in en(WEAK_EN) + zh(WEAK_ZH) if not any(w != x and w in x for x in strong)]
    if strong or (len(weak) >= 2 and any(w not in GENERIC for w in weak)):
        return strong + weak
    return []


#: A slash command of one of these skills names agent_sys too.
SLASH = re.compile(r"(?:^|\s)/(?:infera-agent-sys|agent-sys-[a-z-]+)\b")


def named(prompt: str) -> bool:
    head = prompt[:SCAN_CHARS]
    return bool(EXPLICIT.search(head) or SLASH.search(head))


def engaged(prompt: str, data: dict | None = None) -> bool:
    """The entry skill engages: agent_sys is named, or (auto mode) the prompt is in the domain."""
    data = data or load()
    if named(prompt):
        return True
    return data["trigger"] == "auto" and bool(matches(prompt))


# -- the gated skills, rendered per mode ---------------------------------------- #


def rendered_dir() -> Path:
    return config_dir() / "skills"


_HEADER = re.compile(
    r"<!-- generated by astk mode \((\w+)\) from (.+)/[^/]+/SKILL\.md; edit that file -->"
)


def rendered_from() -> str | None:
    """The checkout the installed entry skills were rendered from (None if never)."""
    p = rendered_dir() / GATED[0] / "SKILL.md"
    m = _HEADER.search(p.read_text()) if p.is_file() else None
    return m.group(2) if m else None


def render_gated(skill_root: Path, trigger: str | None = None) -> list[Path]:
    """Write `<config>/skills/<name>/SKILL.md` for each gated skill.

    The repo's SKILL.md carries both descriptions: `description` (auto) and
    `metadata.explicit_description`. The rendered copy takes the one for the
    mode, and replaces `$(realpath "${CLAUDE_SKILL_DIR}/..")` with the
    checkout's absolute path, because it no longer lives inside the checkout.
    `install.sh` links `~/.claude/skills/<name>` to the rendered dir once;
    changing the mode rewrites only these generated files."""
    trigger = trigger or load()["trigger"]
    out = []
    for name in GATED:
        text = (skill_root / name / "SKILL.md").read_text()
        _, front, body = text.split("---", 2)
        lines = front.strip("\n").splitlines()
        auto = next(ln for ln in lines if ln.startswith("description:"))
        explicit = next(ln for ln in lines if ln.strip().startswith("explicit_description:"))
        chosen = auto if trigger == "auto" else "description:" + explicit.split(":", 1)[1]
        new_front = "\n".join(chosen if ln.startswith("description:") else ln for ln in lines)
        body = body.replace('$(realpath "${CLAUDE_SKILL_DIR}/..")', str(skill_root))
        dst = rendered_dir() / name / "SKILL.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        header = f"<!-- generated by astk mode ({trigger}) from {skill_root / name / 'SKILL.md'}; edit that file -->"
        dst.write_text(f"---\n{new_front}\n---\n{header}{body}")
        for ref in ("references",):
            if (skill_root / name / ref).is_dir() and not (dst.parent / ref).exists():
                (dst.parent / ref).symlink_to(skill_root / name / ref)
        out.append(dst)
    return out
