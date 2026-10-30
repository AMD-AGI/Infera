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

import hashlib
import json
import os
import re
from pathlib import Path

TRIGGERS = ("auto", "explicit")
ALIASES = {"on": "auto", "enable": "auto"} | dict.fromkeys(
    ("off", "disable", "disabled"), "explicit"
)
DEFAULTS = {"trigger": "auto", "ask_remote": True}
#: The entry skills, rendered per mode; the others are linked as they are.
GATED = ("infera-agent-sys", "agent-sys-prior-work")

#: Naming agent_sys: agent_sys, agent-sys, agentsys, "agent sys", infera-agent(-sys), the
#: slash command, also as the head of a module or skill name (`agent_sys.closure`,
#: `agent-sys-task-runner`). Not a path component (`.../agent_sys/x.py` is not a
#: request), nor the tail of a dotted or dashed identifier (`dev.yihou.agentsys.skills`,
#: `my-agentsys-x`).
EXPLICIT = re.compile(
    r"(?<![/.\-A-Za-z0-9_])(?:agent[\s_-]?sys|infera[\s_-]?agent(?:[\s_-]?sys)?)"
    r"(?![A-Za-z0-9_/])",
    re.I,
)


def _terms(text: str) -> list[str]:
    """A comma-separated term list (terms may hold a space: `llm serving`)."""
    return [t.strip() for t in text.split(",") if t.strip()]


#: STRONG terms point at this domain on their own; WEAK ones need a second hit,
#: one of them a SIGNAL term (see `matches`), so "optimise my web page" and
#: "a stack trace from the linux kernel" stay quiet.
STRONG_EN = _terms(
    "agent_sys, agent-sys, infera, inferencex, inference x, sglang, vllm, trt-llm, tensorrt-llm, "
    "llm serving, model serving, ttft, tpot, goodput, rocprof, torch profiler, tracelens, "
    "kernelforge, kernel forge, geak, magpie, triton, aiter, hipblaslt, ck tile, "
    "composable kernel, attention backend, flash attention, attention kernel, gemm, moe, "
    "fused moe, pd disaggregation, disagg, chunked prefill, mooncake, kv cache, "
    "speculative decoding, mxfp4, cuda graph, hip graph, rocm, mi300x, mi325x, mi355x, gfx942, "
    "gfx950, tensor parallel, expert parallel, deepseek, qwen, kimi, llama, task package"
)
WEAK_EN = (
    "atom mla agentic mtp eagle glm handoff itl deploy deployment serving throughput latency "
    "benchmark bench profiling profiler trace kernel occupancy profile e2e end-to-end decode rdma "
    "quantization fp8 fp4 llm inference gpu optimize optimise speedup prefill disaggregation"
).split()
#: Matched case-sensitively: the ATOM engine, not an atom.
STRONG_CASED = ["ATOM"]
STRONG_ZH = _terms(
    "投机解码, kv缓存, 专家并行, 张量并行, 推理加速, pd分离, 大模型, 算子优化, 算子开发, 算子调优, "
    "算子性能, 算子融合, 融合算子, 算子库, 预填充阶段, 首字延迟, 首字时延, 首字时间, 首token, "
    "首 token, 加速比, 投机采样, 推理服务"
)
WEAK_ZH = (
    "部署 推理 性能 优化 调优 内核 压测 基准 吞吐 延迟 剖析 性能分析 分离 解码 量化 多机 集群 显卡 "
    "算子 预填充"
).split()
#: A Chinese term inside a longer common word is not a hit: 运算子 (an operator
#: in C++), 演算子.
ZH_GUARD = {"算子": r"(?<![运演])算子"}

#: Asking for prior work: `agent-sys-prior-work` engages on these too (its own
#: description lists them), in auto mode only.
PRIOR_WORK_EN = _terms(
    "prior work, previous work, previous result, earlier result, knowledge repo, knowledge base, "
    "handoff, anyone done, anybody done, done this before"
)
PRIOR_WORK_ZH = "有没有人做过 有人做过 之前的结果 以前的结果 之前做过 知识库 前人".split()


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
    # a hand-edited file holding a list or a string is ignored
    return {**DEFAULTS, **(data if isinstance(data, dict) else {})}


def save(**changes: object) -> dict:
    if "trigger" in changes:
        changes["trigger"] = ALIASES.get(str(changes["trigger"]), changes["trigger"])
    data = {**load(), **changes}
    if data["trigger"] not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS} (or {sorted(ALIASES)})")
    path().parent.mkdir(parents=True, exist_ok=True)
    path().write_text(json.dumps(data, indent=2) + "\n")
    return data


#: The WEAK terms that point at LLM serving or kernels. The others (trace, kernel,
#: decode, deploy, e2e, 部署, 集群, 分离, 解码, ...) are as common in web, devops and
#: debugging work, so two of those alone are "make my web page faster".
SIGNAL = frozenset("mla agentic mtp eagle glm itl rdma quantization fp8 fp4 量化 显卡 算子".split())
#: BROAD terms point at the domain only next to a WEAK term that is not GENERIC:
#: "prefill throughput", "llm inference" engage; "prefill the form and optimize
#: the page", "optimize this GPU shader", "deploy an LLM chatbot" stay quiet.
BROAD = frozenset("llm gpu prefill disaggregation 预填充".split())
GENERIC = frozenset(
    "optimize optimise speedup deploy deployment e2e end-to-end 优化 调优 性能 部署".split()
)
#: Only the head of a huge pasted prompt is examined (the hook has a time budget).
SCAN_CHARS = 20000


def matches(prompt: str) -> list[str]:
    """Domain terms in `prompt`, or [] when it is not in the domain.

    One STRONG term is enough; otherwise two WEAK terms, at least one of them
    a SIGNAL term (or a BROAD one next to a non-GENERIC one). Model and hardware
    names may carry a version (`qwen3.6`, `glm5.2`, `mi300x`)."""
    head = prompt[:SCAN_CHARS]
    low = head.lower()
    strong = (
        _en(STRONG_EN, low)
        + [k for k in STRONG_CASED if re.search(rf"(?<![A-Za-z0-9]){k}(?![A-Za-z])", head)]
        + _zh(STRONG_ZH, low)
    )
    weak = [w for w in _en(WEAK_EN, low) + _zh(WEAK_ZH, low) if not any(w in x for x in strong)]
    if strong or (len(weak) >= 2 and any(_signals(w, weak) for w in weak)):
        return strong + weak
    return []


def _signals(w: str, weak: list[str]) -> bool:
    return w in SIGNAL or (w in BROAD and any(o != w and o not in GENERIC for o in weak))


def _en(words: list[str], low: str) -> list[str]:
    return [k for k in words if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z])", low)]


def _zh(words: list[str], low: str) -> list[str]:
    hits = [k for k in words if re.search(ZH_GUARD.get(k) or re.escape(k), low)]
    return [k for k in hits if not any(k != o and k in o for o in hits)]  # 性能 inside 性能分析


def asks_prior_work(prompt: str) -> bool:
    """The prompt asks what the team already has (a handoff, the knowledge repo, ...)."""
    low = prompt[:SCAN_CHARS].lower()
    return bool(_en(PRIOR_WORK_EN, low) or _zh(PRIOR_WORK_ZH, low))


#: A slash command of one of these skills names agent_sys too.
SLASH = re.compile(r"(?:^|\s)/(?:infera-agent-sys|agent-sys-[a-z-]+)\b")


def named(prompt: str) -> bool:
    head = prompt[:SCAN_CHARS]
    return bool(EXPLICIT.search(head) or SLASH.search(head))


def engaged(prompt: str, data: dict | None = None, prior_work: bool = False) -> bool:
    """The entry skill engages: agent_sys is named, or (auto mode) the prompt is in
    the domain. With `prior_work` (the gate of `agent-sys-prior-work`), auto mode
    also engages on a request for prior work, which is that skill's own trigger."""
    data = data or load()
    if named(prompt):
        return True
    if data["trigger"] != "auto":
        return False
    return bool(matches(prompt)) or (prior_work and asks_prior_work(prompt))


# -- the gated skills, rendered per mode ---------------------------------------- #


def rendered_root() -> Path:
    return config_dir() / "rendered"


def rendered_dir(skill_root: Path) -> Path:
    """Where the entry skills of the checkout at `skill_root` are rendered: one
    directory per checkout, so installing from a second worktree never re-points
    the entry skills installed from the first."""
    key = hashlib.sha256(str(Path(skill_root).resolve()).encode()).hexdigest()[:12]
    return rendered_root() / key


_HEADER = re.compile(
    r"<!-- generated by astk mode \((\w+)\) from (.+)/[^/]+/SKILL\.md; edit that file -->"
)


def _header(rendered: Path) -> tuple[str, str] | None:
    """(trigger, checkout) a rendered SKILL.md was generated for, or None."""
    m = _HEADER.search(rendered.read_text()) if rendered.is_file() else None
    return (m.group(1), m.group(2)) if m else None


def render_status(skill_root: Path, name: str) -> str | None:
    """What is wrong with the rendered copy of skill `name`, or None if it is current."""
    got = _header(rendered_dir(skill_root) / name / "SKILL.md")
    if got is None:
        return "not rendered"
    want = load()["trigger"]
    if got[0] != want:
        return f"rendered for {got[0]} mode, the mode is {want}"
    return None


def rerender_all(trigger: str | None = None) -> list[Path]:
    """Re-render every checkout's entry skills for `trigger` (the mode is per user,
    the rendered copies per checkout). A checkout that no longer exists is skipped."""
    out = []
    for p in sorted(rendered_root().glob(f"*/{GATED[0]}/SKILL.md")):
        got = _header(p)
        root = Path(got[1]) if got else None
        if root and all((root / n / "SKILL.md").is_file() for n in GATED):
            out += render_gated(root, trigger)
    return out


def render_gated(skill_root: Path, trigger: str | None = None) -> list[Path]:
    """Write `rendered_dir(skill_root)/<name>/SKILL.md` for each gated skill.

    The repo's SKILL.md carries both descriptions: `description` (auto) and
    `metadata.explicit_description`. The rendered copy takes the one for the
    mode, and replaces `$(realpath "${CLAUDE_SKILL_DIR}/..")` with the
    checkout's absolute path, because it no longer lives inside the checkout.
    `install.sh` links `~/.claude/skills/<name>` to `rendered_dir(skill_root)/<name>`
    once; changing the mode rewrites only these generated files."""
    trigger = trigger or load()["trigger"]
    skill_root = Path(skill_root).resolve()
    out = []
    for name in GATED:
        text = (skill_root / name / "SKILL.md").read_text()
        _, front, body = text.split("---", 2)
        lines = front.strip("\n").splitlines()
        if trigger != "auto":
            explicit = next(ln for ln in lines if ln.strip().startswith("explicit_description:"))
            desc = "description:" + explicit.split(":", 1)[1]
            lines = [desc if ln.startswith("description:") else ln for ln in lines]
        body = body.replace('$(realpath "${CLAUDE_SKILL_DIR}/..")', str(skill_root))
        dst = rendered_dir(skill_root) / name / "SKILL.md"
        dst.parent.mkdir(parents=True, exist_ok=True)
        header = f"<!-- generated by astk mode ({trigger}) from {skill_root / name / 'SKILL.md'}; edit that file -->"
        dst.write_text("---\n" + "\n".join(lines) + f"\n---\n{header}{body}")
        out.append(dst)
    return out
