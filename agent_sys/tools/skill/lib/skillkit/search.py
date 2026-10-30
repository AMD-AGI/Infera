"""Find tasks and handoffs in the knowledge repo that may help with a request.

Deliberately simple and explainable: each item is indexed by the text a
person would read first (its META summary and tags, the task readmes, the
handoff README, item names), and scored by weighted term overlap with the
query — English words, and character bigrams for Chinese. The ranking only
decides what Claude reads; Claude then reads the top items and judges
relevance itself. The reason for each hit (matched terms + a snippet) is
returned so that judgement can be checked.

Tier weights: release (reviewed) > staging (self-reviewed) > draft.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from skillkit import AGENT_SYS
from skillkit.remote import Config, Item, items

TIER_WEIGHT = {"release": 1.5, "staging": 1.2, "local": 1.1, "draft": 1.0}
TEXT_CAP = 20000
_WORD = re.compile(r"[a-z0-9][a-z0-9.+-]*[a-z0-9]|[a-z0-9]")
_CJK = re.compile(r"[一-鿿]+")
_STOP = frozenset(
    "the a an and or of to in on for with is are be by as at it this that from use using how what "
    "task handoff readme md items content".split()
)


#: Chinese domain words -> the English the repo's artefacts are written in.
#: Applied to the QUERY only, so a Chinese request finds English handoffs.
# fmt: off
SYNONYMS = {
    "部署": "deploy serve serving launch", "推理": "inference serving",
    "性能": "performance perf throughput latency", "优化": "optimize optimise optimization tuning",
    "调优": "tuning optimize", "算子": "kernel operator op", "内核": "kernel",
    "压测": "bench benchmark load", "基准": "bench benchmark baseline", "吞吐": "throughput",
    "延迟": "latency ttft tpot itl", "首字": "ttft", "预填充": "prefill", "解码": "decode",
    "分离": "disaggregation disagg pd prefill decode", "量化": "quantization quant fp8 fp4 int8",
    "调研": "research survey", "环境": "environment env", "检查": "check verify",
    "机器": "machine node host", "集群": "cluster slurm", "采样": "sampler sampling",
    "注意力": "attention", "专家": "moe expert", "并行": "parallel tp ep dp pp",
    "投机": "speculative mtp eagle", "缓存": "cache kv", "精度": "accuracy precision",
    "回归": "regression", "剖析": "profiling profile trace", "性能分析": "profiling profile trace",
    "追踪": "trace tracing", "多机": "multi-node rdma", "模型": "model",
}
# fmt: on


def expand(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in query]
    return " ".join([query, *extra])


def tokens(text: str) -> list[str]:
    text = text.lower()
    out = []
    for w in _WORD.findall(text):
        parts = [w, *re.split(r"[._+-]", w)] if re.search(r"[._+-]", w) else [w]
        out += [p for p in parts if p and p not in _STOP and len(p) > 1]
    for run in _CJK.findall(text):
        out += [run[i : i + 2] for i in range(len(run) - 1)] or [run]
    return out


def local_items() -> list[Item]:
    """Task packages in this agent_sys checkout's `examples/` — tier `local`."""
    examples, out = AGENT_SYS / "examples", []
    for main in sorted(examples.glob("*/main.yaml")) + sorted(examples.glob("*/*/main.yaml")):
        p = main.parent
        if not (p / "assets").is_dir():
            continue
        readme = p / "README.md"
        head = readme.read_text(errors="replace").splitlines()[:1] if readme.is_file() else []
        meta = {"summary": head[0].lstrip("# ") if head else "", "sealed": None}
        out.append(Item("task", "local", None, p.name, str(p), meta))
    return out


def _read(f: Path, root: Path, cap: int) -> str:
    """At most `cap` chars of `f` — only if it is a regular file that resolves
    inside `root`. A remote item is anybody's push: a symlinked README must not
    read the searcher's own files (or block on a device) into the snippets."""
    try:
        if not f.is_file() or not f.resolve().is_relative_to(root):
            return ""
        with f.open("rb") as fh:
            return fh.read(cap * 4).decode(errors="replace")[:cap]
    except OSError:
        return ""


def _walk(top: Path) -> list[Path]:
    """Every path under `top`, never descending into a symlinked directory."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(top):  # followlinks=False
        out += [Path(dirpath) / n for n in dirnames + filenames]
    return sorted(out)


def item_text(item: Item) -> str:
    """The searchable text. META is indexed as values (summary, tags, context),
    not as raw JSON, so snippets stay readable."""
    p = Path(item.path)
    # Local packages share leaves by symlink inside this checkout; a remote item
    # is read only within its own directory.
    root = (AGENT_SYS if item.tier == "local" else p).resolve()
    meta = item.meta
    ctx = meta.get("context") or {}
    parts = [
        item.name.replace("_", " ").replace("-", " "),
        str(meta.get("summary") or ""),
        " ".join(map(str, meta.get("tags") or [])),
        " ".join(f"{k} {v}" for k, v in ctx.items()),
        str(meta.get("handoff_kind") or ""),
    ]
    if item.type == "task":
        # What the package is about: its README, main readme and plan in full; each
        # leaf readme's opening only (packages share leaves by symlink, and full
        # leaf briefs would make every e2e package look like every other).
        for f in ("README.md", "assets/main.task/readme.md", "plan.task.yaml.txt"):
            parts.append(_read(p / f, root, TEXT_CAP))
        parts += [_read(f, root, 1500) for f in sorted(p.glob("assets/*.task/readme.md"))[:20]]
    else:
        content = p / "content"
        parts.append(_read(content / "README.md", root, TEXT_CAP))
        items_dir = content / "items"
        if items_dir.is_dir() and items_dir.resolve().is_relative_to(root):
            found = _walk(items_dir)
            parts += [x.name for x in found[:200]]
            parts += [_read(f, root, 4000) for f in [x for x in found if x.suffix == ".md"][:5]]
    return "\n".join(parts)


@dataclass
class Hit:
    score: float
    type: str
    tier: str
    user: str | None
    name: str
    path: str
    summary: str
    matched: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    weak: bool = False
    sealed: bool | None = None
    context: dict = field(default_factory=dict)
    created: str = ""


@dataclass
class Result:
    searched: int
    hits: list[Hit]


def search(
    cfg: Config | None,
    query: str,
    *,
    limit: int = 10,
    types: tuple[str, ...] = ("task", "handoff"),
    local: bool = True,
) -> Result:
    q = Counter(tokens(expand(query)))
    pool = (items(cfg) if cfg is not None else []) + (local_items() if local else [])
    docs = [(it, item_text(it)) for it in pool if it.type in types]
    if not q or not docs:
        return Result(searched=len(docs), hits=[])
    tf = [Counter(tokens(text)) for _, text in docs]
    n = len(docs)
    df = Counter(t for c in tf for t in set(c))
    lengths = [sum(c.values()) or 1 for c in tf]
    avg = sum(lengths) / n
    k1, b = 1.2, 0.75  # BM25: a long package does not win by being long
    hits = []
    for (it, text), c, length in zip(docs, tf, lengths):
        matched = [t for t in q if c.get(t)]
        if not matched:
            continue
        score = sum(
            math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            * c[t]
            * (k1 + 1)
            / (c[t] + k1 * (1 - b + b * length / avg))
            * q[t]
            for t in matched
        )
        coverage = len(matched) / len(q)
        score *= TIER_WEIGHT.get(it.tier, 1.0) * (0.5 + 0.5 * coverage)
        hits.append(
            Hit(
                score=round(score, 3),
                type=it.type,
                tier=it.tier,
                user=it.user,
                name=it.name,
                path=it.path,
                summary=str(it.meta.get("summary") or ""),
                matched=sorted(matched, key=lambda t: -c[t])[:12],
                snippets=_snippets(text, matched),
                weak=coverage < 0.25 and len(matched) < 3,
                sealed=it.meta.get("sealed"),
                context=dict(it.meta.get("context") or {}),
                created=str(it.meta.get("created") or ""),
            )
        )
    hits.sort(key=lambda h: -h.score)
    return Result(searched=len(docs), hits=hits[:limit])


def _snippets(text: str, matched: list[str], k: int = 3) -> list[str]:
    out = []
    for line in text.splitlines():
        low = line.lower()
        if any(t in low for t in matched) and line.strip():
            out.append(line.strip()[:200])
            if len(out) == k:
                break
    return out
