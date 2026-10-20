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
SYNONYMS = {
    "部署": "deploy serve serving launch",
    "推理": "inference serving",
    "性能": "performance perf throughput latency",
    "优化": "optimize optimise optimization tuning",
    "调优": "tuning optimize",
    "算子": "kernel operator op",
    "内核": "kernel",
    "压测": "bench benchmark load",
    "基准": "bench benchmark baseline",
    "吞吐": "throughput",
    "延迟": "latency ttft tpot itl",
    "首字": "ttft",
    "分离": "disaggregation disagg pd prefill decode",
    "预填充": "prefill",
    "解码": "decode",
    "量化": "quantization quant fp8 fp4 int8",
    "调研": "research survey",
    "环境": "environment env",
    "检查": "check verify",
    "机器": "machine node host",
    "集群": "cluster slurm",
    "采样": "sampler sampling",
    "注意力": "attention",
    "专家": "moe expert",
    "并行": "parallel tp ep dp pp",
    "投机": "speculative mtp eagle",
    "缓存": "cache kv",
    "精度": "accuracy precision",
    "回归": "regression",
    "剖析": "profiling profile trace",
    "性能分析": "profiling profile trace",
    "追踪": "trace tracing",
    "多机": "multi-node rdma",
    "模型": "model",
}


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
    out = []
    for main in sorted((AGENT_SYS / "examples").glob("*/main.yaml")) + sorted(
        (AGENT_SYS / "examples").glob("*/*/main.yaml")
    ):
        p = main.parent
        if (p / "assets").is_dir():
            readme = p / "README.md"
            summary = (
                readme.read_text(errors="replace").splitlines()[0].lstrip("# ")
                if readme.is_file()
                else ""
            )
            out.append(
                Item(
                    type="task",
                    tier="local",
                    user=None,
                    name=p.name,
                    path=str(p),
                    meta={"summary": summary, "sealed": None},
                )
            )
    return out


def item_text(item: Item) -> str:
    """The searchable text. META is indexed as values (summary, tags, context),
    not as raw JSON, so snippets stay readable."""
    p = Path(item.path)
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
        for f in (
            p / "README.md",
            p / "assets" / "main.task" / "readme.md",
            p / "plan.task.yaml.txt",
        ):
            if f.is_file():
                parts.append(f.read_text(errors="replace")[:TEXT_CAP])
        for f in sorted(p.glob("assets/*.task/readme.md"))[:20]:
            parts.append(f.read_text(errors="replace")[:1500])
    else:
        content = p / "content"
        if (content / "README.md").is_file():
            parts.append((content / "README.md").read_text(errors="replace")[:TEXT_CAP])
        items_dir = content / "items"
        if items_dir.is_dir():
            parts += [x.name for x in sorted(items_dir.rglob("*"))[:200]]
            for f in sorted(items_dir.rglob("*.md"))[:5]:
                parts.append(f.read_text(errors="replace")[:4000])
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
