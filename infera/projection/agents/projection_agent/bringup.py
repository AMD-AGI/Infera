###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Op bring-up: turn one op the projection does not model into a validated op model.

New models keep shipping ops the inference projection's built-in profilers do
not know (a new mixer, a new gate, a new cache layout).  The projection
reports each one as an ``INFERASIM_OP_GAP`` marker; this module turns a gap into
a validated op model under
``infera/projection/core/projection/op_models/generated/`` in four steps:

1. **Context** -- the gap and its config keys, the model YAML chain (comments
   included), the HF ``config.json``, any paper text given, the op-model API,
   and the existing op models / profilers closest to the gap's slot.  Written
   to ``<out>/op_bringup/<op_id>/context.md``.
2. **Create** -- a coder writes one op-model module from that context: an LLM
   through DSPy / LiteLLM, an external coding agent (``--op-coder-cmd``), or
   none (the context pack is the deliverable, for a human or an IDE agent).
3. **Validate** -- :mod:`infera.projection.core.projection.op_models.validate`
   in a subprocess, on prefill and decode steps of the workload: roofline
   floor, closed-form FLOPs, monotonicity in batch / prompt / context, key
   coverage.  Failures go back to the coder as feedback, up to
   ``max_attempts`` times.
4. **Register** -- the passing module is copied into ``generated/`` (the
   projection discovers it on its next run) with a regression test.  A coder
   may instead judge the op negligible and *acknowledge* its keys, with the
   estimate, in ``generated/acknowledged.yaml``.

The projection agent (:class:`~infera.projection.agents.projection_agent.ProjectionAgent`)
drives this for every gap of a workload.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

# <repo>/infera/projection/agents/tuning_agent/op_bringup.py
REPO_ROOT = Path(__file__).resolve().parents[4]

# Section caps keep the context inside a coder's window.
_MAX_SECTION = 40_000
_MAX_PAPER = 60_000
_HF_URL = "https://huggingface.co/{repo}/raw/main/config.json"
_ACK_PREFIX = "ACKNOWLEDGE"

_GUIDE = """\
You are adding an analytical performance model for one op to the InferaSim
inference projection.  The projection prices a prefill or decode step layer by
layer on a simulated GPU; this op is currently priced as whatever built-in
block it resembles, which is wrong.

Write ONE Python module that defines and registers ONE `OpModel` subclass.

`cost(ctx)` prices ONE step of ONE layer on ONE rank and must handle both
phases: a prefill (`ctx.q_len` new tokens per sequence over `ctx.kv_len` of
context) and a decode (`ctx.is_decode`, `q_len` usually 1).  Decode is weight-
and cache-bound: count HBM bytes as carefully as FLOPs.

How to price things -- use the same primitives the built-in profilers use, via
the `OpContext` you are given (never hard-code hardware numbers):
  * every projection: `ctx.linear(n_full, k_full, shard="column"|"row")` with
    the FULL widths (TP sharding is applied for you; M is `ctx.tokens`, this
    rank's new tokens), and `ctx.linear_bytes(...)` for its HBM traffic;
    other dense matmuls: `ctx.gemm(m, n, k)`;
  * a softmax-attention core (any new attention that is still softmax
    attention -- sliding window, sparse, latent, sink, ...): `ctx.sdpa(...)`,
    the FAv3 tile-level simulator, with the heads on this rank
    (`ctx.per_rank(heads)`) and the step's `q_len` / `kv_len`;
  * any other fused kernel (chunked linear-attention recurrences, scans,
    blockwise scores): describe the matmuls ONE workgroup runs as `(m, n, k)`
    tiles and the number of workgroups, and call `ctx.fused_tiles(tiles, n_wg)`
    -- it prices the tile program on one CU and multiplies by the waves, the
    same construction FAv3 uses.  Fold a loop a workgroup runs over the
    sequence into one tile dimension, e.g. `(n_chunks * C, n, k)`;
  * elementwise ops, norms, gathers, tiny convolutions, KV / state reads and
    writes: count the HBM bytes and use `ctx.stream_ms(bytes)`;
  * a memory-bound fused kernel takes `max(compute_ms, ctx.stream_ms(bytes))`.

Rules the validator enforces:
  * `name` snake_case; `slot` one of attention | attention_extra | mlp |
    layer_extra (attention / mlp REPLACE the built-in block on
    `layer_fraction` of layers; *_extra ADD to it);
  * `claims` maps the selector key(s) to the value(s) that turn the op on;
    every gap key is in `claims`, `consumes`, or `ignores` (key -> reason it
    does not change cost);
  * `references` cites the paper / model card / kernel source;
  * cost() returns an `OpCost` with finite, positive ms, FLOPs and HBM bytes
    per rank per layer, and a `breakdown` whose entries sum to `ms`; time
    never beats the roofline max(FLOPs/peak, bytes/BW);
  * `reference_flops()` returns the forward FLOPs counted straight from the
    paper's equations, written independently of cost(); it must agree with
    cost()'s FLOPs within 25 %;
  * cost does not fall with more batch, longer prompts or longer context;
  * when the op changes the serving cache, set `kv_bytes_per_token` (per
    layer per rank; 0 when the layer keeps no KV cache) and / or
    `state_bytes_per_seq` (fixed-size recurrent or conv state per sequence).

If the op is genuinely negligible (< ~1 % of its layer -- a scalar gate, a
routing constraint, a numerics-only flag), do not write a model.  Reply with
lines of the form

    ACKNOWLEDGE <config_key>: <why it does not matter, with your estimate>

for each gap key, and nothing else.

Otherwise reply with the complete module in a single ```python fenced block.
Model it the way the existing implementations below are modeled.
"""

_TEMPLATE = '''\
"""<Op name> (<model family>), inference.

<The op's equations, from the paper, how each kernel is priced, and what the
op keeps in the serving cache.>
"""

from __future__ import annotations

from infera.projection.core.projection.op_models.base import (
    SLOT_ATTENTION,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
    interleave_fraction,
)
from infera.projection.core.projection.op_models.registry import register_op_model


@register_op_model
class MyOp(OpModel):
    name = "my_op"
    slot = SLOT_ATTENTION
    claims = {"selector_key": ("value",)}
    consumes = ("param_key_a", "param_key_b")
    ignores = {}
    references = ("Paper title. arXiv:XXXX.XXXXX",)

    def layer_fraction(self, model_config) -> float:
        return 1.0

    def cost(self, ctx: OpContext) -> OpCost:
        ...
        return OpCost(ms=..., flops=..., bytes=..., breakdown={...},
                      kv_bytes_per_token=None, state_bytes_per_seq=None)

    def reference_flops(self, ctx: OpContext):
        return flops
'''


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BringupResult:
    op_id: str
    status: str  # registered | acknowledged | failed | context_only | skipped
    message: str = ""
    op_name: str | None = None
    module_path: str | None = None
    test_path: str | None = None
    context_path: str | None = None
    attempts: int = 0
    validation: dict = field(default_factory=dict)
    acknowledged: dict = field(default_factory=dict)

    @property
    def changed_projection(self) -> bool:
        return self.status in ("registered", "acknowledged")

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _op_models_dir(repo_root: Path) -> Path:
    return repo_root / "infera" / "projection" / "core" / "projection" / "op_models"


def _clip(text: str, cap: int = _MAX_SECTION) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [{len(text) - cap} more characters omitted]\n"


def _yaml_chain_text(model_path: Path, seen: set | None = None) -> list[tuple]:
    """The model YAML and every file it ``extends`` / ``bases``, comments kept."""
    seen = seen if seen is not None else set()
    model_path = model_path.resolve()
    if model_path in seen or not model_path.is_file():
        return []
    seen.add(model_path)
    text = model_path.read_text()
    out = [(model_path, text)]
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    parents = data.get("extends") or data.get("bases") or []
    if isinstance(parents, str):
        parents = [parents]
    for p in parents:
        out.extend(_yaml_chain_text(model_path.parent / str(p), seen))
    return out


def fetch_hf_config(
    spec: str | None, tokenizer_model: str | None = None, timeout: float = 10.0
) -> str | None:
    """``config.json`` text from a path, a URL, or a HF repo id; ``None`` on failure."""
    for cand in (spec, tokenizer_model):
        if not cand:
            continue
        cand = str(cand)
        p = Path(cand).expanduser()
        if p.is_file():
            return p.read_text()
        url = cand if cand.startswith(("http://", "https://")) else None
        if url is None and re.match(r"^[\w.\-]+/[\w.\-]+$", cand):
            url = _HF_URL.format(repo=cand)
        if url is None:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "inferasim-op-bringup"})
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def _read_paper(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.is_file():
        if p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # optional

                return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
            except Exception:
                return f"(PDF {p} could not be converted to text; pass a .txt / .md export)"
        return p.read_text(errors="replace")
    if spec.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(spec, timeout=15) as resp:
                body = resp.read()
            if body[:4] == b"%PDF":
                return f"(URL {spec} is a PDF; pass a text export with --paper)"
            return body.decode("utf-8", "replace")
        except Exception as exc:
            return f"(could not fetch {spec}: {exc})"
    return spec  # inline text


def _source_of(obj_path: str) -> str:
    """Source of ``module:attr`` (or the whole module), best effort."""
    import importlib

    mod_name, _, attr = obj_path.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        obj = mod
        for part in filter(None, attr.split(".")):
            obj = getattr(obj, part)
        return inspect.getsource(obj)
    except Exception as exc:
        return f"# (source of {obj_path} unavailable: {exc})"


_MP = "infera.projection.core.projection"
# Built-in code each slot's new op should be modeled after.
_REFERENCES_BY_SLOT = {
    "attention": (
        f"{_MP}.module_profilers.attention:AttentionProfiler._get_simulated_results",
        f"{_MP}.module_profilers.linear_attention:LinearAttentionProfiler",
        f"{_MP}.simulation_backends.sdpa_simulator:SDPASimulator._simulate_tile_level",
    ),
    "attention_extra": (f"{_MP}.module_profilers.attention:AttentionProfiler._output_gate_ms",),
    "mlp": (f"{_MP}.module_profilers.moe_mlp", f"{_MP}.module_profilers.dense_mlp"),
    "layer_extra": (f"{_MP}.module_profilers.transformer_layer",),
}


def _existing_op_models(repo_root: Path, slot: str) -> list[Path]:
    gen = _op_models_dir(repo_root) / "generated"
    files = [p for p in sorted(gen.glob("*.py")) if not p.name.startswith("_")]
    same = [p for p in files if re.search(rf"slot\s*=\s*SLOT_{slot.upper()}\b", p.read_text())]
    return same + [p for p in files if p not in same]


def build_context(
    gap,
    *,
    repo_root: Path = REPO_ROOT,
    model_path: Path | None = None,
    raw_model: dict | None = None,
    hf_config: str | None = None,
    papers: Sequence[str] = (),
) -> str:
    """The context pack for one gap, as markdown."""
    sec: list[str] = [f"# Op bring-up: `{gap.op_id}`\n", _GUIDE]
    sec.append(
        "## The gap\n\n"
        f"- op id: `{gap.op_id}`\n- slot (best guess): `{gap.slot}`\n- severity: `{gap.severity}`\n"
        f"- why: {gap.reason}\n- config keys:\n\n```json\n{json.dumps(gap.keys, indent=2, default=str)}\n```\n"
    )
    if model_path is not None:
        chain = _yaml_chain_text(Path(model_path))
        body = "\n".join(f"### `{p.name}`\n\n```yaml\n{t}\n```\n" for p, t in chain)
        sec.append("## Model config (YAML chain, comments included)\n\n" + _clip(body))
    elif raw_model:
        sec.append(
            "## Model config (resolved)\n\n```json\n"
            + _clip(json.dumps(raw_model, indent=2, default=str))
            + "\n```\n"
        )
    tok = (raw_model or {}).get("tokenizer_model")
    hf = fetch_hf_config(hf_config, tok)
    if hf:
        sec.append(f"## HF config.json ({hf_config or tok})\n\n```json\n{_clip(hf, 20_000)}\n```\n")
    else:
        sec.append("## HF config.json\n\n(not available; rely on the YAML above)\n")
    for i, spec in enumerate(papers, 1):
        sec.append(
            f"## Paper {i}: {spec if len(spec) < 200 else spec[:200] + '...'}\n\n"
            + _clip(_read_paper(spec), _MAX_PAPER)
        )

    od = _op_models_dir(repo_root)
    sec.append(
        "## Op-model API (`op_models/base.py`)\n\n```python\n"
        + (od / "base.py").read_text()
        + "\n```\n"
    )
    for p in _existing_op_models(repo_root, gap.slot)[:3]:
        sec.append(
            f"## Existing op model: `generated/{p.name}`\n\n```python\n{_clip(p.read_text())}\n```\n"
        )
    for ref in _REFERENCES_BY_SLOT.get(gap.slot, ()):
        sec.append(f"## Built-in reference: `{ref}`\n\n```python\n{_clip(_source_of(ref))}\n```\n")
    sec.append("## Module template\n\n```python\n" + _TEMPLATE + "```\n")
    return "\n".join(sec)


# ---------------------------------------------------------------------------
# Coders
# ---------------------------------------------------------------------------

Coder = Callable[[str, Path], str]
"""``coder(prompt, target_file) -> reply``: the module source or ACKNOWLEDGE lines."""


def extract_module(reply: str) -> str | None:
    """Python source from a coder reply (fenced block, or the bare module)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply or "", flags=re.S)
    blocks = [b for b in blocks if "OpModel" in b]
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "class " in (reply or "") and "OpModel" in reply:
        return reply.strip() + "\n"
    return None


def parse_acknowledgements(reply: str) -> dict:
    out = {}
    for line in (reply or "").splitlines():
        m = re.match(rf"^\s*{_ACK_PREFIX}\s+([A-Za-z_][\w.]*)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def llm_model_from_env() -> str | None:
    return os.getenv("INFERASIM_OP_LLM_MODEL") or os.getenv("LLM_MODEL")


def dspy_coder(model: str | None = None, *, timeout: int = 600, max_tokens: int = 16000) -> Coder:
    """Coder backed by an LLM through DSPy / LiteLLM.

    Uses the LM already configured in ``dspy.settings`` (the tuning agent's),
    else builds one from *model*, ``$INFERASIM_OP_LLM_MODEL`` or ``$LLM_MODEL``
    -- a LiteLLM model string such as ``anthropic/claude-opus-4-5``, with the
    provider's usual key variable (or ``$LLM_API_KEY`` / ``$OPENAI_API_BASE``).
    """
    import dspy

    lm = getattr(dspy.settings, "lm", None)
    if lm is None:
        model = model or llm_model_from_env()
        if not model:
            raise RuntimeError("no LLM configured: pass --op-llm-model or set LLM_MODEL")
        kw: dict = {"cache": False, "timeout": timeout, "max_tokens": max_tokens}
        if os.getenv("LLM_API_KEY"):
            kw["api_key"] = os.environ["LLM_API_KEY"]
        if os.getenv("OPENAI_API_BASE"):
            kw["api_base"] = os.environ["OPENAI_API_BASE"]
        lm = dspy.LM(model, **kw)

    def _code(prompt: str, target: Path) -> str:
        out = lm(messages=[{"role": "user", "content": prompt}])
        first = out[0] if isinstance(out, list) and out else out
        return first.get("text", "") if isinstance(first, dict) else str(first)

    return _code


def command_coder(cmd_template: str, timeout: int = 1800) -> Coder:
    """Coder that shells out to a coding agent.

    ``cmd_template`` may use ``{prompt_file}``, ``{target_file}`` and
    ``{op_models_dir}``.  The agent either writes ``{target_file}`` or prints
    the module (or ACKNOWLEDGE lines) on stdout.
    """

    def _code(prompt: str, target: Path) -> str:
        prompt_file = target.with_suffix(".prompt.md")
        prompt_file.write_text(prompt)
        if target.exists():
            target.unlink()
        cmd = cmd_template.format(
            prompt_file=shlex.quote(str(prompt_file)),
            target_file=shlex.quote(str(target)),
            op_models_dir=shlex.quote(str(target.parent)),
        )
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if target.is_file() and target.stat().st_size > 0:
            return "```python\n" + target.read_text() + "\n```"
        return proc.stdout or ""

    return _code


def make_coder(kind: str, *, cmd: str | None = None, model: str | None = None) -> Coder | None:
    if kind == "none":
        return None
    if kind == "command":
        if not cmd:
            raise ValueError("--op-coder command needs --op-coder-cmd")
        return command_coder(cmd)
    if kind == "dspy":
        return dspy_coder(model)
    raise ValueError(f"unknown coder {kind!r}")


# ---------------------------------------------------------------------------
# Validate / register
# ---------------------------------------------------------------------------


def _env(repo_root: Path, gpu_arch: str | None, env: dict | None) -> dict:
    e = dict(env or os.environ)
    parts = [p for p in e.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p) != repo_root]
    e["PYTHONPATH"] = os.pathsep.join([str(repo_root), *parts])
    e.setdefault("NNODES", "1")
    e.setdefault("GPUS_PER_NODE", "8")
    if gpu_arch:
        e["INFERASIM_GPU_ARCH"] = gpu_arch
    e["INFERASIM_OP_COVERAGE"] = "0"
    return e


def run_validator(
    module: Path,
    *,
    workload: Path,
    repo_root: Path = REPO_ROOT,
    gap_keys: Iterable[str] = (),
    gpu_arch: str | None = None,
    env: dict | None = None,
    python: str = sys.executable,
    timeout: int = 600,
) -> dict:
    """Validator report as a dict; runs out of process so a broken module cannot hurt the caller."""
    cmd = [
        python,
        "-m",
        "infera.projection.core.projection.op_models.validate",
        "--config",
        str(workload),
        "--op-file",
        str(module),
        "--gap-keys",
        ",".join(gap_keys),
        "--json",
    ]
    if gpu_arch:
        cmd += ["--gpu-arch", gpu_arch]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=_env(repo_root, gpu_arch, env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout or ""
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "checks": [
                {
                    "name": "load",
                    "status": "fail",
                    "detail": f"validator timed out after {timeout}s",
                }
            ],
            "cases": [],
        }
    start = out.find("{\n")
    try:
        return json.loads(out[start:]) if start >= 0 else json.loads(out)
    except ValueError:
        return {
            "ok": False,
            "checks": [{"name": "load", "status": "fail", "detail": out[-3000:]}],
            "cases": [],
        }


def _case_line(case: dict) -> str:
    return (
        f"{case['label']} ({case['phase']}, B={case['batch']}, q={case['q_len']}, kv={case['kv_len']}): "
        f"{case['ms']:.4f} ms, roofline {case['roofline_ms']:.4f} ms, "
        f"flops={case['flops']:.3e}, bytes={case['bytes']:.3e}"
    )


def validation_feedback(report: dict) -> str:
    lines = [
        f"Your module FAILED validation (op={report.get('op')}). Fix these and return the whole module again:"
    ]
    for c in report.get("checks", []):
        if c.get("status") != "pass":
            lines.append(f"- [{c['status'].upper()}] {c['name']}: {c.get('detail', '')}")
    lines += ["  case " + _case_line(case) for case in report.get("cases", [])]
    return "\n".join(lines)


_TEST_TEMPLATE = '''\
"""Regression test for the generated op model ``{name}`` (written by op bring-up)."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from infera.projection.core.projection.op_models.registry import registered_op_models
from infera.projection.core.projection.op_models.validate import load_workload, validate_op_model

_ROOT = Path(__file__).resolve().parents[{depth}]
_WORKLOAD = _ROOT / {workload}
_ENV = {env}
_GAP_KEYS = {gap_keys}


@pytest.mark.skipif(not _WORKLOAD.is_file(), reason="workload YAML not in this checkout")
def test_{name}_validates(monkeypatch):
    for k, v in {{"NNODES": "1", "GPUS_PER_NODE": "8", "INFERASIM_OP_COVERAGE": "0", **_ENV}}.items():
        monkeypatch.setenv(k, v)
    ops = {{c.name: c for c in registered_op_models()}}
    assert "{name}" in ops
    wl = load_workload(str(_WORKLOAD), simulate_backends=False)
    rep = validate_op_model(ops["{name}"], wl, gap_keys=_GAP_KEYS)
    assert rep.ok, rep.feedback()
'''


def _template_env(env: dict | None, workload: Path) -> dict:
    """The ``INFERASIM_*`` variables a templated workload (``${INFERASIM_MODEL:...}``) was resolved with."""
    src = env if env is not None else os.environ
    text = workload.read_text() if workload.is_file() else ""
    return {k: src[k] for k in ("INFERASIM_MODEL",) if src.get(k) and "${" + k in text}


def _write_test(
    repo_root: Path, name: str, workload: Path, gap_keys: Sequence[str], env: dict | None = None
) -> Path:
    tdir = repo_root / "tests" / "unit" / "projection" / "op_models"
    tdir.mkdir(parents=True, exist_ok=True)
    init = tdir / "__init__.py"
    if not init.exists():
        init.write_text("")
    try:
        rel = str(workload.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        rel = str(workload.resolve())
    depth = len(tdir.relative_to(repo_root).parts)
    path = tdir / f"test_generated_{name}.py"
    path.write_text(
        _TEST_TEMPLATE.format(
            name=name,
            workload=json.dumps(rel),
            gap_keys=json.dumps(list(gap_keys)),
            depth=depth,
            env=json.dumps(_template_env(env, workload)),
        )
    )
    return path


def acknowledge_keys(repo_root: Path, entries: dict) -> Path:
    path = _op_models_dir(repo_root) / "generated" / "acknowledged.yaml"
    lines = [] if not path.exists() else [path.read_text().rstrip("\n"), ""]
    for k, why in entries.items():
        why = " ".join(str(why).split())
        lines.append(f"{k}: >-\n" + textwrap.indent(textwrap.fill(why, 86), "  "))
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def bring_up_op(
    gap,
    *,
    workload: Path,
    out_dir: Path,
    coder: Coder | None,
    repo_root: Path = REPO_ROOT,
    gpu_arch: str | None = None,
    model_path: Path | None = None,
    raw_model: dict | None = None,
    hf_config: str | None = None,
    papers: Sequence[str] = (),
    max_attempts: int = 3,
    register: bool = True,
    env: dict | None = None,
    log: Callable[[str], None] = print,
) -> BringupResult:
    """Context -> Create -> Validate -> Register for one gap."""
    if model_path is not None:
        # Validate on the model the gap came from, not whatever ``INFERASIM_MODEL``
        # a templated workload would otherwise pick up from the environment.
        env = {**(env if env is not None else os.environ), "INFERASIM_MODEL": Path(model_path).stem}
    work = Path(out_dir) / "op_bringup" / gap.op_id
    work.mkdir(parents=True, exist_ok=True)
    context = build_context(
        gap,
        repo_root=repo_root,
        model_path=model_path,
        raw_model=raw_model,
        hf_config=hf_config,
        papers=papers,
    )
    ctx_path = work / "context.md"
    ctx_path.write_text(context)
    res = BringupResult(gap.op_id, "context_only", context_path=str(ctx_path))
    log(f"[op-bringup] {gap.op_id}: context pack {ctx_path} ({len(context) // 1000} kB)")
    if coder is None:
        res.message = (
            "no coder configured; hand context.md to a coding agent, then re-run the projection"
        )
        return _finish(res, work)

    feedback = ""
    gap_keys = list(gap.keys)
    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        target = work / f"attempt_{attempt}.py"
        prompt = context + (
            "\n\n## Feedback on your previous attempt\n\n" + feedback if feedback else ""
        )
        t0 = time.time()
        try:
            reply = coder(prompt, target)
        except Exception as exc:
            res.status, res.message = "failed", f"coder failed: {type(exc).__name__}: {exc}"
            break
        (work / f"attempt_{attempt}.reply.md").write_text(reply or "")
        log(f"[op-bringup] {gap.op_id}: attempt {attempt} reply in {time.time() - t0:.0f}s")

        acks = parse_acknowledgements(reply)
        source = extract_module(reply)
        if acks and source is None:
            unknown = sorted(set(acks) - set(gap_keys))
            missing = sorted(set(gap_keys) - set(acks))
            if unknown or missing:
                feedback = (
                    f"ACKNOWLEDGE lines must cover exactly the gap keys {gap_keys}; "
                    f"unknown: {unknown}, missing: {missing}."
                )
                continue
            res.acknowledged = acks
            if register:
                acknowledge_keys(repo_root, acks)
            res.status, res.message = (
                "acknowledged",
                "judged negligible; keys added to generated/acknowledged.yaml",
            )
            break
        if source is None:
            feedback = "Your reply contained no ```python module with an OpModel subclass and no ACKNOWLEDGE lines."
            continue
        target.write_text(source)
        report = run_validator(
            target,
            workload=workload,
            repo_root=repo_root,
            gap_keys=gap_keys,
            gpu_arch=gpu_arch,
            env=env,
        )
        res.validation = report
        (work / f"attempt_{attempt}.validation.json").write_text(json.dumps(report, indent=2))
        if not report.get("ok"):
            feedback = validation_feedback(report)
            log(f"[op-bringup] {gap.op_id}: attempt {attempt} failed validation")
            continue
        name = str(report.get("op") or gap.op_id)
        res.op_name = name
        if register:
            dest = _op_models_dir(repo_root) / "generated" / f"{name}.py"
            if dest.exists():
                shutil.copy2(dest, work / f"previous_{name}.py")
            shutil.copy2(target, dest)
            res.module_path = str(dest)
            res.test_path = str(_write_test(repo_root, name, workload, gap_keys, env))
        else:
            res.module_path = str(target)
        res.status, res.message = "registered", f"validated on attempt {attempt}"
        break
    else:
        res.status, res.message = "failed", f"no valid module after {max_attempts} attempts"
    log(f"[op-bringup] {gap.op_id}: {res.status} -- {res.message}")
    return _finish(res, work)


def _finish(res: BringupResult, work: Path) -> BringupResult:
    (work / "summary.json").write_text(json.dumps(res.as_dict(), indent=2, default=str))
    md = [f"# Op bring-up `{res.op_id}`: {res.status}", "", res.message, ""]
    if res.module_path:
        md.append(f"- module: `{res.module_path}`")
    if res.test_path:
        md.append(f"- test: `{res.test_path}`")
    if res.acknowledged:
        md += [f"- acknowledged `{k}`: {v}" for k, v in res.acknowledged.items()]
    md += ["- " + _case_line(case) for case in (res.validation or {}).get("cases", [])]
    (work / "report.md").write_text("\n".join(md) + "\n")
    return res


def bring_up_gaps(gaps: Sequence, **kwargs) -> list[BringupResult]:
    """Bring up every blocking gap, de-duplicated by op id."""
    out, seen = [], set()
    for g in gaps:
        if g.severity != "gap" or g.op_id in seen:
            continue
        seen.add(g.op_id)
        out.append(bring_up_op(g, **kwargs))
    return out
