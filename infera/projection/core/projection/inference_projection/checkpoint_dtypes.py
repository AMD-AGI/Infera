###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The weight format each op runs at, read from the checkpoint itself.

A precision label ("mxfp4", "fp8") names a checkpoint, not its ops: gpt-oss
ships mxfp4 experts behind bf16 attention and a bf16 router, and DeepSeek-V4
ships fp8 attention with bf16 compressors, fp32 hyper-connection mixers and
fp4 experts. Pricing every op at the label's width is wrong in both. The
safetensors headers carry the stored dtype of every tensor, so this reads them
(the 8-byte length and the JSON header, never the weights) and reports one
format per op role.

    python -m infera.projection.core.projection.inference_projection.checkpoint_dtypes \\
        /models/gpt-oss-120b --out gpt-oss-120b.dtypes.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

SCHEMA = "inferasim.checkpoint_dtypes"

# Ordered: the first pattern a tensor name matches decides its role.
ROLE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (role, re.compile(p))
    for role, p in (
        ("mixer", r"(^|\.)hc_\w*_fn$"),
        ("compressor", r"\.compressor\."),
        ("indexer_weights", r"\.indexer\.weights_proj\."),
        ("indexer_q", r"\.indexer\."),
        ("shared_expert", r"\.shared_experts?\."),
        ("routed_expert", r"\.experts\."),
        ("router", r"\.(mlp|ffn|block_sparse_moe)\.(gate|router)\.weight$"),
        ("attention", r"\.(self_attn|attn|attention|linear_attn|mixer)\."),
        ("dense_mlp", r"\.(mlp|ffn|feed_forward)\."),
        ("lm_head", r"(^|\.)(lm_head|head|output)\.weight$"),
        ("embedding", r"(^|\.)(embed_tokens|embed|wte|tok_embeddings)\.weight$"),
    )
)

# The roles InferaSim's roofline cost model prices with one field each.
COST_FIELDS = {
    "linear_weight_dtype": ("attention", "dense_mlp"),
    "moe_expert_dtype": ("routed_expert",),
}
# What those fields accept (``cli.py``'s choices).
PRICEABLE = {
    "linear_weight_dtype": ("bf16", "fp8", "mxfp4", "fp4"),
    "moe_expert_dtype": ("bf16", "fp8", "mxfp4"),
}

_PLAIN = {
    "BF16": "bf16",
    "F16": "fp16",
    "F32": "fp32",
    "F8_E4M3": "fp8",
    "F8_E4M3FN": "fp8",
    "F8_E5M2": "fp8",
}
_PACKED = ("U8", "I8")
_SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv", "_scales", ".scales")


def _is_weight(name: str, shape: Sequence[int]) -> bool:
    if len(shape) < 2 or "norm" in name or name.endswith(_SCALE_SUFFIXES) or "bias" in name:
        return False
    return bool(
        name.endswith((".weight", "_blocks", ".weight_packed"))
        or re.search(r"experts\.(gate_up_proj|down_proj)$", name)
        or ROLE_PATTERNS[0][1].search(name)
    )


def role_of(name: str) -> str | None:
    for role, pat in ROLE_PATTERNS:
        if pat.search(name):
            return role
    return None


def _scale_of(name: str, dtypes: Mapping[str, str]) -> str | None:
    stem = re.sub(r"(\.weight|_blocks|\.weight_packed)$", "", name)
    for suf in _SCALE_SUFFIXES:
        if stem + suf in dtypes:
            return dtypes[stem + suf]
    return None


def tensor_format(name: str, dtype: str, dtypes: Mapping[str, str], quant: Mapping) -> str:
    """The format one stored weight runs at.

    Plain dtypes are their own format. A byte-typed weight is two 4-bit values
    per byte; its block scales say which 4-bit format: E8M0 (or bytes, gpt-oss'
    ``*_scales``) is MXFP4, an E4M3 scale is NVFP4. ``I32`` is
    compressed-tensors' packed int4.
    """
    if dtype in _PLAIN:
        return _PLAIN[dtype]
    if dtype == "I32" and name.endswith("weight_packed"):
        return "int4"
    if dtype in _PACKED:
        scale = _scale_of(name, dtypes)
        if scale in ("F8_E8M0", "U8"):
            return "mxfp4"
        if scale in ("F8_E4M3", "F8_E4M3FN"):
            return "nvfp4"
        method = str(quant.get("quant_method") or "").lower()
        if method in ("mxfp4", "fp4") or str(quant.get("expert_dtype") or "").lower() == "fp4":
            return "mxfp4"
    return f"unknown:{dtype}"


@dataclass
class CheckpointDtypes:
    """One format per op role, and the tensors that say so."""

    roles: dict[str, str]
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    examples: dict[str, str] = field(default_factory=dict)
    checkpoint: str = ""
    path: str = ""

    def fields(self) -> dict[str, str]:
        """The cost-model fields the roles set (``linear_weight_dtype``, ...)."""
        out = {}
        for f, roles in COST_FIELDS.items():
            fmt = next((self.roles[r] for r in roles if r in self.roles), None)
            if fmt:
                out[f] = fmt
        return out

    def unpriceable(self) -> list[str]:
        """Roles whose format the cost model has no kernel for."""
        bad = []
        for f, fmt in self.fields().items():
            if fmt not in PRICEABLE[f]:
                bad.append(f"{f}={fmt} (InferaSim prices {', '.join(PRICEABLE[f])})")
        return bad

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "checkpoint": self.checkpoint,
            "roles": self.roles,
            "fields": self.fields(),
            "counts": self.counts,
            "examples": self.examples,
        }

    @classmethod
    def from_json(cls, d: Mapping, path: str = "") -> CheckpointDtypes:
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path or 'checkpoint dtypes'}: schema is not {SCHEMA}")
        return cls(
            roles=dict(d["roles"]),
            counts={k: dict(v) for k, v in (d.get("counts") or {}).items()},
            examples=dict(d.get("examples") or {}),
            checkpoint=str(d.get("checkpoint") or ""),
            path=path,
        )

    @classmethod
    def load(cls, path: str) -> CheckpointDtypes:
        with open(path) as f:
            return cls.from_json(json.load(f), path)


def classify(
    tensors: Iterable[tuple[str, str, Sequence[int]]],
    quant: Mapping | None = None,
    checkpoint: str = "",
) -> CheckpointDtypes:
    """Roles from ``(name, safetensors dtype, shape)`` triples.

    A role holding more than one format (a quantized model with a few layers
    left in bf16) takes the one most of its weights use; ``counts`` keeps the
    split.
    """
    tensors = list(tensors)
    quant = dict(quant or {})
    dtypes = {n: d for n, d, _ in tensors}
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    examples: dict[str, str] = {}
    for name, dtype, shape in tensors:
        if not _is_weight(name, shape):
            continue
        role = role_of(name)
        if role is None:
            continue
        counts[role][tensor_format(name, dtype, dtypes, quant)] += 1
        examples.setdefault(role, f"{name} {dtype}")
    roles = {r: c.most_common(1)[0][0] for r, c in sorted(counts.items())}
    return CheckpointDtypes(
        roles=roles,
        counts={r: dict(c) for r, c in sorted(counts.items())},
        examples=dict(sorted(examples.items())),
        checkpoint=checkpoint,
    )


class CheckpointMismatch(ValueError):
    """A flag prices an op at a format its checkpoint does not store."""


_FLAG_FIELDS = {
    "--linear-weight-dtype": "linear_weight_dtype",
    "--moe-expert-dtype": "moe_expert_dtype",
}
# Formats that run the same kernels: fp4 is the CLI's name for mxfp4.
_SAME = {"fp4": "mxfp4", "fp16": "bf16"}


def flags_in(argv: Sequence[str]) -> dict[str, str]:
    """The dtype fields an argv sets explicitly (the last of a repeated flag wins)."""
    out: dict[str, str] = {}
    argv = list(argv)
    for i, tok in enumerate(argv):
        flag, eq, value = tok.partition("=")
        if flag in _FLAG_FIELDS:
            if not eq:
                value = argv[i + 1] if i + 1 < len(argv) else ""
            out[_FLAG_FIELDS[flag]] = value
    return out


def check_flags(ckpt: CheckpointDtypes, given: Mapping[str, str | None]) -> None:
    """Fail when an explicit dtype field contradicts the checkpoint, or it cannot be priced."""
    problems = []
    want = ckpt.fields()
    for f, value in given.items():
        if not value or f not in want:
            continue
        if _SAME.get(value.lower(), value.lower()) != _SAME.get(want[f], want[f]):
            roles = "/".join(r for r in COST_FIELDS[f] if r in ckpt.roles)
            problems.append(
                f"{f}={value} but {ckpt.checkpoint or 'the checkpoint'} stores {roles} as "
                f"{want[f]} (e.g. {ckpt.examples.get(COST_FIELDS[f][0], '?')})"
            )
    problems += [f"cannot price {p}" for p in ckpt.unpriceable()]
    if problems:
        raise CheckpointMismatch("; ".join(problems))


def _header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def read_tensors(model_dir: str) -> list[tuple[str, str, list[int]]]:
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"{model_dir}: no *.safetensors files")
    out = []
    for path in files:
        for name, meta in _header(path).items():
            if name != "__metadata__":
                out.append((name, str(meta["dtype"]), list(meta["shape"])))
    return out


def quantization_config(model_dir: str) -> dict:
    """``config.json``'s ``quantization_config``, with a top-level ``expert_dtype``."""
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    quant = dict(cfg.get("quantization_config") or {})
    if cfg.get("expert_dtype"):
        quant.setdefault("expert_dtype", cfg["expert_dtype"])
    return quant


def read(model_dir: str) -> CheckpointDtypes:
    model_dir = model_dir.rstrip("/")
    return classify(
        read_tensors(model_dir), quantization_config(model_dir), os.path.basename(model_dir)
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dir", help="checkpoint directory (config.json + *.safetensors)")
    ap.add_argument("--out", default="", help="write the map here (default: stdout)")
    a = ap.parse_args(argv)
    d = read(a.model_dir)
    text = json.dumps(d.to_json(), indent=2) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    for problem in d.unpriceable():
        print(f"[checkpoint_dtypes] cannot price {problem}", file=sys.stderr)
    return 1 if d.unpriceable() else 0


if __name__ == "__main__":
    raise SystemExit(main())
