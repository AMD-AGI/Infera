#!/usr/bin/env python3
"""Classifying a device kernel symbol, driven by `kernel_taxonomy.yaml`.

The rules live in YAML so that a reviewer can read and extend them without
reading Python. Everything here is first-match-wins over an ordered list, which
is why the YAML comments say where to insert a new rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_CACHE: dict | None = None


def load(path: Path | None = None) -> dict:
    global _CACHE
    if _CACHE is None:
        where = path or (Path(__file__).resolve().parent / "kernel_taxonomy.yaml")
        _CACHE = yaml.safe_load(where.read_text(encoding="utf-8"))
    return _CACHE


def _first_match(name: str, rules: list[dict], key: str) -> dict | None:
    for rule in rules:
        for pattern in rule.get("patterns", []):
            if re.search(pattern, name):
                return rule
    return None


def bucket_of(name: str, rules: dict | None = None) -> dict:
    """`{bucket, routable, excluded_reason}` for one symbol.

    An unmatched symbol lands in `unknown`, which is recorded and excluded
    rather than dropped: the unmatched set is where the next rule comes from.
    """
    rules = rules or load()
    match = _first_match(name, rules.get("buckets", []), "buckets")
    if match is None:
        return {
            "bucket": "unknown",
            "routable": False,
            "excluded_reason": "no_taxonomy_rule_matched",
        }
    return {
        "bucket": match["name"],
        "routable": bool(match.get("routable")),
        "excluded_reason": match.get("excluded_reason", ""),
    }


def fellow_of(name: str, rules: dict | None = None) -> dict:
    """`{fellow, language}` — the forge-loop `--fellow` guess."""
    rules = rules or load()
    match = _first_match(name, rules.get("fellows", []), "fellows")
    if match is None:
        return {"fellow": "", "language": ""}
    return {"fellow": match["fellow"], "language": match["language"]}


def category_of(name: str, rules: dict | None = None) -> str:
    rules = rules or load()
    match = _first_match(name, rules.get("categories", []), "categories")
    return match["category"] if match else "unknown"


def dtypes_of(name: str, rules: dict | None = None) -> dict:
    """dtype evidence recovered from the symbol name.

    AMD kernel names frequently encode operand precision, for example
    `mfma_moe1_silu_mul_afp4_wfp4_bf16_...` carries fp4 activations, fp4 weights
    and a bf16 accumulate. Returns `{role: dtype}` for whatever was found;
    an empty dict means the name carried no evidence and the caller records
    `invocation.arguments[*].dtype` in `missing_fields`.

    Longest token first, so `bf16` does not shadow `mxfp4`.
    """
    rules = rules or load()
    tokens = rules.get("dtype_tokens", {})
    found: dict[str, str] = {}
    for token in sorted(tokens, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", name):
            spec = tokens[token]
            found.setdefault(spec["role"], spec["dtype"])
    return found


def classify(name: str, rules: dict | None = None) -> dict:
    """Everything the taxonomy can say about one symbol, in one call."""
    rules = rules or load()
    out = bucket_of(name, rules)
    out.update(fellow_of(name, rules))
    out["category"] = category_of(name, rules)
    out["dtypes"] = dtypes_of(name, rules)
    return out
