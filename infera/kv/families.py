###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFamily:
    canonical: str
    aliases: tuple[str, ...]


# Built-in registry. Operators extend via ~/.infera/families.yaml at startup.
# See 11-attention-kinds.md § "Model family with aliases".
KNOWN_FAMILIES: list[ModelFamily] = [
    ModelFamily("llama-3", aliases=("llama3", "Llama-3", "meta-llama/Llama-3")),
    ModelFamily("qwen2", aliases=("Qwen2", "qwen_2")),
    ModelFamily("qwen3", aliases=("Qwen3", "qwen-3", "qwen_3")),
    ModelFamily("qwen3.5", aliases=("Qwen3.5", "qwen-3.5", "qwen3_5", "qwen35")),
    ModelFamily("qwen3.6", aliases=("Qwen3.6", "qwen-3.6", "qwen3_6", "qwen36")),
    ModelFamily("qwen3-next", aliases=("Qwen3-Next", "qwen3_next")),
    ModelFamily(
        "gpt-oss",
        aliases=("gpt_oss", "GPT-OSS", "openai/gpt-oss-120b", "openai/gpt-oss-20b"),
    ),
    ModelFamily(
        "deepseek-v3",
        aliases=("DeepSeek-V3", "deepseek_v3", "deepseek-ai/DeepSeek-V3"),
    ),
    ModelFamily("deepseek-v2", aliases=("DeepSeek-V2", "deepseek_v2")),
    ModelFamily("mistral", aliases=("Mistral", "mistralai/Mistral")),
    ModelFamily("gemma-3", aliases=("gemma3", "google/gemma-3")),
    ModelFamily("jamba", aliases=("Jamba", "ai21/jamba")),
    ModelFamily(
        "granite-hybrid",
        aliases=("granite_hybrid", "ibm-granite/granite-3-hybrid"),
    ),
]


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_family(name: str, registry: list[ModelFamily] | None = None) -> str:
    """Map any known alias to canonical. Unknown → return lowercased input.

    Resolution rules (in order):
      1. Exact canonical match → return canonical.
      2. Exact alias match → return alias's canonical.
      3. Alias is a substring of `name` → return alias's canonical.
         (Catches HF-style IDs like "Qwen/Qwen3.6-27B" → "qwen3.6".)
      4. Otherwise → return normalized name unchanged.

    Resolution is case- and separator-insensitive (`_` and `-` interchangeable).
    """
    norm = _normalize(name)
    reg = registry if registry is not None else KNOWN_FAMILIES
    for fam in reg:
        if norm == fam.canonical:
            return fam.canonical
    for fam in reg:
        for alias in fam.aliases:
            alias_norm = _normalize(alias)
            if norm == alias_norm:
                return fam.canonical
    # Substring pass: find ALL candidates, return the one whose matched
    # token is the longest. Without this, "Qwen/Qwen3.6-27B" would match
    # the shorter canonical "qwen3" before reaching alias "qwen-3.6".
    best: tuple[int, str] | None = None
    for fam in reg:
        # Include canonical itself in the substring search to handle e.g.
        # the HF ID containing only the canonical, no alias.
        candidates = (fam.canonical, *(_normalize(a) for a in fam.aliases))
        for cand in candidates:
            if not cand:
                continue
            if cand in norm:
                if best is None or len(cand) > best[0]:
                    best = (len(cand), fam.canonical)
    if best is not None:
        return best[1]
    return norm


def load_registry_from_yaml(path: str | Path) -> list[ModelFamily]:
    """Load an additional registry from a YAML file. Format:

        - canonical: my-new-model
          aliases: [MyNewModel, my_new_model, hf-org/my-new-model]

    Returns the combined list (builtins + file entries). Operator can also
    override builtins by repeating a canonical name in the file; later
    entries win. Malformed YAML is logged and the file is ignored.
    """
    import logging

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        logging.getLogger(__name__).warning("PyYAML not installed; cannot load %s. Skipping.", path)
        return KNOWN_FAMILIES.copy()

    p = Path(path).expanduser()
    if not p.exists():
        return KNOWN_FAMILIES.copy()

    try:
        with p.open("r", encoding="utf-8") as f:
            entries = yaml.safe_load(f) or []
    except yaml.YAMLError as exc:
        logging.getLogger(__name__).warning("Failed to parse %s: %s. Skipping.", path, exc)
        return KNOWN_FAMILIES.copy()

    # Merge: file entries override builtins on canonical name.
    overrides: dict[str, ModelFamily] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        canonical = raw.get("canonical")
        if not isinstance(canonical, str):
            continue
        aliases_raw = raw.get("aliases") or []
        if not isinstance(aliases_raw, list):
            continue
        aliases = tuple(str(a) for a in aliases_raw)
        overrides[canonical] = ModelFamily(canonical=canonical, aliases=aliases)

    out: list[ModelFamily] = []
    seen: set[str] = set()
    for fam in KNOWN_FAMILIES:
        if fam.canonical in overrides:
            out.append(overrides.pop(fam.canonical))
        else:
            out.append(fam)
        seen.add(fam.canonical)
    # Remaining overrides are pure additions.
    out.extend(overrides.values())
    return out
