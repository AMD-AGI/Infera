###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from infera.kv.types import AttentionGroup

# Files that define tokenization behavior. The directory-hash path includes
# ONLY these (never model weights), so the digest stays cheap and stable no
# matter how a tokenizer is stored: HF fast (tokenizer.json), sentencepiece
# (*.model), or tiktoken + remote code (tiktoken.model + tokenization_*.py).
# Without this allowlist, a directory like a 1TB model dir would hash every
# safetensors shard. Matched case-insensitively against the file name.
_TOKENIZER_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "tiktoken.model",
    "*.tiktoken",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokeniz*.py",  # remote code referenced by auto_map, e.g. tokenization_kimi.py
)


def _is_tokenizer_file(name: str) -> bool:
    lname = name.lower()
    return any(fnmatch.fnmatch(lname, pat) for pat in _TOKENIZER_FILE_PATTERNS)


def compute_tokenizer_digest(tokenizer_json_path: str | Path) -> str:
    """Compute the canonical tokenizer digest used by ``compat_key``.

    For tokenizers stored as a single ``tokenizer.json``, this hashes the
    file contents. For tokenizer setups without a self-contained
    ``tokenizer.json`` (e.g. tiktoken/sentencepiece + remote code), pass
    the directory path and the function hashes only the tokenizer-defining
    files (see ``_TOKENIZER_FILE_PATTERNS``) — never model weights.

    Returns 16 lowercase hex chars (sha256 first-16).
    """
    p = Path(tokenizer_json_path).expanduser()
    h = hashlib.sha256()
    if p.is_file():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
    elif p.is_dir():
        # Hash ONLY tokenizer-defining files, canonical-sorted by name for
        # determinism across the router and every worker.
        matched = sorted(
            (c for c in p.iterdir() if c.is_file() and _is_tokenizer_file(c.name)),
            key=lambda c: c.name,
        )
        if not matched:
            raise FileNotFoundError(
                f"no tokenizer files found under directory: {tokenizer_json_path}"
            )
        for child in matched:
            h.update(child.name.encode("utf-8"))
            h.update(b"\x00")
            with child.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            h.update(b"\x00\x00")
    else:
        raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer_json_path}")
    return h.hexdigest()[:16]


def _attention_group_to_canonical(g: AttentionGroup) -> dict[str, Any]:
    """Convert AttentionGroup to a canonical dict for hashing.

    `layer_indices` is normalized to a sorted tuple before serialization,
    so two groups with the same layers in different listing orders hash
    identically.
    """
    d = asdict(g)
    if d.get("layer_indices") is not None:
        d["layer_indices"] = sorted(d["layer_indices"])
    return d


def compute_compat_key(
    model_id: str,
    kv_dtype: str,
    quant: str | None,
    index_block_size: int,
    attention_groups: list[AttentionGroup],
    tokenizer_digest: str,
) -> str:
    """Derive the 16-hex-char `compat_key` from KV-bytes-affecting config.

    See 15-infera-kvd.md § "Cache consistency: compat_key" for the
    rationale on which fields are included (and notably which are NOT:
    tp, pp, rank, engine_version, infera_version).
    """
    canonical = {
        "model_id": model_id,
        "kv_dtype": kv_dtype,
        "quant": quant,
        "index_block_size": index_block_size,
        "attention_groups": [_attention_group_to_canonical(g) for g in attention_groups],
        "tokenizer_digest": tokenizer_digest,
    }
    # sort_keys ensures canonical ordering. separators eliminate cosmetic whitespace.
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# The fixed canary string used for cross-worker tokenizer consistency
# verification (see 03-data-model.md § "Cross-worker tokenizer consistency").
# Do not change without bumping event_version — would invalidate every fleet.
CANARY_STRING = "Hello, world!\n你好，世界！<|im_start|>system\n<|endoftext|><|im_end|>🦄"


def tokenize_canary(tokenizer: Any) -> list[int]:
    """Tokenize CANARY_STRING with a HuggingFace-compatible tokenizer.

    The result is intended for inclusion in worker registration as
    `kv.tokenizer_canary`. We deliberately pass `add_special_tokens=False`
    because chat templates / sentinels are already embedded in
    CANARY_STRING; if a tokenizer wraps them again, two equivalent
    tokenizers would produce different IDs and we'd false-reject.
    """
    return tokenizer.encode(CANARY_STRING, add_special_tokens=False)
