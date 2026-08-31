###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The Python router's render, against the engine's, byte for byte.

Same corpus and same goldens as the Rust hasher's
``render_parity_matches_the_engine`` -- `rust/router/tests/render_parity/`.
Two ports of the same contract deserve one oracle, and a corpus that only one
of them runs is a corpus that drifts.

This half looks easy: the Python router calls the real
``transformers.apply_chat_template``, so it cannot diverge on Jinja dialect the
way the Rust port can. What it can still get wrong -- and did -- is the
*context*: which of `tools`, `reasoning_effort`, `chat_template_kwargs` reach
the template, in what shape. Those are exactly the bytes at the front of the
prompt, where a divergence costs every block.

Skipped unless ``INFERA_TEST_RENDER_PARITY=name=/path[,...]`` names a model dir
with goldens, so a machine without weights stays green.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from infera.router.kv_event.block_hasher import BlockHasher
from infera.router.kv_event.hasher import hash_request

CORPUS = pathlib.Path(__file__).resolve().parents[3] / "rust" / "router" / "tests" / "render_parity"
BLOCK_SIZE = 64
RENDER_ERROR = "__RENDER_ERROR__"


def _targets() -> list[tuple[str, str]]:
    spec = os.environ.get("INFERA_TEST_RENDER_PARITY", "")
    out = []
    for entry in spec.split(","):
        if entry.strip():
            name, _, path = entry.partition("=")
            out.append((name.strip(), path.strip()))
    return out


def _cases() -> list[tuple[str, str, pathlib.Path]]:
    return [
        (name, path, body)
        for name, path in _targets()
        for body in sorted((CORPUS / "bodies").glob("*.json"))
    ]


@pytest.mark.skipif(not _targets(), reason="set INFERA_TEST_RENDER_PARITY=name=/path")
@pytest.mark.parametrize(
    ("name", "model_dir", "body_path"),
    _cases(),
    ids=lambda v: v.stem if isinstance(v, pathlib.Path) else str(v),
)
def test_render_parity(name: str, model_dir: str, body_path: pathlib.Path):
    golden_path = CORPUS / "goldens" / name / f"{body_path.stem}.txt"
    if not golden_path.exists():
        pytest.skip(f"no golden; run scripts/gen_render_goldens.py --model-dir {model_dir} --name {name}")
    golden = golden_path.read_text()
    body = {"model": name, **json.loads(body_path.read_text())}

    from transformers import AutoTokenizer

    hasher = BlockHasher(tokenizer_path=model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    hasher._tokenizers[(None, model_dir)] = tokenizer

    got = hasher.hash_for(body, block_size=BLOCK_SIZE)

    if golden.startswith(RENDER_ERROR):
        # The engine could not render this body at all. Hashing *something*
        # here is the worst outcome: a prefix that cannot exist, cached forever.
        assert got == [], f"engine refused this body ({golden.strip()}) but the router hashed it"
        return

    want = hash_request(tokenizer.encode(golden, add_special_tokens=False), BLOCK_SIZE)
    assert got == want, (
        f"{name}/{body_path.stem}: the router's prompt is not the engine's. "
        "Nothing fails at runtime when this happens -- the block hashes just never "
        "match again and kv-aware silently becomes load balancing."
    )
