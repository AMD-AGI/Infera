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

CORPUS = pathlib.Path(__file__).resolve().parents[3] / "rust" / "router" / "tests" / "render_parity"
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
        pytest.skip(
            f"no golden for {name} at {golden_path}; goldens are recorded out of tree "
            "(see the comment above render_parity_matches_the_engine in block_hasher.rs)"
        )
    golden = golden_path.read_text()
    body = {"model": name, **json.loads(body_path.read_text())}

    from transformers import AutoTokenizer

    hasher = BlockHasher(tokenizer_path=model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    hasher._tokenizers[(None, model_dir)] = tokenizer

    # Token ids, not block hashes. `hash_request` drops the trailing partial
    # block, so at the production block size of 64 most of this corpus hashes
    # to `[]` on both sides and the assertion passes without comparing
    # anything -- including `reasoning_effort_low`, the divergence this file
    # exists for. Ids also name the position of a divergence, which is the
    # difference between "our preamble is wrong, every request misses" and
    # "one message shape is wrong".
    got = hasher.token_ids_for(body)

    if golden.startswith(RENDER_ERROR):
        # The engine could not render this body at all. Producing *something*
        # here is the worst outcome: a prefix that cannot exist, cached forever.
        assert got is None, (
            f"engine refused this body ({golden.strip()}) but the router rendered it"
        )
        return

    want = tokenizer.encode(golden, add_special_tokens=False)
    assert got is not None, (
        f"{name}/{body_path.stem}: the engine rendered this body and the router "
        "declined to, so every request like it routes on load."
    )
    assert got == want, (
        f"{name}/{body_path.stem}: the router's prompt is not the engine's; "
        f"{_first_divergence(got, want)}. Nothing fails at runtime when this "
        "happens -- the block hashes just never match again and kv-aware "
        "silently becomes load balancing."
    )


def _first_divergence(ours: list[int], theirs: list[int]) -> str:
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if a != b:
            lo = max(0, i - 4)
            return (
                f"diverges at token {i} of {len(theirs)} "
                f"(router {ours[lo : i + 4]} vs engine {theirs[lo : i + 4]})"
            )
    return (
        f"agrees for {min(len(ours), len(theirs))} tokens then ours is {len(ours)} vs {len(theirs)}"
    )
