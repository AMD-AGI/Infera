###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared helpers for the deterministic_l3 corpus generators.

All three generators (`corpus.py`, `corpus_rag.py`, `corpus_swebench.py`)
emit the same on-disk shape that `replay.py` consumes::

    {
      "config":   {...},
      "sessions": [
        {"session_id": 0, "messages": [{"role": ..., "content": ...}, ...]},
        ...
      ],
    }

The synthetic generators additionally share one primitive: a
seeded, deterministic "word salad" whose bytes are a pure function of
a seed key, so the same key always renders the same bytes across runs
and vLLM restarts (that byte-stability is the whole point — it's what
makes block hashes collide and kvd hits reproducible).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

# Heuristic only — actual tokens vary by tokenizer. The `<prefix>NNNNNN`
# word pattern splits into ~1-2 tokens under most BPE vocabularies. The
# bench sizes token targets generously, so overshoot is fine; undershoot
# would force a resize loop.
APPROX_TOKENS_PER_WORD = 1.3


def word_salad(
    seed_key: str,
    target_tokens: int,
    make_word: Callable[[random.Random], str],
) -> str:
    """Deterministic word salad ~`target_tokens` long.

    Seeds a `random.Random` from `seed_key` (so the bytes are a pure
    function of the key, independent of call order), computes the word
    count from `APPROX_TOKENS_PER_WORD`, and joins `make_word(rng)`
    that many times.

    `make_word` owns the per-corpus token format — callers deliberately
    use distinct prefixes (`word`/`sys`/`t00_`/`q000-`) so bytes from
    different corpora never collide into the same content hash.
    """
    rng = random.Random(seed_key)
    n_words = max(1, int(target_tokens / APPROX_TOKENS_PER_WORD))
    return " ".join(make_word(rng) for _ in range(n_words))


def write_corpus(out_path: str | Path, corpus: dict) -> None:
    """Serialize a corpus dict to `out_path` as indented JSON (the shape
    `replay.py` reads via `json.load`)."""
    Path(out_path).write_text(json.dumps(corpus, indent=2))
