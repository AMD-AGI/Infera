###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Generate a deterministic chat corpus for L3 (kvd) cache validation.

The agent-bench harness in `light-trace-benchmark/` chains turns by
feeding the model's generated output back into the next turn's
prompt — which means the prompt content depends on the model's
outputs, which are *not* bit-reproducible across vLLM restarts
(sampling kernels diverge at FP rounding). Cross-restart cache
replay therefore can't validate kvd's L3 retention.

This generator instead emits a corpus where:

- Every user turn's text is a deterministic function of
  `(session_id, turn_idx)`.
- Every assistant turn is a fixed-byte string (we'll send the
  request with `max_tokens=1` so the model's actual generation
  doesn't matter — what we measure is prefill TTFT).
- A session's turn-N prompt is exactly turn-(N-1)'s prompt +
  fixed-assistant-text + new-user-turn. This makes the growing
  prefix bit-identical to the *previous* run of the same session.

Usage:

    python -m bench.kvcache.deterministic_l3.corpus \\
        --sessions 32 --turns 8 --turn-tokens 6000 \\
        --out /tmp/corpus.json

Sizing rules of thumb (for MiniMax-M2.5 ~1.3 tokens/word):

- 32 sessions × 8 turns × 6000 tokens/turn → turn-N prompt has
  N × 6000 tokens. By turn 7: ~48K tokens. Total 256 requests.
- Each session's working set grows to ~48K tokens; 32 simultaneous
  sessions → ~1.5M tokens in play — comfortably exceeds the
  ~500K-token L1 cap on a TP=1 MiniMax-M2.5 at MI355X 256GB.
  Forces L1 eviction → kvd reads on the replay phase.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Re-exported for callers/tests that referenced it here before the shared
# helper landed.
from ._common import APPROX_TOKENS_PER_WORD, word_salad, write_corpus

__all__ = ["APPROX_TOKENS_PER_WORD", "gen_user_turn", "gen_session", "main"]


def gen_user_turn(session_id: int, turn_idx: int, target_tokens: int) -> str:
    """Build a deterministic ~target-token word salad. Keyed by
    `(session_id, turn_idx)` so a given turn always renders the same
    bytes regardless of run order."""
    return word_salad(
        f"sess{session_id}-turn{turn_idx}",
        target_tokens,
        lambda rng: f"word{rng.randint(0, 999999):06d}",
    )


def gen_session(session_id: int, n_turns: int, turn_tokens: int) -> list[dict]:
    """One full session as a chat-completions `messages` list.

    Layout: user, assistant, user, assistant, ..., user
    (always ends on a user turn — the next request is "predict the
    assistant's reply to this user").

    The final assistant message is omitted — that's what the engine
    is being asked to generate. The earlier assistant slots hold a
    fixed 12-byte string so the conversation has alternating roles
    (required by most chat templates including MiniMax-M2.5)."""
    messages = []
    for turn in range(n_turns):
        messages.append({"role": "user", "content": gen_user_turn(session_id, turn, turn_tokens)})
        if turn < n_turns - 1:
            messages.append({"role": "assistant", "content": f"OK turn {turn}."})
    return messages


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sessions", type=int, default=32, help="number of independent sessions")
    p.add_argument("--turns", type=int, default=8, help="user-turn count per session")
    p.add_argument(
        "--turn-tokens",
        type=int,
        default=6000,
        help="approx prompt tokens added per user turn",
    )
    p.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = p.parse_args()

    corpus = {
        "config": {
            "sessions": args.sessions,
            "turns": args.turns,
            "turn_tokens": args.turn_tokens,
            "approx_tokens_per_word": APPROX_TOKENS_PER_WORD,
        },
        "sessions": [
            {
                "session_id": s,
                "messages": gen_session(s, args.turns, args.turn_tokens),
            }
            for s in range(args.sessions)
        ],
    }
    write_corpus(args.out, corpus)
    n_requests = args.sessions * args.turns
    last_turn_tokens_approx = args.turn_tokens * args.turns
    print(
        f"wrote {args.sessions} sessions × {args.turns} turns "
        f"= {n_requests} requests to {args.out} "
        f"(final turn ~{last_turn_tokens_approx:,} tokens)"
    )


if __name__ == "__main__":
    main()
