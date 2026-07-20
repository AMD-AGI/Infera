###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Generate a RAG-shaped deterministic corpus for L2 cache validation.

The growing-prefix synthetic corpus (`corpus.py`) and the SWE-bench
trace corpus (`corpus_swebench.py`) both turned out poorly for L2
validation:

- `corpus.py` sessions are independent — no cross-session shared
  bytes for kvd to hold and L2 to re-serve.
- `corpus_swebench.py` does have shared content but the cross-trial
  share is a tool-definitions system prompt that drives L1 toward
  ~85% hit, leaving L2 with ~0 work.

The L2 pool was built for the **high-concurrency RAG pattern**:
N users hit the same long system prompt with their own queries.
Per-user content is short; the shared prefix dominates. When N
users land concurrently, L1 evicts and refills the shared prefix
repeatedly — that's the thrashing L2 was designed to absorb.

This module emits exactly that shape::

    [
      {"role": "system",    "content": <shared, ~SYSTEM_PROMPT_TOKENS tokens>},
      {"role": "user",      "content": <session-unique, ~QUERY_TOKENS tokens>},
    ]

For N sessions, the corpus has N entries. Each session's first
message is bit-identical (same shared prompt); the second is
session-unique (deterministic from seed + session_id).

The first message's bytes are designed to span multiple
(`block_size=16`) kvd blocks. A 32 KB system prompt at ~4 chars/
token is ~8K tokens = ~500 blocks per session — exactly the
"thousands of blocks shared across N users" pattern that bends
L2's value curve.

Usage::

    python -m bench.kvcache.deterministic_l3.corpus_rag \\
        --num-sessions 32 --system-prompt-tokens 8000 \\
        --query-tokens 200 --out /tmp/rag-corpus.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Re-exported for callers/tests that referenced it here before the shared
# helper landed.
from ._common import APPROX_TOKENS_PER_WORD, word_salad, write_corpus


def _fixed_system_prompt(target_tokens: int, *, seed: int = 1337) -> str:
    """Deterministic shared system prompt. Uses a separate seed-stream
    from the per-session query generation so changing `--query-tokens`
    or `--num-sessions` doesn't perturb the system-prompt bytes."""
    return word_salad(
        f"rag-system-prompt-{seed}",
        target_tokens,
        lambda rng: f"sys{rng.randint(0, 999999):06d}",
    )


def _tenant_system_prompt(tenant_id: int, target_tokens: int, *, seed: int = 1337) -> str:
    """Per-tenant system prompt. Tenants are independent — different
    tenant ids produce different prompt bytes, so a multi-tenant
    corpus can force L1 eviction across tenant boundaries.

    Within a tenant, the prompt is byte-identical across all
    sessions (same deduplication property as the single-tenant
    `_fixed_system_prompt`)."""
    return word_salad(
        f"rag-tenant-{tenant_id}-{seed}",
        target_tokens,
        lambda rng: f"t{tenant_id:02d}_{rng.randint(0, 999999):06d}",
    )


def _session_query(session_id: int, target_tokens: int) -> str:
    """Per-session unique query bytes. Keyed by session_id so two
    runs with the same num-sessions produce bit-identical corpora."""
    return word_salad(
        f"rag-query-{session_id}",
        target_tokens,
        lambda rng: f"q{session_id:03d}-{rng.randint(0, 99999):05d}",
    )


def build_corpus(
    *,
    num_sessions: int,
    system_prompt_tokens: int,
    query_tokens: int,
    seed: int,
    num_tenants: int = 1,
) -> tuple[dict, dict]:
    """Pure-function pipeline so tests don't need any IO. Returns
    `(corpus, stats)` matching the shape `replay.py` consumes.

    `num_tenants=1` (the default) builds the single-tenant variant
    documented at the top of the file: every session shares the same
    prefix bytes — L1 dedupes to one copy, L2 has no work.

    `num_tenants>1` builds the multi-tenant variant: M distinct
    prefixes, each repeated K=ceil(num_sessions/num_tenants) times,
    INTERLEAVED in dispatch order (tenant_0 sess_0, tenant_1 sess_0,
    ..., tenant_{M-1} sess_0, tenant_0 sess_1, ...). With M*S total
    prefix bytes exceeding L1 capacity, the interleaving forces L1
    to evict tenant_0's prefix between its sessions — that's the
    pattern L2 was built to absorb.
    """
    if num_tenants < 1:
        raise ValueError(f"num_tenants must be >= 1, got {num_tenants}")
    if num_sessions > 0 and num_tenants > num_sessions:
        raise ValueError(f"num_tenants ({num_tenants}) must be <= num_sessions ({num_sessions})")
    # Replay.py's contract is user-led alternation (the engine
    # generates the next assistant turn). We combine the shared
    # prefix and the per-session query into a single `user`
    # message rather than using a separate `system` role — the
    # tokens-on-the-wire bytes are identical from the engine's
    # KV-cache perspective, and we avoid having to teach replay
    # about strict OpenAI role semantics. The leading bytes of
    # `content` are the shared prefix for every session, which is
    # what L1 / L2 / kvd actually cache by content hash.
    if num_tenants == 1:
        prefixes = [_fixed_system_prompt(system_prompt_tokens, seed=seed)]
    else:
        prefixes = [
            _tenant_system_prompt(t, system_prompt_tokens, seed=seed) for t in range(num_tenants)
        ]

    # Build interleaved (tenant_id, session_within_tenant) order.
    # Round-robin so consecutive sessions in dispatch order have
    # distinct tenants — this is what forces L1 thrash.
    sessions = []
    for s in range(num_sessions):
        tenant_id = s % num_tenants
        session_within_tenant = s // num_tenants
        query = _session_query(s, query_tokens)
        sessions.append(
            {
                "session_id": s,
                "tenant_id": tenant_id,
                "session_within_tenant": session_within_tenant,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prefixes[tenant_id]}\n\n---\n\n{query}",
                    },
                ],
            }
        )

    corpus = {
        "config": {
            "shape": "rag" if num_tenants == 1 else "rag-multi-tenant",
            "num_sessions": num_sessions,
            "num_tenants": num_tenants,
            "system_prompt_tokens_target": system_prompt_tokens,
            "query_tokens_target": query_tokens,
            "seed": seed,
            "approx_tokens_per_word": APPROX_TOKENS_PER_WORD,
            "prefix_chars_per_tenant": len(prefixes[0]),
        },
        "sessions": sessions,
    }
    total_bytes = sum(sum(len(m["content"]) for m in s["messages"]) for s in sessions)
    # In multi-tenant mode, "shared prefix bytes" is per-tenant, and
    # each tenant's prefix is repeated K times. Report both for
    # operator visibility.
    stats = {
        "num_sessions": num_sessions,
        "num_user_turns": num_sessions,  # one user message per session in RAG shape
        "num_tenants": num_tenants,
        "prefix_bytes_per_tenant": len(prefixes[0]),
        "distinct_prefix_bytes": sum(len(p) for p in prefixes),
        "total_content_bytes": total_bytes,
        # Back-compat alias for the single-tenant case.
        "shared_prefix_bytes": len(prefixes[0]) if num_tenants == 1 else 0,
    }
    return corpus, stats


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--num-sessions",
        type=int,
        default=32,
        help="N — number of distinct users hitting the same system prompt",
    )
    p.add_argument(
        "--system-prompt-tokens",
        type=int,
        default=8000,
        help="Approx tokens of shared prefix. 32K-token system prompt = "
        "~24000-26000 in MiniMax tokenization; sized to span hundreds "
        "of kvd blocks so the L2-vs-L1 thrash signal is observable.",
    )
    p.add_argument(
        "--query-tokens",
        type=int,
        default=200,
        help="Approx tokens of per-session unique content. Kept small "
        "so the shared prefix is the dominant work.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--num-tenants",
        type=int,
        default=1,
        help="1 = single-tenant (L1 dedupes the one prefix). >1 = "
        "multi-tenant: M distinct prefixes interleaved round-robin so "
        "L1 evicts tenant_0's prefix before tenant_0's next session — "
        "this is what L2 was built for.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    corpus, stats = build_corpus(
        num_sessions=args.num_sessions,
        system_prompt_tokens=args.system_prompt_tokens,
        query_tokens=args.query_tokens,
        seed=args.seed,
        num_tenants=args.num_tenants,
    )
    write_corpus(args.out, corpus)
    mb = stats["total_content_bytes"] / (1024 * 1024)
    distinct_mb = stats["distinct_prefix_bytes"] / (1024 * 1024)
    print(
        f"wrote {stats['num_sessions']} sessions ({stats['num_user_turns']} requests) "
        f"to {args.out} — {mb:.1f} MB total. "
        f"tenants={stats['num_tenants']}, distinct prefix bytes={distinct_mb:.2f} MB, "
        f"per-tenant prefix={stats['prefix_bytes_per_tenant'] / 1024:.1f} KB"
    )


if __name__ == "__main__":
    main()
