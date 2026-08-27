#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Convert SemiAnalysis Weka agentic coding traces to SGLang's agentic-trace format.

SGLang's ``--dataset-name agentic-trace`` replays multi-turn sessions correctly and
ships inside the sglang wheel, so a customer needs no extra tooling. What it does
not ship is data. The public traces (Apache-2.0) record only per-turn token counts
``in`` / ``out`` and KV block ids ``hash_ids`` -- no text -- so the text is
synthesized here while the structure (per-turn lengths, prefix reuse) is preserved.

    python weka_to_agentic_trace.py traces.jsonl -o out.json --output-len 220

Two properties of the target format drive the conversion:

1. History accumulates at replay time and cannot shrink, so each run of
   prefix-extending turns becomes its own conversation. A turn whose ``hash_ids``
   do not extend its predecessor's is a context reset and starts a new one.
2. Every turn shares one ``--sharegpt-output-len``. That is what makes exact input
   lengths reachable (``ignore_eos`` is on by default, so replies are exactly that
   long) and it is why ``--output-len`` here must equal the value passed to
   ``sglang.benchmark.serving``, or lengths drift turn over turn.

Not modelled: subagent fan-out (``type: "subagent"`` entries are skipped) and
inter-turn think time (the format has no timing channel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys

# Every entry is a single token when prefixed with a space under the GLM-5.2
# tokenizer; re-check before trusting it on a different model. Kept as a grid
# rather than one word per line, which is what the formatter would do to it.
# fmt: off
FILLER_POOL = [
    "def", "return", "self", "if", "else", "for", "while", "class", "import",
    "value", "data", "name", "file", "path", "line", "code", "test", "error",
    "func", "args", "type", "list", "dict", "true", "false", "none", "size",
]
# fmt: on

# GLM-5.2 chat template cost: prompt_tokens = 10 + 2 * n_messages + sum(content).
TEMPLATE_PREAMBLE = 10
TEMPLATE_PER_MSG = 2
MIN_BODY = 8  # a turn contributing fewer tokens than this is not worth emitting


def filler(n_tokens: int, seed: int) -> str:
    """n_tokens single-token words, deterministic in ``seed``."""
    rng = random.Random(seed)
    return "".join(" " + FILLER_POOL[rng.randrange(len(FILLER_POOL))] for _ in range(n_tokens))


def stable_seed(trace_id: object, segment: int, base: int) -> int:
    """Per-segment RNG seed, reproducible across processes.

    ``hash()`` cannot be used here: it is salted per process for str keys, so the
    same trace and ``--seed`` would synthesise different filler on every run and
    the dataset this writes would not be rebuildable.
    """
    digest = hashlib.sha256(f"{trace_id}:{segment}:{base}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def top_level_calls(trace: dict) -> list[dict]:
    return [q for q in trace["requests"] if q["type"] in ("n", "s")]


def extends(prev_ids: list[int], ids: list[int]) -> bool:
    return len(ids) >= len(prev_ids) and ids[: len(prev_ids)] == prev_ids


def split_segments(calls: list[dict]) -> list[list[dict]]:
    """Split at context resets: a turn that does not extend its predecessor's prefix."""
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for i, q in enumerate(calls):
        if i and not extends(calls[i - 1].get("hash_ids", []), q.get("hash_ids", [])):
            segs.append(cur)
            cur = []
        cur.append(q)
    if cur:
        segs.append(cur)
    return segs


def build_conversation(seg: list[dict], output_len: int, seed: int):
    """Emit turns whose replayed input lengths match ``seg``'s recorded ``in``.

    Returns (turns, achieved_in). Stops early if the recorded growth cannot cover
    the fixed reply length.
    """
    turns = []
    achieved = []
    n_msgs = 0
    content = 0

    for i, q in enumerate(seg):
        target = q["in"]
        if i == 0:
            body = target - TEMPLATE_PREAMBLE - TEMPLATE_PER_MSG
        else:
            # +2 messages since the last turn: the assistant reply and this user turn.
            body = (
                target - TEMPLATE_PREAMBLE - TEMPLATE_PER_MSG * (n_msgs + 2) - content - output_len
            )
        if body < MIN_BODY:
            break

        if i:
            n_msgs += 1  # assistant reply that replay appended
            content += output_len
        n_msgs += 1  # this turn's user message
        content += body

        turns.append(
            {
                "messages": [{"role": "user", "content": filler(body, seed + i)}],
                "prompt_tokens": target,
            }
        )
        achieved.append(TEMPLATE_PREAMBLE + TEMPLATE_PER_MSG * n_msgs + content)

    return turns, achieved


def pct(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p / 100
    f = int(k)
    return v[f] if f + 1 >= len(v) else v[f] + (v[f + 1] - v[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Weka agentic trace -> SGLang agentic-trace JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("source", help="Weka traces.jsonl (one JSON trace per line)")
    ap.add_argument("-o", "--out", help="output agentic-trace JSON (omit with --dry-run)")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report the resulting distribution without writing; the cheap way to "
        "pick --max-context",
    )
    ap.add_argument(
        "--output-len",
        type=int,
        default=220,
        help="tokens the server will generate per turn; MUST equal the "
        "--sharegpt-output-len passed to sglang.benchmark.serving (default: 220)",
    )
    ap.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="drop conversations shorter than this; the corpus is full of 1-2 turn "
        "fragments that say nothing about multi-turn behaviour (default: 4)",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="truncate conversations to this many turns; 0 = no cap",
    )
    ap.add_argument(
        "--max-context",
        type=int,
        default=0,
        help="drop conversations whose peak input exceeds this many tokens, to fit "
        "the corpus to what the deployment can prefill; 0 = no cap",
    )
    ap.add_argument(
        "--max-conversations",
        type=int,
        default=0,
        help="stop after this many conversations; 0 = all",
    )
    ap.add_argument("--seed", type=int, default=1337, help="filler RNG seed (default: 1337)")
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="tokenize the first N conversations and compare against the recorded "
        "input lengths (needs --tokenizer)",
    )
    ap.add_argument("--tokenizer", default=None, help="tokenizer path for --verify")
    args = ap.parse_args()
    if not args.out and not args.dry_run:
        ap.error("need -o/--out unless --dry-run")

    conversations = []
    peaks = []
    n_traces = 0
    n_segs = 0
    dropped_short = 0
    dropped_long = 0
    truncated = 0

    with open(args.source) as f:
        for line in f:
            if not line.strip():
                continue
            trace = json.loads(line)
            n_traces += 1
            calls = top_level_calls(trace)
            if not calls:
                continue

            for si, seg in enumerate(split_segments(calls)):
                n_segs += 1
                if args.max_turns:
                    seg = seg[: args.max_turns]
                if len(seg) < args.min_turns:
                    dropped_short += 1
                    continue

                seed = stable_seed(trace["id"], si, args.seed)
                turns, achieved = build_conversation(seg, args.output_len, seed)
                if len(turns) < args.min_turns:
                    dropped_short += 1
                    continue
                if len(turns) < len(seg):
                    truncated += 1

                peak = achieved[-1]
                if args.max_context and peak > args.max_context:
                    dropped_long += 1
                    continue

                conversations.append(turns)
                peaks.append(peak)
                if args.max_conversations and len(conversations) >= args.max_conversations:
                    break
            if args.max_conversations and len(conversations) >= args.max_conversations:
                break

    if not conversations:
        print("no conversations survived the filters", file=sys.stderr)
        return 1

    if not args.dry_run:
        out = {
            "metadata": {
                "source": args.source,
                "generator": "weka_to_agentic_trace.py",
                "output_len": args.output_len,
                "min_turns": args.min_turns,
                "max_turns": args.max_turns or None,
                "max_context": args.max_context or None,
                "seed": args.seed,
                "note": "Text is synthetic filler; token counts, per-turn growth and "
                "prefix reuse follow the source trace. Replay with "
                f"--sharegpt-output-len {args.output_len}.",
            },
            "conversations": conversations,
        }
        with open(args.out, "w") as f:
            json.dump(out, f)

    nturns = [len(c) for c in conversations]
    print(f"traces read:        {n_traces}")
    print(f"segments found:     {n_segs}")
    print(f"  dropped (short):  {dropped_short}")
    print(f"  dropped (long):   {dropped_long}")
    print(f"  truncated early:  {truncated}")
    print(f"conversations out:  {len(conversations)}")
    print(
        f"turns/conversation: p50={pct(nturns, 50):.0f}  p90={pct(nturns, 90):.0f}  "
        f"max={max(nturns)}  mean={st.mean(nturns):.1f}"
    )
    print(
        f"peak context (tok): p50={pct(peaks, 50):,.0f}  p90={pct(peaks, 90):,.0f}  "
        f"p99={pct(peaks, 99):,.0f}  max={max(peaks):,.0f}"
    )
    if args.dry_run:
        print("dry run: nothing written")
    else:
        print(f"wrote {args.out}")

    if args.verify:
        if not args.tokenizer:
            print("--verify needs --tokenizer", file=sys.stderr)
            return 1
        verify(conversations[: args.verify], args.tokenizer, args.output_len)
    return 0


def verify(conversations, tokenizer_path, output_len):
    """Replay history growth locally and compare with each turn's prompt_tokens."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    diffs = []
    for conv in conversations:
        history = []
        for turn in conv:
            history.extend(turn["messages"])
            text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            got = len(tok.encode(text, add_special_tokens=False))
            diffs.append(got - turn["prompt_tokens"])
            history.append({"role": "assistant", "content": filler(output_len, 0)})

    exact = sum(1 for d in diffs if d == 0)
    print(
        f"\nverify: {len(diffs)} turns, {exact} exact ({100 * exact / len(diffs):.1f}%), "
        f"diff min={min(diffs)} max={max(diffs)} mean={st.mean(diffs):+.2f}"
    )


if __name__ == "__main__":
    sys.exit(main())
