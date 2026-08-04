#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Did L3 hold the pages a turn failed to reuse, or was it never asked?

kvd's counters cannot answer this. `misses_total` moves only on the get path, and
prefetch gets nothing it has not already confirmed with `batch_exists`, so a
negative exists -- or a prefetch abandoned before it queried at all -- leaves
every counter at zero. See README §5.2.

This asks L3 directly. It rebuilds a turn's page hashes the way
`hiradix_cache.query_storage_hit_length` does and checks them against the live key
set of a kvd tablespace, then replays `_storage_hit_query`'s own walk so the
answer is comparable with the `cached_tokens` the run recorded.

Both halves are read-only. The tablespace index is an append-only newline-
delimited JSON journal, so the live set comes from replaying PUT/DEL over the file
-- no daemon, and nothing is written back. Safe to run against a region a live
daemon is still using: a torn final line stops the replay, which is what kvd's own
recovery does.

TWO WAYS TO GET A WRONG ANSWER, both guarded:

* The token sequence must be the engine's. The dataset records each turn's exact
  `prompt_tokens`, so a length mismatch aborts that turn instead of reporting it
  as absent.
* The key derivation must be the engine's. `RadixKey` is hashed in a bigram view
  when speculative decoding is on (`is_bigram=self.is_eagle`), which shifts every
  hash and drops one page. Hashing raw token ids matches *nothing* on an MTP
  deployment, which looks exactly like an empty L3. So both derivations are
  scored and the mismatched one is printed as a control: if the control is 0 and
  the primary is not, the primary is the engine's.

    python probe_l3_page_hashes.py --trace t.json --model M --journal .../index.log

`--no-bigram` if the deployment runs without speculative decoding.
"""

import argparse
import json


def load_live_keys(paths: list[str]) -> set[str]:
    """Replay a tablespace journal to the set of bare KV page hashes it still holds."""
    live: set[str] = set()
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    op = json.loads(line)
                    key = bytes.fromhex(op["key_hex"]).decode("utf-8")
                except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                    break  # torn tail; everything before it is still valid
                if "." in key:
                    continue  # .indexer / .draft companions of the same page
                if op.get("op") == "PUT":
                    live.add(key)
                elif op.get("op") == "DEL":
                    live.discard(key)
    return live


def leading_run(hashes: list[str], live: set[str]) -> int:
    """Pages present before the first absent one -- what batch_exists returns."""
    n = 0
    for h in hashes:
        if h not in live:
            break
        n += 1
    return n


def simulate_hit_query(hashes: list[str], live: set[str], page: int, batch: int) -> int:
    """Tokens `_storage_hit_query` would return, had it run to completion."""
    total = 0
    for start in range(0, len(hashes), batch):
        chunk = hashes[start : start + batch]
        hit = leading_run(chunk, live)
        total += hit * page
        if hit < len(chunk):
            break
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="agentic-trace dataset json")
    ap.add_argument("--model", required=True, help="tokenizer path (the served model)")
    ap.add_argument("--journal", required=True, nargs="+", help="pool*/index.log")
    ap.add_argument("--details", help="run .jsonl, to print what the engine reused")
    ap.add_argument("--num-conv", type=int, default=20)
    ap.add_argument("--turn", type=int, default=0, help="turn index within each conv")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--storage-batch-size", type=int, default=128)
    ap.add_argument("--no-bigram", action="store_true", help="no speculative decoding")
    args = ap.parse_args()

    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.utils import get_hash_str
    from transformers import AutoTokenizer

    live = load_live_keys(args.journal)
    print(f"L3 live KV page hashes: {len(live):,}")
    if not live:
        print("  journal replayed empty -- wrong path, or the region was cleared.")
        return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    convs = json.load(open(args.trace))["conversations"][: args.num_conv]

    recorded: list[int] = []
    if args.details:
        cached = json.load(open(args.details))["cached_tokens"]
        i = 0
        for conv in convs:
            recorded.append(cached[i + args.turn])
            i += len(conv)

    bigram = not args.no_bigram
    print(f"key derivation: RadixKey(is_bigram={bigram}), page_size={args.page_size}\n")
    head = f"{'conv':>4} {'pages':>6} {'in L3':>6} {'lead':>6} {'control':>8} {'L3 could serve':>15}"
    print(head + (f" {'engine used':>12} {'ratio':>7}" if recorded else ""))

    rows = []
    for ci, conv in enumerate(convs):
        turn = conv[args.turn]
        text = tok.apply_chat_template(
            turn["messages"], tokenize=False, add_generation_prompt=True
        )
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) != turn["prompt_tokens"]:
            print(f"{ci:>4}  {len(ids)} tokens != recorded {turn['prompt_tokens']}"
                  " -- chat template drift, turn skipped")
            continue

        def hashes_for(is_bigram: bool) -> list[str]:
            key = RadixKey(ids, is_bigram=is_bigram).page_aligned(args.page_size)
            return get_hash_str(key, None, page_size=args.page_size)

        hs = hashes_for(bigram)
        control = sum(h in live for h in hashes_for(not bigram))
        present, lead = sum(h in live for h in hs), leading_run(hs, live)
        could = simulate_hit_query(hs, live, args.page_size, args.storage_batch_size)
        line = (f"{ci:>4} {len(hs):>6} {present:>6} {lead:>6} {control:>8} "
                f"{could:>15,}")
        if recorded:
            got = recorded[ci]
            line += f" {got:>12,} {(got / could if could else 0):>6.1%}"
            rows.append((len(hs), lead, could, got))
        else:
            rows.append((len(hs), lead, could, None))
        print(line)

    if not rows:
        return 1
    whole = sum(1 for r in rows if r[1] == r[0])
    print(f"\nturns present in L3 end to end: {whole} / {len(rows)}")
    print(f"tokens L3 could have served: {sum(r[2] for r in rows):,}")
    if recorded:
        served, could = sum(r[3] for r in rows), sum(r[2] for r in rows)
        print(f"tokens the engine reused:    {served:,}  ({served / max(could,1):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
