###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Convert the Inferact/codex_swebenchpro_traces HF dataset into the
deterministic_l3 corpus format so it can be replayed by `replay.py`.

Why this dataset is a good L2 validation target:

- It's 610 real Codex agent traces against SWE-bench-Pro.
- Per the dataset card, ~93.8 % of trials (568 / 610) share an
  ~11.5 K-token system prompt — every replay request fans out from
  the same prefix bytes, which is exactly the within-uptime
  kvd_key repeat pattern the kvd L2 cache is built for.
- The traces are recorded (not generated on the fly), so they are
  bit-deterministic across vLLM restarts and across runs.
- The conversations are strict `human` / `gpt` alternation, which
  trivially maps to OpenAI chat-completions `user` / `assistant`.

Tokenizer note: Codex tokenized the original 11.5 K shared prefix
under cl100k_base. We replay against (e.g.) MiniMax-M2.5 which has
its own tokenizer — so the *count* of tokens in MiniMax's view will
differ by ~10-15 %, but the **byte sequence** of the shared prefix
is identical across requests, so MiniMax will produce identical
block hashes for those bytes and our content-hashed kvd keys will
collide as intended.

Usage::

    python -m bench.kvcache.deterministic_l3.corpus_swebench \\
        --num-trials 16 --strategy smallest \\
        --max-turns-per-trial 16 --out /tmp/swebench-corpus.json

The output JSON is the same shape that `replay.py` expects:

    {
      "config":   {...},
      "sessions": [
        {"session_id": 0, "messages": [{"role":"user","content":...}, ...]},
        ...
      ]
    }

with `messages` always ending on a `user` turn (the next-token
target).

Download path: we use `huggingface_hub.hf_hub_download` (already
in our dependency tree via transformers) rather than `datasets`,
which would add a heavy dep just for one parquet/JSON pull.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from ._common import write_corpus

logger = logging.getLogger(__name__)

DATASET_REPO_ID = "Inferact/codex_swebenchpro_traces"
DATASET_FILENAME = "codex_swebenchpro.json"


def download_or_use_cached(
    repo_id: str = DATASET_REPO_ID, filename: str = DATASET_FILENAME
) -> Path:
    """Fetch the dataset's JSON blob via huggingface_hub. Re-downloads
    are no-ops (hub uses content-addressed snapshot dirs); returns the
    local path to the fully-resolved JSON file."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return Path(path)


def trial_to_session(trial: dict, session_id: int) -> dict | None:
    """Convert one dataset entry (a `{"conversations": [...]}` dict)
    into a `replay.py`-compatible session.

    Filters that drop the trial (returns None) when:

    - Missing or empty `conversations` field.
    - Any message has non-string `value` (tool calls, multimodal).
    - First message isn't from `human` (we need a user-led prompt).
    - Conversation ends on `gpt` (we need to send up to a user turn).
      The trace is truncated to the last user turn instead — we keep
      the trial.

    Role mapping: `human` → `user`, `gpt` → `assistant`. The trace
    format predates standardized OpenAI roles; the mapping is
    determined by inspection (every entry in the dataset uses
    these two literals).
    """
    convs = trial.get("conversations")
    if not isinstance(convs, list) or not convs:
        return None

    messages: list[dict] = []
    for entry in convs:
        if not isinstance(entry, dict):
            return None
        from_field = entry.get("from")
        value = entry.get("value")
        if not isinstance(value, str):
            return None
        if from_field == "human":
            role = "user"
        elif from_field == "gpt":
            role = "assistant"
        else:
            # Unknown role — bail rather than guess. If a future
            # dataset version introduces `system` we can extend here.
            return None
        messages.append({"role": role, "content": value})

    # We need at least one user turn, and the conversation must start
    # on user (a trace that opens with an assistant message is
    # malformed for our chat-completion replay).
    if not messages or messages[0]["role"] != "user":
        return None

    # Truncate trailing assistant runs — replay sends requests whose
    # last message is a user turn (engine is asked to generate the
    # next assistant). Drop the tail "assistant ... assistant" so the
    # last message is a user.
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages:
        return None

    return {"session_id": session_id, "messages": messages}


def cap_turns(session: dict, max_user_turns: int) -> dict:
    """Cap a session to at most `max_user_turns` user messages. Used
    to bound the bench's compute cost on a workload where mean trial
    size is 33 LLM calls and some trials exceed 100."""
    if max_user_turns <= 0:
        return session
    msgs = session["messages"]
    user_count = 0
    end = 0
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            user_count += 1
            if user_count > max_user_turns:
                break
        end = i + 1
    # Ensure we end on a user turn — same invariant as `trial_to_session`.
    capped = msgs[:end]
    while capped and capped[-1]["role"] == "assistant":
        capped.pop()
    return {"session_id": session["session_id"], "messages": capped}


def select_trials(
    trials: list[dict],
    n: int,
    strategy: str,
    seed: int,
) -> list[dict]:
    """Pick `n` trials from the candidate list using the given strategy.

    - ``smallest``: pick the N with the smallest total content bytes.
      Good for fast smoke runs.
    - ``largest``: invert — useful for stress-testing kvd capacity.
    - ``random``: seeded shuffle then take first N.
    - ``first``: just take the first N (matches the dataset's natural
      ordering; deterministic without needing a seed).
    """
    sized = [(sum(len(m["content"]) for m in t["messages"]), t) for t in trials]
    if strategy == "smallest":
        sized.sort(key=lambda x: x[0])
    elif strategy == "largest":
        sized.sort(key=lambda x: -x[0])
    elif strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(sized)
    elif strategy == "first":
        pass
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return [t for _, t in sized[:n]]


def build_corpus(
    raw_trials: list[dict],
    num_trials: int,
    strategy: str,
    max_turns_per_trial: int,
    seed: int,
) -> tuple[dict, dict]:
    """Pure-function pipeline so tests can feed a synthetic
    `raw_trials` list and exercise the full conversion without
    network access.

    Returns ``(corpus_dict, stats_dict)``. The corpus shape is the
    same one `replay.py` consumes; the stats dict has request /
    byte counts for logging.
    """
    converted: list[dict] = []
    skipped_invalid = 0
    for trial in raw_trials:
        sess = trial_to_session(trial, len(converted))
        if sess is None:
            skipped_invalid += 1
            continue
        converted.append(sess)

    selected = select_trials(converted, num_trials, strategy, seed)
    selected = [cap_turns(s, max_turns_per_trial) for s in selected]
    # Drop any trial that became empty after capping (shouldn't happen
    # with sane max_turns but defensive).
    selected = [s for s in selected if s["messages"]]
    # Re-id so session_id is dense [0..N-1].
    for i, s in enumerate(selected):
        s["session_id"] = i

    total_msgs = sum(len(s["messages"]) for s in selected)
    total_user = sum(1 for s in selected for m in s["messages"] if m["role"] == "user")
    total_bytes = sum(len(m["content"]) for s in selected for m in s["messages"])

    corpus = {
        "config": {
            "source_dataset": DATASET_REPO_ID,
            "num_trials_in_corpus": len(selected),
            "selection_strategy": strategy,
            "selection_seed": seed,
            "max_turns_per_trial": max_turns_per_trial,
            "raw_trials_skipped_invalid": skipped_invalid,
        },
        "sessions": selected,
    }
    stats = {
        "num_sessions": len(selected),
        "num_messages": total_msgs,
        "num_user_turns": total_user,  # this is the request count `replay.py` will issue
        "total_content_bytes": total_bytes,
        "skipped_invalid": skipped_invalid,
    }
    return corpus, stats


def _format_stats(stats: dict) -> str:
    mb = stats["total_content_bytes"] / (1024 * 1024)
    return (
        f"{stats['num_sessions']} sessions, "
        f"{stats['num_messages']} messages "
        f"({stats['num_user_turns']} user turns = requests), "
        f"{mb:.1f} MB total content, "
        f"{stats['skipped_invalid']} raw trials skipped as invalid"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-repo", default=DATASET_REPO_ID)
    p.add_argument("--dataset-filename", default=DATASET_FILENAME)
    p.add_argument(
        "--num-trials",
        type=int,
        default=16,
        help="Number of trials to include in the corpus",
    )
    p.add_argument(
        "--strategy",
        choices=["smallest", "largest", "random", "first"],
        default="smallest",
        help="How to pick num-trials out of all available "
        "(smallest = cheapest bench, largest = max kvd pressure)",
    )
    p.add_argument(
        "--max-turns-per-trial",
        type=int,
        default=16,
        help="Cap on user turns kept per trial. Some traces have "
        "100+ turns; truncating bounds replay wall time.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    json_path = download_or_use_cached(args.dataset_repo, args.dataset_filename)
    logger.info("loaded dataset from %s", json_path)
    with json_path.open() as f:
        raw_trials = json.load(f)
    logger.info("dataset has %d raw trials", len(raw_trials))

    corpus, stats = build_corpus(
        raw_trials=raw_trials,
        num_trials=args.num_trials,
        strategy=args.strategy,
        max_turns_per_trial=args.max_turns_per_trial,
        seed=args.seed,
    )
    write_corpus(args.out, corpus)
    logger.info("wrote %s — %s", args.out, _format_stats(stats))


if __name__ == "__main__":
    main()
