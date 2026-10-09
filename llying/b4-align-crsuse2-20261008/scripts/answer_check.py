#!/usr/bin/env python3
"""Measure the known-answer error rate of the live stack (decode at real acceptance).

Same prompt format as placement_answer_probe.py (400-record registry, temperature
0). Two parts:
  replay  the probe question that failed in b4-crsuse2-136-138-20261008T0956Z,
          sent REPLAY_COLD times after flushing P and D caches, then REPLAY_WARM
          times without flushing;
  sample  SAMPLE new registries, each asked twice in one session (the second
          turn reuses the P/D prefix).
Each answer is joined to its P/D rank and cache hits from the aus diagnostics.
Writes RUN/answer-check.json; never fails the run on wrong answers.
"""

import collections, json, os, subprocess, sys, urllib.error, uuid
from pathlib import Path

E = os.environ
RUN = Path(E["RUN"])
sys.path.insert(0, str(Path(E["TRACE_RUNTIME"]) / "scripts"))
import radix_gate as probe  # noqa: E402
from analyze import jsonl  # noqa: E402

ROUTER = f"http://{E['PREFILL_IP']}:28000"
FAILED_TAG, FAILED_INDEX = "ae9d022509a8446db6f86ababe9878dd", 5
REPLAY_COLD, REPLAY_WARM, SAMPLE = 4, 4, 64


def ask(tag, index, turn, session):
    """One question of placement_answer_probe.py's batch loop."""
    seed, identity = index + 1, f"{tag}-{index}"
    doc = "Registry " + identity + "\n" + "\n".join(
        f"Record {j}: city {probe.CITIES[j % len(probe.CITIES)]}; code Z{(j * 31337 + seed * 97) % 9973:04d}."
        for j in range(400))
    record = (index * 17 + 311 + turn * 23) % 400
    answer = f"Z{(record * 31337 + seed * 97) % 9973:04d}"
    probe.SESSION_ID = session
    row = {"tag": tag, "index": index, "turn": turn, "record": record, "expected": answer}
    try:
        reply = probe.chat(ROUTER, E["SERVED_MODEL"], [
            {"role": "system", "content": doc},
            {"role": "user", "content": f"What is the code in record {record}? Reply with only the code."}], 512)
    except urllib.error.HTTPError as e:  # e.g. a failed KV transfer: counted, not an answer
        return dict(row, correct=False, error=f"HTTP {e.code}", rid=None, content="")
    return dict(row, correct=reply["content"].strip() == answer, error=None, **reply)


def flush_all():
    for w in json.loads((RUN / "placement-resolved.json").read_text()):
        probe.flush(w["url"])


def main():
    results = []
    for i in range(REPLAY_COLD + REPLAY_WARM):
        if i < REPLAY_COLD:
            flush_all()
        results.append(dict(ask(FAILED_TAG, FAILED_INDEX, 0, f"answers-replay-{i}"), part="replay", cold=i < REPLAY_COLD))
    tag = uuid.uuid4().hex
    for index in range(SAMPLE):
        for turn in range(2):
            results.append(dict(ask(tag, index, turn, f"answers-{tag}-{index}"), part="sample"))
        (RUN / "answer-check-progress.json").write_text(json.dumps(results, indent=2) + "\n")

    subprocess.run([sys.executable, str(Path(E["TRACE_RUNTIME"]) / "scripts/capture_placement.py")], check=True)
    events = collections.defaultdict(dict)
    for role in ["prefill", "decode"]:
        for path in (RUN / "diagnostics" / role).glob("*.jsonl"):
            for e in jsonl(path):
                if e.get("event") in ("request_summary", "cache", "decode_prefix"):
                    events[e.get("rid")][f"{role}:{e['event']}"] = e
    for r in results:
        ev = events.get(r["rid"], {})
        r["p_rank"] = ev.get("prefill:request_summary", {}).get("dp_rank")
        r["d_rank"] = ev.get("decode:request_summary", {}).get("dp_rank")
        cache = ev.get("prefill:cache", {})
        r["p_cached"] = cache.get("cached_device", 0) + cache.get("cached_host", 0)
        r["d_prefix"] = ev.get("decode:decode_prefix", {}).get("prefix_tokens")

    def rate(rows):
        errors = sum(r["error"] is not None for r in rows)
        return {"requests": len(rows), "errors": errors, "wrong": sum(not r["correct"] for r in rows) - errors}

    sample = [r for r in results if r["part"] == "sample"]
    summary = {
        "replay_cold": rate([r for r in results if r["part"] == "replay" and r["cold"]]),
        "replay_warm": rate([r for r in results if r["part"] == "replay" and not r["cold"]]),
        "sample": rate(sample),
        "sample_turn0": rate([r for r in sample if r["turn"] == 0]),
        "sample_turn1": rate([r for r in sample if r["turn"] == 1]),
        "sample_same_rank": rate([r for r in sample if r["p_rank"] is not None and r["p_rank"] == r["d_rank"]]),
        "sample_cross_rank": rate([r for r in sample if r["p_rank"] is not None and r["p_rank"] != r["d_rank"]]),
        "sample_unjoined": sum(r["p_rank"] is None or r["d_rank"] is None for r in sample),
        "not_correct": [{k: r[k] for k in ["part", "index", "turn", "record", "expected", "content", "error",
                                           "p_rank", "d_rank", "p_cached", "d_prefix"]}
                        for r in results if not r["correct"]],
    }
    (RUN / "answer-check.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
