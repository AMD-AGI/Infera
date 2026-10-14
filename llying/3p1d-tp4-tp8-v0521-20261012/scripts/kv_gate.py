#!/usr/bin/env python3
"""Known-answer KV-transfer gate through the router (decode at REAL acceptance).

Each request carries its own registry document (unique seed, so no prefix is shared
between requests) of N records "Record j: city C; code Zdddd." and asks for one
record's code at temperature 0. Short documents (~40k tokens) span several prefill
chunks; long ones (~115k tokens) approach the AgentX mean input. With prefill DP4,
decode DP8 and PD rank affinity off, almost every transfer crosses ranks, so a
correct answer means the decode rank received the prefill rank's KV intact.

Wrong answers are split into "other_record" (the code of another record in the same
document: a retrieval miss, which a correct deployment also makes on long documents)
and "fabricated" (no record has that code, or garbage: the signature of corrupted KV);
"missing_prefix" = the right digits without the leading "Z" (the first, prefill-made token).

Usage: kv_gate.py --url ROUTER_URL --model NAME --out FILE.json
                  [--short 40] [--long 8] [--parallel 8] [--min-correct 0.95] [--max-fabricated 0]
Exit 0 when there are no HTTP errors, at most --max-fabricated fabricated answers and
at least --min-correct of answers match.
"""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request

CITIES = ("Lisbon", "Osaka", "Quito", "Tallinn", "Nairobi", "Hobart", "Bergen",
          "Cusco", "Dakar", "Perth", "Tbilisi", "Valparaiso", "Leipzig", "Hanoi")
SHORT_RECORDS, LONG_RECORDS = 2800, 8000


def code(record: int, seed: int) -> str:
    return f"Z{(record * 31337 + seed * 97) % 9973:04d}"


def document(seed: int, records: int) -> str:
    lines = [f"Registry {seed}"]
    lines += [f"Record {j}: city {CITIES[j % len(CITIES)]}; code {code(j, seed)}."
              for j in range(records)]
    return "\n".join(lines)


def ask(url: str, model: str, seed: int, records: int, timeout: float) -> dict:
    record = (seed * 997 + 123) % records
    expected = code(record, seed)
    body = {
        "model": model, "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": document(seed, records)},
            {"role": "user", "content": f"What is the code in record {record}? Reply with only the code."},
        ],
    }
    row = {"seed": seed, "records": records, "record": record, "expected": expected}
    started = time.monotonic()
    try:
        request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return dict(row, verdict="error", error=str(exc)[:300],
                    elapsed_s=round(time.monotonic() - started, 2))
    message = payload["choices"][0].get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content") or ""
    found = re.findall(r"Z\d{4}", content)
    if content.rstrip(".") == expected:
        verdict = "exact"
    elif found and found[-1] == expected:
        verdict = "contains"
    elif not content and expected in reasoning:
        verdict = "reasoning_only"
    else:
        guess = found[-1] if found else ("Z" + content.rstrip(".") if re.fullmatch(r"\d{4}\.?", content) else "")
        if guess == expected:
            verdict = "missing_prefix"   # right digits without the leading "Z" (first token)
        elif guess and any(code(j, seed) == guess for j in range(records)):
            verdict = "other_record"
        else:
            verdict = "fabricated"
    return dict(row, verdict=verdict, content=content[:200],
                finish=payload["choices"][0].get("finish_reason"),
                prompt_tokens=(payload.get("usage") or {}).get("prompt_tokens"),
                elapsed_s=round(time.monotonic() - started, 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--short", type=int, default=40)
    parser.add_argument("--long", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--min-correct", type=float, default=0.95)
    parser.add_argument("--max-fabricated", type=int, default=0)
    parser.add_argument("--short-records", type=int, default=SHORT_RECORDS)
    parser.add_argument("--long-records", type=int, default=LONG_RECORDS)
    args = parser.parse_args()
    base = int(time.time()) % 100000 * 100
    jobs = [(base + i, args.short_records) for i in range(args.short)]
    jobs += [(base + 10000 + i, args.long_records) for i in range(args.long)]
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda job: ask(args.url, args.model, job[0], job[1], args.timeout), jobs))
    counts = {}
    for row in results:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    answered = len(results) - counts.get("error", 0)
    correct = counts.get("exact", 0) + counts.get("contains", 0) + counts.get("reasoning_only", 0)
    summary = {
        "requests": len(results), "counts": counts,
        "correct_rate": round(correct / len(results), 4) if results else 0.0,
        "prompt_tokens_min_max": [
            min((r.get("prompt_tokens") or 0) for r in results),
            max((r.get("prompt_tokens") or 0) for r in results)],
        "passed": counts.get("error", 0) == 0 and answered > 0
        and counts.get("fabricated", 0) <= args.max_fabricated
        and correct / len(results) >= args.min_correct,
        "not_correct": [r for r in results if r["verdict"] in ("missing_prefix", "other_record", "fabricated", "error")],
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump({"summary": summary, "results": results}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "not_correct"}, indent=2))
    for row in summary["not_correct"]:
        print("NOT CORRECT:", json.dumps(row))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
