#!/usr/bin/env python3
"""Two arms in, one accept-or-reject out, with the numbers that produced it.

The arithmetic lives in `assets/lib/eval_stats.py` and not here, because
`check_no_regression` recomputes every line of this report and the two agreeing
is the whole point of that validator — which cannot happen if each carries its
own copy of the statistics.

**What counts as a regression, and what does not.**

  llm-eval    the Newcombe interval on the difference excludes zero AND the
              difference is negative. Not "the score went down": two runs of the
              same deployment differ by several points at 200 questions, and
              treating that as evidence would make this a coin-flip detector.
  needle      a depth the stock arm retrieved and the patched arm did not. Both
              failing is not a regression — it is a property of the model at that
              length, which is exactly why the frontier run exists and exactly
              why it is not gated.
  smoke       any check that passed on stock and failed on patched.
  performance a relative change worse than the bar, compared round for round.
              Round 1 against round 1 only: cold and warm differ by an order of
              magnitude on this trace.

Nothing here judges an arm on its own. An absolute eval score has no external
baseline, an absolute throughput number has no meaning without the configuration
it was measured on, and both arms were measured in the same session precisely so
that neither has to stand alone.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import eval_stats  # noqa: E402

#: Metric -> (label, higher_is_better). Only metrics that describe the SERVICE.
#: `output_token_throughput_tps` is the headline; the latencies are there because
#: a change can trade one for the other and a single metric would hide it.
PERF_METRICS = {
    "output_token_throughput_tps": ("output token throughput", True),
    "request_throughput_rps": ("request throughput", True),
    "ttft_ms": ("time to first token", False),
    "inter_token_latency_ms": ("inter-token latency", False),
    "request_latency_ms": ("request latency", False),
}

#: Which column of the AIPerf summary each metric is read from. A throughput has
#: no percentile; a latency's p95 is the one that moves first under a queue.
PERF_COLUMNS = {"ttft_ms": ("avg", "p90"), "inter_token_latency_ms": ("avg",),
                "request_latency_ms": ("avg", "p90")}


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_eval_index(result: Path) -> dict[str, dict]:
    """`.index` plus each result json: eval name -> {scored, score}.

    `scored` comes from the index because it was counted out of the html report;
    the dataset size is the wrong answer (GSM8K ships 1319 rows, `gsm8k` scores
    1314 and `mixed_prefix_gsm8k` 1299) and the result json does not carry it.
    """
    out: dict[str, dict] = {}
    index = result / "lm_eval" / ".index"
    if not index.is_file():
        return out
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name, scored, filename = parts[0], parts[1], parts[2]
        payload = read_json(result / "lm_eval" / filename, {}) or {}
        score = payload.get("score", payload.get("mean_score"))
        out[name] = {
            "scored": int(scored or 0),
            "score": float(score) if score is not None else None,
            "seconds": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
        }
    return out


def needle_map(needle: dict) -> dict[str, dict]:
    """(run label, depth) -> the depth record, flattened for comparison."""
    out = {}
    for run in (needle or {}).get("runs", []):
        for depth in run.get("depths", []):
            out[f"{run['label']}@{depth['depth']}"] = depth
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("accept-stock", "accept-patched", "bench-stock", "bench-patched", "patch"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    # The same two package variables `check_no_regression` reads out of its own
    # args block. One source, because that validator recomputes this report's
    # verdict and fails when it disagrees — two sources would make them differ
    # over a threshold nobody changed.
    ap.add_argument("--max-throughput-regression", type=float,
                    default=float(os.environ.get("IT_MAX_THROUGHPUT_REGRESSION", 0.05)))
    ap.add_argument("--max-latency-regression", type=float,
                    default=float(os.environ.get("IT_MAX_TTFT_REGRESSION", 0.10)))
    args = ap.parse_args()

    out = Path(args.out)
    arms = {
        "stock": {"accept": Path(args.accept_stock), "bench": Path(args.bench_stock)},
        "patched": {"accept": Path(args.accept_patched), "bench": Path(args.bench_patched)},
    }
    manifest = read_json(Path(args.patch) / "items" / "codes" / "manifest.json", {}) or {}

    loaded = {}
    for arm, paths in arms.items():
        result = paths["accept"] / "items" / "result"
        loaded[arm] = {
            "smoke": read_json(result / "smoke.json", {}) or {},
            "needle": read_json(result / "needle.json", {}) or {},
            "probe": read_json(result / "probe.json", {}) or {},
            "evals": read_eval_index(result),
            "bench": paths["bench"] / "items" / "result",
            "steps": read_json(paths["accept"] / "items" / "env" / "steps.json", {}) or {},
            "context": read_json(paths["accept"] / "items" / "env" / "context.json", {}) or {},
        }

    reasons: list[str] = []

    # ---- smoke ---------------------------------------------------------------
    smoke_rows = []
    stock_checks = {c["name"]: c["ok"] for c in loaded["stock"]["smoke"].get("checks", [])}
    patched_checks = {c["name"]: c["ok"] for c in loaded["patched"]["smoke"].get("checks", [])}
    for name in sorted(set(stock_checks) | set(patched_checks)):
        a, b = stock_checks.get(name), patched_checks.get(name)
        verdict = "REGRESSED" if (a and not b) else ("same" if a == b else "differs")
        smoke_rows.append({"check": name, "stock": a, "patched": b, "verdict": verdict})
        if verdict == "REGRESSED":
            reasons.append(f"smoke check {name!r} passed on stock and failed on patched")

    # ---- needle --------------------------------------------------------------
    needle_rows = []
    a_map, b_map = needle_map(loaded["stock"]["needle"]), needle_map(loaded["patched"]["needle"])
    for key in sorted(set(a_map) | set(b_map)):
        a, b = a_map.get(key, {}), b_map.get(key, {})
        gated = key.startswith("gated@")
        regressed = bool(a.get("ok")) and not bool(b.get("ok"))
        needle_rows.append({
            "run": key,
            "gated": gated,
            "stock_ok": a.get("ok"),
            "patched_ok": b.get("ok"),
            "stock_prompt_tokens": a.get("prompt_tokens"),
            "patched_prompt_tokens": b.get("prompt_tokens"),
            "verdict": "REGRESSED" if regressed else "same",
        })
        if regressed:
            where = "gated" if gated else "frontier"
            reasons.append(f"needle {key} ({where}) retrieved on stock and not on patched")
    # A depth failing on BOTH arms is NOT a regression and is not reported as one.
    # Retrieval at any given depth is not a stable property of this deployment --
    # measured, nine times, in assets/accept/needle.py's table -- so the only
    # thing the needle can say about a patch is that a depth stopped working.
    # What it can say about the deployment is that nothing came back at all, and
    # that floor is check_acceptance's, enforced per arm.

    # ---- llm-eval ------------------------------------------------------------
    # The probe decides whether a score can be read at all. It catches failures
    # that produce a NUMBER rather than an error -- a reasoning parser eating the
    # answer, a shared prefix changing it -- and a score measured through one of
    # those is indistinguishable from a real regression. So a failed probe on
    # either arm makes every eval comparison `uninterpretable`, which is a reason
    # to reject rather than a quiet `same`.
    probe_failed = [arm for arm in arms if not loaded[arm]["probe"].get("ok")]
    eval_rows = []
    for name in sorted(set(loaded["stock"]["evals"]) | set(loaded["patched"]["evals"])):
        a = loaded["stock"]["evals"].get(name)
        b = loaded["patched"]["evals"].get(name)
        if not a or not b or a["score"] is None or b["score"] is None:
            eval_rows.append({"name": name, "verdict": "unmeasured",
                              "stock": a, "patched": b})
            reasons.append(f"eval {name!r} did not produce a score on both arms")
            continue
        row = eval_stats.compare_scores(a["score"], a["scored"], b["score"], b["scored"])
        row["name"] = name
        if probe_failed:
            row["verdict"] = "uninterpretable"
            row["uninterpretable_because"] = f"probe failed on: {', '.join(probe_failed)}"
        eval_rows.append(row)
        if row["verdict"] == "REGRESSED":
            reasons.append(
                f"eval {name!r} regressed: {row['score_a']:.3f} -> {row['score_b']:.3f}, "
                f"95% CI of the difference {row['ci95_delta']} excludes zero"
            )
    if probe_failed:
        for arm in probe_failed:
            for failure in loaded[arm]["probe"].get("failures") or ["(none recorded)"]:
                reasons.append(f"probe failed on the {arm} arm: {failure}")
        reasons.append(
            "every eval comparison is uninterpretable while a probe is failing: the "
            "failures it catches produce a score rather than an error, and that score "
            "cannot be told apart from a real regression"
        )

    # The gap between gsm8k and its shared-prefix variant measures prefix reuse
    # without needing an external baseline: the same questions, one set behind
    # partially shared few-shot prefixes.
    prefix_reuse = {}
    for arm in ("stock", "patched"):
        evals = loaded[arm]["evals"]
        if "gsm8k" in evals and "mixed_prefix_gsm8k" in evals:
            base, mixed = evals["gsm8k"], evals["mixed_prefix_gsm8k"]
            if base["score"] is not None and mixed["score"] is not None:
                prefix_reuse[arm] = round(mixed["score"] - base["score"], 6)

    # ---- performance ---------------------------------------------------------
    perf_rows = []
    rounds = sorted(
        {p.name for arm in arms for p in (loaded[arm]["bench"]).glob("r*") if p.is_dir()}
    )
    for round_name in rounds:
        summaries = {
            arm: (read_json(loaded[arm]["bench"] / round_name / "summary.json", {}) or {})
            .get("metrics", {})
            for arm in ("stock", "patched")
        }
        for metric, (label, higher_better) in PERF_METRICS.items():
            for column in PERF_COLUMNS.get(metric, ("avg",)):
                a = summaries["stock"].get(metric, {}).get(column)
                b = summaries["patched"].get(metric, {}).get(column)
                bar = args.max_throughput_regression if higher_better else args.max_latency_regression
                row = eval_stats.perf_verdict(a, b, max_regression=bar, higher_is_better=higher_better)
                row.update(round=round_name, metric=metric, column=column, label=label, bar=bar)
                perf_rows.append(row)
                if row["verdict"] == "REGRESSED":
                    reasons.append(
                        f"{label} ({column}) in {round_name}: {a} -> {b}, "
                        f"{row['rel_delta']:+.1%} against a bar of {bar:.0%}"
                    )
        counts = {
            arm: summaries[arm].get("request_count", {}).get("avg") for arm in ("stock", "patched")
        }
        perf_rows.append({
            "round": round_name, "metric": "request_count", "column": "avg",
            "label": "requests replayed", "stock": counts["stock"], "patched": counts["patched"],
            "rel_delta": eval_stats.relative_change(counts["stock"], counts["patched"]),
            "bar": None,
            # Not a pass/fail: the trace decides how many requests there are, and a
            # difference here means the two arms did not replay the same work,
            # which invalidates the comparison rather than failing the patch.
            "verdict": "context",
        })
        if counts["stock"] and counts["patched"] and counts["stock"] != counts["patched"]:
            reasons.append(
                f"{round_name}: the two arms replayed different request counts "
                f"({counts['stock']} vs {counts['patched']}) — they are not comparable"
            )

    # ---- did both arms do the same things in the same order? -----------------
    order = {arm: [s["step"] for s in loaded[arm]["steps"].get("steps", [])] for arm in arms}
    if order["stock"] != order["patched"]:
        reasons.append(
            f"the two arms ran different sequences: {order['stock']} vs {order['patched']}"
        )

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "patch": {
            "operator_id": manifest.get("operator_id"),
            "logical_operator": manifest.get("logical_operator"),
            "apply_mode": manifest.get("apply_mode"),
            "image": manifest.get("image"),
            "files": len(manifest.get("files", [])),
            "declares_runtime_marker": bool(manifest.get("runtime_marker")),
            "expect": manifest.get("expect"),
        },
        "arms": {
            arm: {
                "node": loaded[arm]["context"].get("node"),
                "endpoint": loaded[arm]["context"].get("endpoint"),
                "sequence": order[arm],
                "probe_ok": loaded[arm]["probe"].get("ok"),
                "smoke_ok": loaded[arm]["smoke"].get("ok"),
                "needle_ok": loaded[arm]["needle"].get("ok"),
            }
            for arm in arms
        },
        "bars": {
            "max_throughput_regression": args.max_throughput_regression,
            "max_latency_regression": args.max_latency_regression,
            "eval_confidence": 0.95,
        },
        "correctness": {
            "smoke": smoke_rows,
            "needle": needle_rows,
            "evals": eval_rows,
            "prefix_reuse_delta": prefix_reuse,
        },
        "performance": perf_rows,
        "verdict": {"accepted": not reasons, "reasons": reasons},
    }

    items = out / "items"
    items.mkdir(parents=True, exist_ok=True)
    (items / "text.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (items / "schema").write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "integration_report",
                "type": "object",
                "required": ["generated_at", "patch", "arms", "bars", "correctness",
                             "performance", "verdict"],
                "properties": {
                    "patch": {"type": "object",
                              "description": "identity of the change under test"},
                    "arms": {"type": "object",
                             "description": "one entry per arm, and the sequence it ran"},
                    "bars": {"type": "object",
                             "description": "the thresholds the verdict was decided against"},
                    "correctness": {
                        "type": "object",
                        "required": ["smoke", "needle", "evals"],
                        "description": (
                            "smoke and needle are per-check arm comparisons; evals carry "
                            "Wilson intervals per arm and a Newcombe interval on the "
                            "difference. verdict is one of same / REGRESSED / improved / "
                            "unmeasured."
                        ),
                    },
                    "performance": {
                        "type": "array",
                        "description": (
                            "one row per round x metric x column. rel_delta is "
                            "(patched - stock) / stock. Rows with verdict 'context' are "
                            "reported and not judged."
                        ),
                    },
                    "verdict": {
                        "type": "object",
                        "required": ["accepted", "reasons"],
                        "description": (
                            "accepted is true when reasons is empty. check_no_regression "
                            "recomputes this from the numbers above and fails if it "
                            "disagrees."
                        ),
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- the human-readable half --------------------------------------------
    lines = [
        f"# Integration verdict: {'ACCEPTED' if not reasons else 'REJECTED'}",
        "",
        f"Patch `{manifest.get('operator_id')}` "
        f"({manifest.get('logical_operator')}), {len(manifest.get('files', []))} file(s), "
        f"applied as `{manifest.get('apply_mode')}` to `{manifest.get('image')}`.",
        "",
        "Both arms were measured in the same session on "
        f"`{loaded['stock']['context'].get('node')}`, in the same order, against the same "
        "trace. Neither arm's numbers mean anything on their own.",
        "",
        "## Correctness",
        "",
        "| check | stock | patched | verdict |",
        "|---|---|---|---|",
    ]
    for row in smoke_rows:
        lines.append(f"| smoke: {row['check']} | {row['stock']} | {row['patched']} | {row['verdict']} |")
    for row in needle_rows:
        tag = "needle" if row["gated"] else "needle (frontier, not gated)"
        lines.append(
            f"| {tag}: {row['run']} | {row['stock_ok']} | {row['patched_ok']} | {row['verdict']} |"
        )
    lines += ["", "| eval | scored | stock | patched | delta | 95% CI of delta | verdict |",
              "|---|---|---|---|---|---|---|"]
    for row in eval_rows:
        if row.get("verdict") == "unmeasured":
            lines.append(f"| {row['name']} | — | — | — | — | — | unmeasured |")
            continue
        lines.append(
            f"| {row['name']} | {row['scored_a']}/{row['scored_b']} | {row['score_a']:.3f} | "
            f"{row['score_b']:.3f} | {row['delta']:+.3f} | "
            f"[{row['ci95_delta'][0]:+.3f}, {row['ci95_delta'][1]:+.3f}] | {row['verdict']} |"
        )
    if prefix_reuse:
        lines += [
            "",
            "Prefix-reuse delta (`mixed_prefix_gsm8k` − `gsm8k`): "
            + ", ".join(f"{arm} {value:+.3f}" for arm, value in prefix_reuse.items())
            + ". The same questions behind partially shared few-shot prefixes; a "
            "difference here is prefix reuse changing answers, which nothing else "
            "in this suite can see.",
        ]

    lines += ["", "## Performance", "",
              "| round | metric | stock | patched | change | bar | verdict |",
              "|---|---|---|---|---|---|---|"]
    for row in perf_rows:
        change = "—" if row["rel_delta"] is None else f"{row['rel_delta']:+.1%}"
        bar = "—" if row.get("bar") is None else f"{row['bar']:.0%}"
        lines.append(
            f"| {row['round']} | {row['label']} ({row['column']}) | {row['stock']} | "
            f"{row['patched']} | {change} | {bar} | {row['verdict']} |"
        )
    lines += [
        "",
        "Round 1 is cold for this trace and round 2 is warm; they are compared only "
        "against their own counterpart. On this trace and this deployment the two "
        "differ by roughly an order of magnitude, because a Mooncake trace carries "
        "`hash_ids` and prefix hit rate decides how much prefill there is to do.",
        "",
        "## Verdict",
        "",
    ]
    if reasons:
        lines.append("Rejected. Reasons:")
        lines += [f"- {reason}" for reason in reasons]
    else:
        lines.append(
            "Accepted: no smoke check, needle depth, eval interval or performance metric "
            "moved past its bar."
        )
    if manifest.get("expect", {}).get("source") == "mock":
        lines += [
            "",
            "**The patch under test is a mock.** It changes no arithmetic, so the correct "
            "answer here is 'no difference' and an accepted verdict is evidence that the "
            "comparison does not invent regressions — not evidence about any real "
            "optimisation.",
        ]
    (items / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "README.md").write_text(
        """# integration_report

## Purpose

The two arms of the integration test side by side, and the accept-or-reject
argument built from them. This is what the stage exists to produce.

Nothing here judges an arm on its own. An absolute eval score has no external
baseline and an absolute throughput figure has no meaning without the
configuration behind it; both arms were measured in one session, on one node,
against one trace, in one order, so that neither has to.

`items/report.md` is the same content for a reader. `items/text.json` is what
`check_no_regression` recomputes — it does not read the `verdict` field, it
rebuilds it from the numbers and fails if the two disagree.

## Schema

`items/schema` is the JSON Schema. In prose:

- `patch` — what was under test: operator, apply mode, image, file count, and
  whether it declared runtime markers.
- `arms` — per arm, the node, the endpoint, the step sequence, and the three
  self-reported pass flags.
- `bars` — the thresholds the verdict used. Recorded because a verdict without
  the bar it was measured against cannot be re-read later.
- `correctness.smoke` / `correctness.needle` — per-check arm comparison.
  `verdict` is `REGRESSED` only when stock passed and patched did not; both
  failing is not a regression, and for the frontier needle length it is the
  measured expectation.
- `correctness.evals` — per eval, both arms' scores with Wilson intervals, and
  the difference with a Newcombe interval. `REGRESSED` means the interval
  excludes zero and the difference is negative.
- `correctness.prefix_reuse_delta` — `mixed_prefix_gsm8k` minus `gsm8k` per arm.
- `performance` — one row per round, metric and column. `rel_delta` is
  `(patched - stock) / stock`. Rows with `verdict: context` are reported and not
  judged; the request count is one, because a difference there means the arms
  replayed different work rather than that the patch is slow.
- `verdict` — `accepted` plus every reason it is not.
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "redact.py"),
            str(out),
            f"TASK_PACKAGE={args.package}",
            f"ZONE={Path.cwd()}",
            "TMPDIR=/tmp",
            f"HOME={Path.home()}",
        ],
        check=True,
    )

    print(f"compare: {'ACCEPTED' if not reasons else 'REJECTED'}")
    for reason in reasons:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
