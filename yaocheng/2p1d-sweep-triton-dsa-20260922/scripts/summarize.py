#!/usr/bin/env python3
"""Summarize completed 2P1D points, or validate one aggregate before accepting it."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

FIELDS = ("conc", "gpus", "total_tok_s_gpu", "output_tok_s_gpu", "ttft_p50_s", "ttft_p90_s",
          "itl_p50_ms", "intvty_p50", "profiled", "errors", "duration_s")

def metrics(path, conc, duration=0):
    record = json.loads(path.read_text())
    expected = {"prefill_num_workers": 2, "decode_num_workers": 1,
                "prefill_tp": 8, "decode_tp": 8, "num_prefill_gpu": 16, "num_decode_gpu": 8,
                "conc": conc}
    for key, value in expected.items():
        if int(record[key]) != value:
            raise ValueError(f"{path}: {key}={record[key]}, expected {value}")
    throughput = record["request_metrics"]["throughput"]
    seconds = float(throughput["duration_seconds"])
    if duration and seconds < duration * 0.98:
        raise ValueError(f"{path}: truncated measurement: {seconds}s, requested {duration}s")
    total = float(throughput["total"]["tokens_per_second"]) / 24
    claimed = float(throughput["per_gpu"]["total_tput_tps"])
    if not math.isclose(total, claimed, rel_tol=1e-5):
        raise ValueError(f"{path}: per-GPU throughput is not normalized by 24 GPUs")
    lat = record["request_metrics"]["latency"]
    accounting = record["request_accounting"]
    row = {
        "conc": conc, "gpus": 24, "total_tok_s_gpu": total,
        "output_tok_s_gpu": float(throughput["output"]["tokens_per_second"]) / 24,
        "ttft_p50_s": float(lat["ttft"]["p50"]), "ttft_p90_s": float(lat["ttft"]["p90"]),
        "itl_p50_ms": float(lat["full_response_itl"]["p50"]) * 1000,
        "intvty_p50": float(lat["full_response_intvty"]["p50"]),
        "profiled": int(accounting["records_profiled"]),
        "errors": int(accounting["records_error_dropped"]), "duration_s": seconds,
    }
    if row["profiled"] <= 0 or any(not math.isfinite(v) or v < 0 for v in row.values()):
        raise ValueError(f"{path}: invalid/empty metrics")
    return row


def summarize(directory):
    rows, excluded = [], []
    for point in sorted(directory.glob("c[0-9]*")):
        if not point.is_dir() or not point.name[1:].isdigit():
            continue
        status_path = point / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("state") != "complete":
            excluded.append(f"- {point.name}: {status.get('state', 'incomplete')}（未计入性能结果）")
            continue
        conc = int(point.name[1:])
        rows.append(metrics(point / f"agentx_conc{conc}.json", conc, status["requested_duration_s"]))
    rows.sort(key=lambda r: r["conc"])
    with (directory / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    text = ["# 2P1D AgentX concurrency sweep", "",
            "只统计完成测量、24 GPU 元数据校验和前后服务检查的档位。", "",
            "total tok/s/GPU 包含输入和输出 token（含命中缓存的输入）；output tok/s/GPU 单列输出吞吐。",
            "默认使用模拟 acceptance 3.61，结果只用于性能对比，不代表生成正确性。", "",
            "| conc | GPUs | total tok/s/GPU | output tok/s/GPU | TTFT p50 s | TTFT p90 s | ITL p50 ms | intvty p50 | profiled | error dropped* |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        text.append(f"| {r['conc']} | 24 | {r['total_tok_s_gpu']:.2f} | {r['output_tok_s_gpu']:.2f} | "
                    f"{r['ttft_p50_s']:.3f} | {r['ttft_p90_s']:.3f} | {r['itl_p50_ms']:.3f} | "
                    f"{r['intvty_p50']:.2f} | {r['profiled']} | {r['errors']} |")
    if not rows:
        text.extend(["", "尚无完成的实测档位。"])
    text.extend(["", "*error dropped 为原始聚合的 records_error_dropped，可能包含预热阶段；不等同于正式测量失败数。详见 measurement_notes.md。", "", *excluded, ""])
    (directory / "summary.md").write_text("\n".join(text))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--conc", type=int)
    parser.add_argument("--duration", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.validate:
            if args.conc is None:
                parser.error("--validate requires --conc")
            print(json.dumps(metrics(args.validate, args.conc, args.duration), indent=2))
        elif args.directory:
            summarize(args.directory)
        else:
            parser.error("provide a run directory or --validate")
    except (OSError, ValueError, KeyError) as exc:
        print(f"summary: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
