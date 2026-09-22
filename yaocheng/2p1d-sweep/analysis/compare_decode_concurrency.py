"""Compare archived 1P1D/2P1D AgentX concurrency; never rewrite source results.

Run with python3 analysis/compare_decode_concurrency.py from any directory.
The 2P1D server gauge export lives in the kit's ignored .tmp/raw directory.
"""

import csv
import json
import os
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
REFERENCE = KIT.parents[1] / "yihou/glm52.p8d8.agentx-sweep.packup_20260920"
OUT = Path(__file__).resolve().parent
MANIFEST = json.loads((KIT / "results/run.json").read_text())
RAW = KIT / ".tmp/raw" / MANIFEST["run_id"]


def read_sections(path, header_start):
    """AIPerf CSVs have different headers for gauges/counters/histograms."""
    header = None
    with path.open() as stream:
        for values in csv.reader(line for line in stream if not line.startswith("#")):
            if not values:
                continue
            if values[0] == header_start:
                header = values
                continue
            if header is None or len(values) != len(header):
                raise ValueError(f"Unexpected CSV section in {path}")
            yield dict(zip(header, values))


def gauge(rows, metric, engine, mean=False):
    selected = [r for r in rows if r.get("Type") == "gauge"
                and r["Metric"] == "sglang:" + metric
                and r.get("engine_type") == engine]
    assert len(selected) == (8 if engine == "decode" else 16), (metric, engine)
    value = sum(float(r["avg"]) for r in selected)
    return value / len(selected) if mean else value


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


comparison, queues = [], []
for shape, kit, gpus in (("1P1D", REFERENCE, 16), ("2P1D", KIT, 24)):
    for point in sorted((kit / "results").glob("c[0-9]*")):
        conc = int(point.name[1:])
        exported = {r["Metric"]: r for r in read_sections(
            point / "profile_export_aiperf.csv", "Metric")}

        def stat(metric, column="avg"):
            r = exported[metric]
            return float(r.get(column, r.get("Value")))

        result = json.loads((point / f"agentx_conc{conc}.json").read_text())
        metrics = result["request_metrics"]
        count = result["request_accounting"]["records_profiled"]
        duration = metrics["throughput"]["duration_seconds"]
        prompt = stat("Total Usage Prompt Tokens (tokens)")
        cached = stat("Total Usage Prompt Cache Read Tokens (tokens)")
        row = {
            "Shape": shape, "ConcurrencySessionTrees": conc, "GPUs": gpus,
            "ClientEffectiveTotal": stat("Effective Concurrency"),
            "ClientPreFirstToken": stat("Effective Prefill Concurrency"),
            "ClientEffectiveDecode": stat("Effective Decode Concurrency"),
            "ServerDecodeRunning": "",
            "MeanTTFTSeconds": stat("Time to First Token (ms)") / 1000,
            "P50TTFTSeconds": stat("Time to First Token (ms)", "p50") / 1000,
            "P90TTFTSeconds": stat("Time to First Token (ms)", "p90") / 1000,
            "MeanDecodeSeconds": stat("Full Decode Duration (ms)") / 1000,
            "MeanISL": stat("Input Sequence Length (tokens)"),
            "MeanOSL": stat("Output Sequence Length (tokens)"),
            "UsageCacheHitPct": cached / prompt * 100,
            "TheoreticalCacheHitPct": metrics["cache"]["theoretical_cache_hit_rate"] * 100,
            "APIUncachedTokensPerSecond": (prompt - cached) / duration,
            "CompletedRequests": count, "CompletedRequestsPerSecond": count / duration,
            "TotalTokensPerSecond": metrics["throughput"]["total"]["tokens_per_second"],
            "OutputTokensPerSecond": metrics["throughput"]["output"]["tokens_per_second"],
            "Source": str((point / "profile_export_aiperf.csv").relative_to(KIT.parents[1])),
        }
        assert abs(row["UsageCacheHitPct"] - stat("Overall Usage Prompt Cache Read % (%)")) < 0.006
        if shape == "2P1D":
            source = RAW / point.name / "aiperf_artifacts/server_metrics_export.csv"
            server = list(read_sections(source, "Endpoint"))
            row["ServerDecodeRunning"] = gauge(server, "num_running_reqs", "decode")
            queues.append({
                "ConcurrencySessionTrees": conc,
                "DecodeRunning": row["ServerDecodeRunning"],
                "DecodePreallocQueue": gauge(server, "num_decode_prealloc_queue_reqs", "decode"),
                "DecodeTransferQueue": gauge(server, "num_decode_transfer_queue_reqs", "decode"),
                "DecodeReadyQueue": gauge(server, "num_queue_reqs", "decode"),
                "DecodeTokenUsageMeanPct": 100 * gauge(server, "token_usage", "decode", mean=True),
                "PrefillQueue": gauge(server, "num_queue_reqs", "prefill"),
                "PrefillBootstrapQueue": gauge(server, "num_prefill_bootstrap_queue_reqs", "prefill"),
                "PrefillHostKVUsageMeanPct": 100 * gauge(server, "hicache_host_used_tokens", "prefill", mean=True)
                / gauge(server, "hicache_host_total_tokens", "prefill", mean=True),
                "Source": str(source.relative_to(KIT)),
            })
            assert abs(row["ServerDecodeRunning"] - row["ClientEffectiveDecode"]) < 1
            # Same completed-record window: end-to-end = pre-first-token + decode.
            assert abs(row["ClientEffectiveTotal"] - row["ClientPreFirstToken"]
                       - row["ClientEffectiveDecode"]) < 0.02
        comparison.append(row)

write_csv("decode_concurrency_comparison.csv", comparison)
write_csv("2p1d_queue_metrics.csv", queues)

os.environ.setdefault("MPLCONFIGDIR", str(KIT / ".tmp/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
styles = {"1P1D": ("#bc5b26", "s"), "2P1D": ("#2466a4", "o")}
panels = [
    (axes[0, 0], "ClientEffectiveDecode", "Requests actively decoding", "Mean concurrent requests"),
    (axes[0, 1], "MeanTTFTSeconds", "Time spent before first token", "Mean TTFT (seconds)"),
    (axes[1, 0], "UsageCacheHitPct", "Prompt cache hit rate (API usage)", "Cached / input tokens (%)"),
]
for ax, field, title, ylabel in panels:
    for shape, (color, marker) in styles.items():
        rows = [r for r in comparison if r["Shape"] == shape]
        x, values = [r["ConcurrencySessionTrees"] for r in rows], [r[field] for r in rows]
        ax.plot(x, values, color=color, marker=marker, linewidth=2,
                label=f"{shape} ({rows[0]['GPUs']} GPUs)")
        if field == "ClientEffectiveDecode":
            for xi, y in zip(x, values):
                ax.annotate(f"{y:.1f}", (xi, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", color=color, fontsize=9)
    ax.set(title=title, ylabel=ylabel)
    ax.legend(frameon=False)
    ax.set_ylim((70, 100) if field == "UsageCacheHitPct" else (0, ax.get_ylim()[1] * 1.12))

ax = axes[1, 1]
for field, label, color, marker in (
    ("DecodeRunning", "Running", "#2466a4", "o"),
    ("DecodePreallocQueue", "Waiting for KV allocation", "#bc5b26", "s"),
    ("DecodeTransferQueue", "Waiting for KV readiness", "#69843d", "^"),
):
    ax.plot([r["ConcurrencySessionTrees"] for r in queues], [r[field] for r in queues],
            label=label, color=color, marker=marker, linewidth=2)
ax.set(title="2P1D decode: running and waiting queues", ylabel="Server mean, sum of 8 DP ranks")
ax.set_ylim(0, 155)
ax.legend(frameon=False)
for ax in axes.flat:
    ax.set_xlabel("Configured concurrency (AgentX session trees)")
    ax.set_xticks([80, 112, 144, 192, 256])
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("GLM-5.2 MXFP4: why configured concurrency exceeds active decode", fontsize=15)
fig.supxlabel("Client metrics use the same exported definition for both shapes. Server gauges: profiling + drain (~3,640 s).\n"
              "3,600 s sending windows; simulated acceptance 3.61. 2P1D C256 had warmup recovery and 210 end-of-window cancellations.",
              fontsize=9)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"decode_concurrency_comparison.{ext}", dpi=160)
plt.close(fig)
print(f"Wrote 10 comparison rows, 5 queue rows, and PNG/SVG to {OUT}")
