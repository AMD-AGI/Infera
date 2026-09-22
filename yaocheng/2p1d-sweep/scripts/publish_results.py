#!/usr/bin/env python3
"""Publish completed points in the reference results/ layout; keep diagnostics in .tmp."""
import csv
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from summarize import metrics

FIELDS = ["Point", "Concurrency", "Model", "Hardware", "Framework", "Precision",
          "PrefillWorkers", "DecodeWorkers", "PrefillTP", "DecodeTP", "PrefillGPUs", "DecodeGPUs",
          "SpecDecoding", "DPAttention", "DurationS", "TokenThroughputPerChip", "InputPerChip",
          "OutputPerChip", "TotalTokensPerSec", "TTFT_p50_s", "TTFT_p90_s", "ITL_p50_ms",
          "ITL_p90_ms", "Interactivity_p50", "RecordsProfiled", "RecordsTotal", "RecordsErrorDropped",
          "TheoreticalCacheHit", "RailFaultsBeforeAfter"]


def publish(run, output):
    run, output = run.resolve(), output.resolve()
    run_id = run.name
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "run.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous["run_id"] != run_id:
            raise ValueError(f"{output} belongs to {previous['run_id']}; use a different RESULTS_DIR")
    elif any(output.iterdir()):
        raise ValueError(f"refusing to replace existing results without a matching run.json: {output}")
    staging = run.parent.parent / "publish" / run_id
    staging.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        temporary = staging / name
        temporary.write_text(value)
        os.replace(temporary, output / name)

    topology = []
    if (run / "topology.tsv").exists():
        with (run / "topology.tsv").open() as stream:
            topology = list(csv.DictReader(stream, delimiter="\t"))
    manifest = {"run_id": run_id, "intermediate_dir": str(run), "topology": topology,
                "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "status": (run / "status.txt").read_text().strip() if (run / "status.txt").exists() else "sweeping",
                "simulation_accept_length": os.environ.get("DECODE_SIMULATE_ACC_LEN", ""),
                "requested_duration_s": os.environ.get("DURATION", ""),
                "warmup_requests_per_lane": os.environ.get("AGENTX_WARMUP_REQUESTS_PER_LANE", ""),
                "reused_preflight_run_id": os.environ.get("REUSED_PREFLIGHT_RUN_ID", ""),
                "image": os.environ.get("IMAGE", ""), "image_id": os.environ.get("EXPECTED_IMAGE_ID", "")}
    events_file = run / "measurement_events.json"
    if events_file.exists():
        manifest["measurement_events"] = json.loads(events_file.read_text())
    write("run.json", json.dumps(manifest, indent=2) + "\n")
    rows = []
    for point in sorted(run.glob("c[0-9]*")):
        status_file = point / "status.json"
        if not status_file.is_file():
            continue
        status = json.loads(status_file.read_text())
        if status.get("state") != "complete":
            continue
        conc = int(point.name[1:])
        source = point / f"agentx_conc{conc}.json"
        m = metrics(source, conc, status["requested_duration_s"])
        record = json.loads(source.read_text())
        before, after = (json.loads((point / f"{stage}.json").read_text()) for stage in ("before", "after"))
        rails = [sum(counts["rail"] for counts in audit["faults"].values()) for audit in (before, after)]
        target = output / point.name
        if not target.exists():
            temporary = staging / point.name
            temporary.mkdir(exist_ok=True)
            shutil.copyfile(source, temporary / source.name)
            profiles = list((point / "aiperf-summary").rglob("profile_export_aiperf.csv"))
            if len(profiles) == 1:
                shutil.copyfile(profiles[0], temporary / "profile_export_aiperf.csv")
            details = [f"point={point.name} rail_faults_before={rails[0]} rail_faults_after={rails[1]} exit=0"]
            for instance in before["faults"]:
                details.append(f'{instance} before={json.dumps(before["faults"][instance], sort_keys=True)} '
                               f'after={json.dumps(after["faults"][instance], sort_keys=True)}')
            (temporary / f"rails-{point.name}.txt").write_text("\n".join(details) + "\n")
            os.replace(temporary, target)
        elif (target / source.name).read_bytes() != source.read_bytes():
            raise ValueError(f"refusing to replace different measured data: {target}")
        through = record["request_metrics"]["throughput"]
        latency = record["request_metrics"]["latency"]
        accounting = record["request_accounting"]
        rows.append(dict(zip(FIELDS, [point.name, conc, "GLM-5.2-MXFP4", record["hw"],
            record["framework"], record["precision"], 2, 1, 8, 8, 16, 8, record["spec_decoding"],
            record["dp_attention"], m["duration_s"], m["total_tok_s_gpu"],
            float(through["input"]["tokens_per_second"]) / 24, m["output_tok_s_gpu"],
            through["total"]["tokens_per_second"], m["ttft_p50_s"], m["ttft_p90_s"],
            m["itl_p50_ms"], float(latency["full_response_itl"]["p90"]) * 1000, m["intvty_p50"],
            m["profiled"], accounting["records_total"], m["errors"],
            record["request_metrics"].get("cache", {}).get("theoretical_cache_hit_rate", ""),
            f"{rails[0]}/{rails[1]}"])))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader(); writer.writerows(rows)
    write("results.csv", buffer.getvalue())
    summary = (run / "summary.md").read_text() if (run / "summary.md").exists() else "尚无完成的实测档位。\n"
    header = (f"Run: `{run_id}`  \n状态：`{manifest['status']}`  \n"
              f"P/D 节点：`{json.dumps(topology, ensure_ascii=False)}`  \n"
              f"完整运行信息：`{run}`\n\n")
    write("sweep_results.md", header + summary)
    print(f"published {len(rows)} completed points to {output}")


if __name__ == "__main__":
    try:
        publish(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise SystemExit(f"publish: {exc}")
