#!/usr/bin/env python3
"""Run the first-pass GLM-5.2 1P1D projection scan and save raw results.

Scope:

* one fixed workload: ISL=111787, OSL=911, prefix hit=0.97369;
* TP4/TP8 pools and exact 8/12/16-GPU fleet budgets;
* EP and attention-DP are either off or equal to TP;
* concurrency 1,2,4,8,16,32,64;
* measured Mooncake KV-transfer bandwidth: 37.85 GB/s per rank;
* no aggregation or Pareto processing in this script.

Usage (run from ``bench/glm5p2_pd``):

    # Inspect all 448 projections without writing files.
    ./projection_sweep.py --dry-run

    # Run, resume, or retry failed points.
    ./projection_sweep.py --output-dir results/projection-1p1d
    ./projection_sweep.py --output-dir results/projection-1p1d --resume
    ./projection_sweep.py \
      --output-dir results/projection-1p1d --resume --retry-errors

    # Inspect the 26-point >16-GPU TP8+DPA replica scan; this does not run it.
    ./projection_sweep.py --dry-run \
      --budgets 24,32 --pool-widths 8 --modes tp_dpa \
      --replica-plan \
        '1x2:32,64,96,128,168;2x1:32,64,96,128,168;1x3:32,64,96,128,168;2x2:64,96,128,192,256,336;3x1:32,64,96,128,168'

    # Analyze the immutable raw records separately.
    ./analyze_projection_sweep.py results/projection-1p1d \
      --output-dir results/projection-1p1d/analysis/baseline

Raw JSONL records are append-only under ``RUN_DIR/raw``. Every attempt also
writes a human-readable InferaSim report under ``RUN_DIR/raw/reports``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shlex
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 2
BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "infera" / "projection" / "examples" / "exp_pretrain.yaml"

GLM52_SCAN_TP = frozenset({4, 8})
MODE_ORDER = ("tp", "tp_ep", "tp_ep_dpa", "tp_dpa")
MODE_LABELS = {
    "tp": "TP",
    "tp_ep": "TP+EP",
    "tp_ep_dpa": "TP+EP+DPA",
    "tp_dpa": "TP+DPA",
}


@dataclass(frozen=True)
class PoolStrategy:
    tp: int
    ep: int
    attention_dp: int
    mode: str

    @property
    def gpus(self) -> int:
        return max(self.tp, self.ep)


@dataclass(frozen=True)
class SweepSpec:
    total_gpus: int
    prefill: PoolStrategy
    decode: PoolStrategy
    concurrency: int
    draft_cost_factor: float
    input_len: int
    output_len: int
    prefill_replicas: int = 1
    decode_replicas: int = 1

    @property
    def prefill_pool_gpus(self) -> int:
        return self.prefill.gpus * self.prefill_replicas

    @property
    def decode_pool_gpus(self) -> int:
        return self.decode.gpus * self.decode_replicas

    @property
    def point_id(self) -> str:
        factor = f"{self.draft_cost_factor:g}".replace("-", "m").replace(".", "p")
        replica_tag = (
            ""
            if self.prefill_replicas == 1 and self.decode_replicas == 1
            else f"_np{self.prefill_replicas}_nd{self.decode_replicas}"
        )
        return (
            f"g{self.total_gpus}_"
            f"p{self.prefill.tp}-{self.prefill.mode}_"
            f"d{self.decode.tp}-{self.decode.mode}_"
            f"c{self.concurrency}{replica_tag}_df{factor}"
        )


def _csv_values(raw: str, convert: Any) -> list[Any]:
    try:
        return [convert(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated value: {raw}") from exc


def positive_int_csv(raw: str) -> tuple[int, ...]:
    values = _csv_values(raw, int)
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return tuple(dict.fromkeys(values))


def nonnegative_float_csv(raw: str) -> tuple[float, ...]:
    values = _csv_values(raw, float)
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative numbers")
    return tuple(dict.fromkeys(values))


def mode_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower().replace("+", "_") for value in raw.split(","))
    unknown = [value for value in values if value not in MODE_LABELS]
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown mode(s) {unknown}; expected a subset of {','.join(MODE_ORDER)}"
        )
    return tuple(dict.fromkeys(values))


def replica_plan(raw: str) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Parse ``Np x Nd:conc,...;...`` with whitespace ignored."""
    entries: list[tuple[int, int, tuple[int, ...]]] = []
    try:
        for item in raw.split(";"):
            pair, concurrencies = item.strip().split(":", 1)
            prefill, decode = (
                int(value.strip()) for value in pair.lower().split("x", 1)
            )
            values = positive_int_csv(concurrencies)
            if prefill <= 0 or decode <= 0:
                raise ValueError
            entries.append((prefill, decode, values))
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected 'P_replicas x D_replicas:conc,...;...', "
            "for example '1x2:32,64;2x2:64,128'"
        ) from exc
    if not entries:
        raise argparse.ArgumentTypeError("replica plan cannot be empty")
    return tuple(entries)


def pool_strategies(
    width: int,
    modes: Sequence[str] = MODE_ORDER,
) -> list[PoolStrategy]:
    shapes = {
        "tp": PoolStrategy(width, 1, 1, "tp"),
        "tp_ep": PoolStrategy(width, width, 1, "tp_ep"),
        "tp_ep_dpa": PoolStrategy(width, width, width, "tp_ep_dpa"),
        "tp_dpa": PoolStrategy(width, 1, width, "tp_dpa"),
    }
    return [shapes[mode] for mode in modes]


def enumerate_specs(
    *,
    budgets: Sequence[int],
    pool_widths: Sequence[int],
    concurrencies: Sequence[int],
    draft_cost_factors: Sequence[float],
    input_len: int,
    output_len: int,
    modes: Sequence[str] = MODE_ORDER,
    replicas: Sequence[tuple[int, int, Sequence[int]]] | None = None,
) -> list[SweepSpec]:
    specs: list[SweepSpec] = []
    replica_entries = replicas or ((1, 1, concurrencies),)
    for budget in budgets:
        for prefill_width in pool_widths:
            for decode_width in pool_widths:
                for prefill_replicas, decode_replicas, replica_concurrencies in (
                    replica_entries
                ):
                    fleet_gpus = (
                        prefill_width * prefill_replicas
                        + decode_width * decode_replicas
                    )
                    if fleet_gpus != budget:
                        continue
                    for prefill in pool_strategies(prefill_width, modes):
                        for decode in pool_strategies(decode_width, modes):
                            for concurrency in replica_concurrencies:
                                for factor in draft_cost_factors:
                                    specs.append(
                                        SweepSpec(
                                            total_gpus=budget,
                                            prefill=prefill,
                                            decode=decode,
                                            concurrency=concurrency,
                                            draft_cost_factor=factor,
                                            input_len=input_len,
                                            output_len=output_len,
                                            prefill_replicas=prefill_replicas,
                                            decode_replicas=decode_replicas,
                                        )
                                    )
    return specs


def build_projection_argv(spec: SweepSpec, args: argparse.Namespace) -> list[str]:
    argv = [
        "inference",
        "--config",
        str(args.config),
        "--inference-mode",
        "performance",
        "--profiling-mode",
        "simulate",
        "--gpu-arch",
        args.gpu_arch,
        "--hbm-capacity-gb",
        str(args.hbm_capacity_gb),
        "--input-len",
        str(spec.input_len),
        "--output-len",
        str(spec.output_len),
        "--inference-batch-size",
        str(spec.concurrency),
        "--max-concurrency",
        str(spec.concurrency),
        "--weight-dtype",
        args.weight_dtype,
        "--linear-weight-dtype",
        args.linear_weight_dtype,
        "--moe-expert-dtype",
        args.moe_expert_dtype,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--kv-cache-memory-fraction",
        str(args.kv_cache_memory_fraction),
        "--prefix-cache-hit-rate",
        str(args.prefix_cache_hit_rate),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--sparse-attention-topk",
        str(args.sparse_attention_topk),
        "--attention-backend",
        args.attention_backend,
        "--speculative-num-tokens",
        str(args.speculative_num_tokens),
        "--speculative-acceptance-rate",
        str(args.speculative_acceptance_rate),
        "--speculative-draft-cost-factor",
        str(spec.draft_cost_factor),
        "--cudagraph-mode",
        args.cudagraph_mode,
        "--disaggregate",
        "--prefill-tp",
        str(spec.prefill.tp),
        "--prefill-ep",
        str(spec.prefill.ep),
        "--prefill-attention-dp",
        str(spec.prefill.attention_dp),
        "--prefill-replicas",
        str(spec.prefill_replicas),
        "--decode-tp",
        str(spec.decode.tp),
        "--decode-ep",
        str(spec.decode.ep),
        "--decode-attention-dp",
        str(spec.decode.attention_dp),
        "--decode-replicas",
        str(spec.decode_replicas),
        "--transfer-backend",
        args.transfer_backend,
        "--kv-transfer-bw-gbps",
        str(args.kv_transfer_bw_gbps),
        f"tensor_model_parallel_size={spec.prefill.tp}",
        f"expert_model_parallel_size={spec.prefill.ep}",
        "pipeline_model_parallel_size=1",
    ]
    if args.fused_kernels:
        argv.append("--fused-kernels")
    if args.fuse_rmsnorm_allreduce:
        argv.append("--fuse-rmsnorm-allreduce")
    if args.max_context_len:
        argv.extend(["--max-context-len", str(args.max_context_len)])
    return argv


def spec_payload(spec: SweepSpec) -> dict[str, Any]:
    return {
        "total_gpus": spec.total_gpus,
        "prefill": {
            **asdict(spec.prefill),
            "mode_label": MODE_LABELS[spec.prefill.mode],
            "replicas": spec.prefill_replicas,
            "gpus_per_replica": spec.prefill.gpus,
            "gpus": spec.prefill_pool_gpus,
        },
        "decode": {
            **asdict(spec.decode),
            "mode_label": MODE_LABELS[spec.decode.mode],
            "replicas": spec.decode_replicas,
            "gpus_per_replica": spec.decode.gpus,
            "gpus": spec.decode_pool_gpus,
        },
        "concurrency": spec.concurrency,
        "draft_cost_factor": spec.draft_cost_factor,
        "workload": {
            "input_len": spec.input_len,
            "output_len": spec.output_len,
        },
    }


def schedule_record(spec: SweepSpec, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "point_id": spec.point_id,
        "spec": spec_payload(spec),
        "projection": {
            "source": "simulate",
            "calibration": "unanchored",
            "cli_argv": build_projection_argv(spec, args),
        },
    }


def _set_model_environment(spec: SweepSpec, args: argparse.Namespace) -> None:
    os.environ.update(
        {
            "INFERASIM_MODEL": args.model,
            "INFERASIM_TP": str(spec.prefill.tp),
            "INFERASIM_EP": str(spec.prefill.ep),
            "INFERASIM_PP": "1",
            "INFERASIM_SEQ_LENGTH": str(spec.input_len),
            "INFERASIM_MAX_POSITION_EMBEDDINGS": str(
                args.max_position_embeddings
            ),
        }
    )


def _per_replica_concurrency(total: int, replicas: int) -> int:
    return max(1, (total + replicas - 1) // replicas)


def project_one(
    spec: SweepSpec,
    args: argparse.Namespace,
    *,
    attempt: int,
    parser: Any,
    launcher: Any,
    memory_projector: Any,
) -> tuple[dict[str, Any], str]:
    record = schedule_record(spec, args)
    record.update(
        {
            "attempt": attempt,
            "status": "pending",
            "error": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    report = io.StringIO()
    started = time.perf_counter()
    try:
        _set_model_environment(spec, args)
        cli_args, overrides = parser.parse_known_args(
            record["projection"]["cli_argv"]
        )
        with contextlib.redirect_stdout(report), contextlib.redirect_stderr(report):
            result = launcher(cli_args, overrides)
            config = result["config"]
            disaggregation = config.disaggregation_config
            prefill_concurrency = _per_replica_concurrency(
                spec.concurrency,
                spec.prefill_replicas,
            )
            decode_concurrency = _per_replica_concurrency(
                spec.concurrency,
                spec.decode_replicas,
            )
            prefill_config = replace(
                config,
                model_parallel_config=disaggregation.prefill_parallel(
                    config.model_parallel_config
                ),
                request_config=replace(
                    config.request_config,
                    batch_size=prefill_concurrency,
                    max_concurrency=prefill_concurrency,
                ),
                disaggregation_config=replace(disaggregation, enabled=False),
            )
            decode_config = replace(
                config,
                model_parallel_config=disaggregation.decode_parallel(
                    config.model_parallel_config
                ),
                request_config=replace(
                    config.request_config,
                    batch_size=decode_concurrency,
                    max_concurrency=decode_concurrency,
                ),
                disaggregation_config=replace(disaggregation, enabled=False),
            )
            prefill_memory = memory_projector(
                prefill_config,
                hbm_capacity_gb=args.hbm_capacity_gb,
                verbose=False,
            )
            decode_memory = memory_projector(
                decode_config,
                hbm_capacity_gb=args.hbm_capacity_gb,
                verbose=False,
            )

        performance = result["performance"]
        fleet_gpus = (
            int(performance.prefill_replica_gpus) * spec.prefill_replicas
            + int(performance.decode_replica_gpus) * spec.decode_replicas
        )
        if fleet_gpus != spec.total_gpus:
            raise ValueError(
                f"projector counted {fleet_gpus} GPUs; "
                f"schedule requires {spec.total_gpus}"
            )
        record["status"] = (
            "infeasible"
            if prefill_memory.fits is False or decode_memory.fits is False
            else "ok"
        )
        record["projection"].update(
            {
                "resolved_config": _json_safe(result["config"]),
                "performance": _json_safe(performance),
                "prefill_memory": _json_safe(prefill_memory),
                "decode_memory": _json_safe(decode_memory),
            }
        )
    except Exception as exc:  # noqa: BLE001 - failed points remain raw evidence
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_seconds"] = time.perf_counter() - started
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    return record, render_human_report(record, report.getvalue())


def render_human_report(
    record: Mapping[str, Any],
    inferasim_output: str,
) -> str:
    """Combine the original CLI output with identity and per-pool memory."""
    spec = record["spec"]
    projection = record["projection"]
    prefill = spec["prefill"]
    decode = spec["decode"]
    workload = spec["workload"]
    lines = [
        "=" * 100,
        "[projection-sweep] Raw Projection Report",
        "=" * 100,
        f"Point:       {record['point_id']}",
        f"Attempt:     {record.get('attempt', 0)}",
        f"Status:      {record.get('status', 'pending')}",
        (
            f"Fleet:       {spec['total_gpus']} GPU "
            f"(Prefill {prefill['gpus']} + Decode {decode['gpus']})"
        ),
        (
            f"Prefill:     TP={prefill['tp']} EP={prefill['ep']} "
            f"attention-DP={prefill['attention_dp']} "
            f"replicas={prefill['replicas']}"
        ),
        (
            f"Decode:      TP={decode['tp']} EP={decode['ep']} "
            f"attention-DP={decode['attention_dp']} "
            f"replicas={decode['replicas']}"
        ),
        (
            f"Workload:    input={workload['input_len']} "
            f"output={workload['output_len']} concurrency={spec['concurrency']}"
        ),
        f"Started:     {record.get('started_at', '')}",
        f"Finished:    {record.get('finished_at', '')}",
        f"Elapsed:     {float(record.get('elapsed_seconds', 0.0)):.3f} s",
        "",
        "Reproduce this projection:",
        "  " + shlex.join(["inferasim", *projection["cli_argv"]]),
    ]
    if record.get("error"):
        lines.extend(["", f"Error: {record['error']}"])
    lines.extend(
        [
            "",
            "-" * 100,
            "Original InferaSim command-line output",
            "-" * 100,
            inferasim_output.rstrip() or "(no command-line output)",
            "",
            "-" * 100,
            "Per-pool memory limits",
            "-" * 100,
        ]
    )
    for label, key in (
        ("Prefill", "prefill_memory"),
        ("Decode", "decode_memory"),
    ):
        memory = projection.get(key)
        if not memory:
            lines.append(f"{label}: unavailable")
            continue
        pool = spec[label.lower()]
        replicas = int(pool["replicas"])
        per_replica_max = int(memory["max_concurrent_sequences"])
        total_gb = float(memory["total_bytes"]) / (1024.0**3)
        lines.append(
            f"{label}: fits={memory.get('fits')} | "
            f"projected={total_gb:.2f} GiB/GPU | "
            f"per-replica max concurrency={per_replica_max} | "
            f"fleet max concurrency={per_replica_max * replicas}"
        )
    lines.extend(["=" * 100, ""])
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {
        "dry_run",
        "limit_configurations",
        "output_dir",
        "progress_every",
        "resume",
        "retry_errors",
    }
    settings = {
        key: _json_safe(value)
        for key, value in vars(args).items()
        if key not in excluded
    }
    settings["config"] = str(args.config.resolve())
    return settings


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm5p2_pd_projection_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": run_settings(args),
        "input_fingerprints": {
            "projection_config_sha256": _sha256(args.config),
            "scanner_sha256": _sha256(Path(__file__)),
        },
    }


def prepare_output(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path, bool]:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "run_config.json"
    raw_dir = output_dir / "raw"
    is_new = not output_dir.exists() or not any(output_dir.iterdir())
    if not is_new:
        if not args.resume:
            raise SystemExit(f"output directory is not empty; pass --resume: {output_dir}")
        if not manifest_path.is_file():
            raise SystemExit(f"cannot resume without {manifest_path}")
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = ("schema_version", "kind", "settings", "input_fingerprints")
        if any(prior.get(key) != manifest.get(key) for key in comparable):
            raise SystemExit("resume configuration or input fingerprints changed")
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output_dir, raw_dir, is_new


def write_schedule(
    raw_dir: Path,
    specs: Sequence[SweepSpec],
    args: argparse.Namespace,
) -> None:
    (raw_dir / "schedule.jsonl").write_text(
        "".join(
            json.dumps(schedule_record(spec, args), sort_keys=True) + "\n"
            for spec in specs
        ),
        encoding="utf-8",
    )


def load_checkpoint(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    latest: dict[str, dict[str, Any]] = {}
    attempts: Counter[str] = Counter()
    if not path.exists():
        return latest, attempts
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("schema_version") != SCHEMA_VERSION:
                raise SystemExit(f"{path}:{line_number}: unsupported schema version")
            latest[record["point_id"]] = record
            attempts[record["point_id"]] += 1
    return latest, attempts


def load_projection_api() -> tuple[Any, Any, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from infera.projection.cli import build_parser
        from infera.projection.core.projection.inference_projection import (
            launch_projection_from_cli,
        )
        from infera.projection.core.projection.inference_projection.memory import (
            project_inference_memory,
        )
    except ImportError as exc:
        raise SystemExit(
            "projection dependencies are unavailable; "
            "run in the same environment as a working inferasim installation"
        ) from exc
    return build_parser(), launch_projection_from_cli, project_inference_memory


def print_dry_run(
    specs: Sequence[SweepSpec],
    limit_configurations: int,
) -> None:
    allocations = Counter(
        (
            spec.total_gpus,
            spec.prefill.gpus,
            spec.prefill_replicas,
            spec.decode.gpus,
            spec.decode_replicas,
        )
        for spec in specs
    )
    print(f"scheduled projections: {len(specs)}")
    if limit_configurations:
        print(
            f"this invocation limit: "
            f"{min(limit_configurations, len(specs))} projections"
        )
    for (
        budget,
        prefill_worker,
        prefill_replicas,
        decode_worker,
        decode_replicas,
    ), count in sorted(allocations.items()):
        print(
            f"  total={budget:2d}: "
            f"P{prefill_worker}x{prefill_replicas} + "
            f"D{decode_worker}x{decode_replicas} -> {count} projections"
        )
    print("modes:")
    for mode in MODE_ORDER:
        print(f"  {mode:<10} {MODE_LABELS[mode]}")
    print(
        f"workload: ISL={specs[0].input_len}, OSL={specs[0].output_len} "
        f"(one fixed group)"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scan = parser.add_argument_group("scan space")
    scan.add_argument("--budgets", type=positive_int_csv, default=(8, 12, 16))
    scan.add_argument("--pool-widths", type=positive_int_csv, default=(4, 8))
    scan.add_argument(
        "--modes",
        type=mode_csv,
        default=MODE_ORDER,
        help="subset of tp,tp_ep,tp_ep_dpa,tp_dpa",
    )
    scan.add_argument(
        "--concurrencies",
        type=positive_int_csv,
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    scan.add_argument(
        "--replica-plan",
        type=replica_plan,
        default=(),
        help=(
            "optional topology-specific concurrency plan, for example "
            "'1x2:32,64;2x2:64,128'; overrides --concurrencies"
        ),
    )
    scan.add_argument(
        "--draft-cost-factors",
        type=nonnegative_float_csv,
        default=(0.05,),
    )

    workload = parser.add_argument_group("fixed workload")
    workload.add_argument("--input-len", type=int, default=218922)
    workload.add_argument("--output-len", type=int, default=1077)
    workload.add_argument("--prefix-cache-hit-rate", type=float, default=0.97369)
    workload.add_argument("--max-position-embeddings", type=int, default=1048576)
    workload.add_argument(
        "--max-context-len",
        type=int,
        default=0,
        help="0 uses ISL + OSL",
    )

    projection = parser.add_argument_group("projection")
    projection.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    projection.add_argument("--model", default="glm5")
    projection.add_argument("--gpu-arch", default="mi355x")
    projection.add_argument("--hbm-capacity-gb", type=float, default=288.0)
    projection.add_argument("--weight-dtype", default="mxfp4")
    projection.add_argument("--linear-weight-dtype", default="bf16")
    projection.add_argument("--moe-expert-dtype", default="mxfp4")
    projection.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    projection.add_argument("--kv-cache-memory-fraction", type=float, default=0.85)
    projection.add_argument("--chunked-prefill-size", type=int, default=32768)
    projection.add_argument("--max-num-batched-tokens", type=int, default=16384)
    projection.add_argument("--sparse-attention-topk", type=int, default=2048)
    projection.add_argument("--attention-backend", default="aiter")
    # Drafted tokens only, so a verify step spans k+1 query positions. The
    # engine is launched with --speculative-num-steps 5 --speculative-eagle-topk 1
    # --speculative-num-draft-tokens 6 (config.sh), and SGLang's
    # num-draft-tokens counts the whole verify tree: each request contributes
    # exactly that many query rows (eagle_info.py builds qo_indptr with it as
    # the stride). 6 tree nodes = 1 root + 5 drafted, hence k=5, not 6.
    projection.add_argument("--speculative-num-tokens", type=int, default=5)
    # Solved so that sum(a^0..a^k) equals the 3.61 the engine is pinned to via
    # SGLANG_SIMULATE_ACC_LEN. Depends on k: 0.765763 was the k=6 solution.
    projection.add_argument("--speculative-acceptance-rate", type=float, default=0.79067)
    projection.add_argument("--cudagraph-mode", default="full")
    projection.add_argument("--transfer-backend", default="mooncake")
    projection.add_argument(
        "--kv-transfer-bw-gbps",
        type=float,
        default=37.85,
        help="measured effective KV-transfer bandwidth per TP rank in GB/s",
    )
    projection.add_argument(
        "--fused-kernels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    projection.add_argument(
        "--fuse-rmsnorm-allreduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_DIR
        / "results"
        / f"projection_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    execution.add_argument("--dry-run", action="store_true")
    execution.add_argument("--resume", action="store_true")
    execution.add_argument("--retry-errors", action="store_true")
    execution.add_argument(
        "--limit-configurations",
        "--max-points",
        dest="limit_configurations",
        type=int,
        default=0,
        help="run at most N not-yet-complete projections; 0 runs all",
    )
    execution.add_argument("--progress-every", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    illegal_widths = sorted(set(args.pool_widths) - GLM52_SCAN_TP)
    if illegal_widths:
        parser.error(
            f"GLM-5.2 scan pool TP must be one of {sorted(GLM52_SCAN_TP)}; "
            f"got {illegal_widths}"
        )
    if args.input_len <= 0 or args.output_len <= 0:
        parser.error("--input-len and --output-len must be positive")
    if not 0 <= args.prefix_cache_hit_rate < 1:
        parser.error("--prefix-cache-hit-rate must be in [0, 1)")
    if not 0 < args.kv_cache_memory_fraction <= 1:
        parser.error("--kv-cache-memory-fraction must be in (0, 1]")
    if args.kv_transfer_bw_gbps <= 0:
        parser.error("--kv-transfer-bw-gbps must be positive")
    if args.limit_configurations < 0:
        parser.error("--limit-configurations cannot be negative")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if not args.config.is_file():
        parser.error(f"projection config does not exist: {args.config}")


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(argv)
    validate_args(args, argument_parser)
    specs = enumerate_specs(
        budgets=args.budgets,
        pool_widths=args.pool_widths,
        concurrencies=args.concurrencies,
        draft_cost_factors=args.draft_cost_factors,
        input_len=args.input_len,
        output_len=args.output_len,
        modes=args.modes,
        replicas=args.replica_plan or None,
    )
    if not specs:
        argument_parser.error(
            "no exact budget can be formed; with TP4/8 use budgets 8,12,16"
        )
    if args.dry_run:
        print_dry_run(specs, args.limit_configurations)
        return 0

    manifest = build_manifest(args)
    output_dir, raw_dir, is_new = prepare_output(args, manifest)
    if is_new:
        write_schedule(raw_dir, specs, args)

    checkpoint_path = raw_dir / "projections.jsonl"
    completed, attempts = load_checkpoint(checkpoint_path)
    pending = [
        spec
        for spec in specs
        if spec.point_id not in completed
        or (args.retry_errors and completed[spec.point_id]["status"] == "error")
    ]
    if args.limit_configurations:
        pending = pending[: args.limit_configurations]

    print(f"run directory: {output_dir}")
    print(
        f"schedule: {len(specs)} projections; "
        f"latest raw records: {len(completed)}; this invocation: {len(pending)}"
    )
    if not pending:
        print(f"raw data: {checkpoint_path}")
        return 0

    parser, launcher, memory_projector = load_projection_api()
    reports_dir = raw_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    interrupted = False
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        try:
            for index, spec in enumerate(pending, 1):
                attempt = attempts[spec.point_id] + 1
                record, report = project_one(
                    spec,
                    args,
                    attempt=attempt,
                    parser=parser,
                    launcher=launcher,
                    memory_projector=memory_projector,
                )
                checkpoint.write(json.dumps(record, sort_keys=True) + "\n")
                checkpoint.flush()
                completed[spec.point_id] = record
                attempts[spec.point_id] = attempt
                report_name = f"{spec.point_id}.attempt-{attempt}.txt"
                (reports_dir / report_name).write_text(report, encoding="utf-8")
                if (
                    index == 1
                    or index % args.progress_every == 0
                    or index == len(pending)
                ):
                    elapsed = time.perf_counter() - started
                    rate = index / elapsed if elapsed else 0.0
                    eta = (len(pending) - index) / rate if rate else 0.0
                    print(
                        f"[{index:>5}/{len(pending)}] {spec.point_id} "
                        f"{record['status']} ({rate:.1f} point/s, ETA {eta:.0f}s)"
                    )
        except KeyboardInterrupt:
            interrupted = True
            print("\ninterrupted; rerun with --resume", file=sys.stderr)

    print(f"raw data: {checkpoint_path}")
    print(f"analyze:  {BENCH_DIR / 'analyze_projection_sweep.py'} {output_dir}")
    if interrupted:
        return 130
    error_count = sum(record["status"] == "error" for record in completed.values())
    if error_count:
        print(f"latest projection errors: {error_count}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
