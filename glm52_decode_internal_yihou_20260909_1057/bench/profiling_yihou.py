"""Opt-in profiling for the scheduler-free internal decode bench.

Nothing here runs unless ``--profile`` is passed: ``profile_decode.py`` imports this
module for its argparse definitions only (stdlib at import time, torch lazily), and
constructs a :class:`ProfileSession` solely on the ``--profile`` branch.

Three complementary mechanisms, because CUDA-graph replay breaks the usual ones:

A. ``torch.profiler`` over a window of *measured* decode iterations. Under graph
   replay the Python-side ``record_function`` spans inside ``model_runner.forward``
   never execute, so per-kernel attribution may collapse into the graph launch.
B. ``DeviceTimer`` (``sglang.srt.utils.device_timer``) attached by us to the target
   and draft model runners. CUDA events bracket the replay *from outside*, so the
   per-stage split (``decode`` / ``eagle_draft`` / ``eagle_draft_extend`` / ``idle``)
   is valid in both graph modes. Only the Scheduler's metrics reporter normally
   populates ``model_runner.device_timer``; this bench has no Scheduler, so we
   attach our own.
C. ``--enable-profile-cuda-graph`` (a real ServerArgs flag, passed through) plus
   ``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE``. Kernels are *recorded, not executed*
   during capture, so those timings are a kernel identity/shape inventory and NOT
   performance data; the emitted metadata says so.

Why not ``sglang.srt.utils.profile_utils._ProfilerTorch``: its ``stop()`` ends with
``torch.distributed.barrier(cpu_group)``. With ``--profile-ranks 0`` only one rank
would reach that barrier and the run would hang. We call ``torch.profiler`` directly
and reproduce ``_ProfilerTorch``'s filename convention verbatim instead.
"""
import json
import os
from pathlib import Path
import time

# Categories emitted by the `device_timer_ctx(...)` call sites in the pinned tree.
KNOWN_DEVICE_TIMER_CATEGORIES = (
    "decode", "extend", "idle", "target_verify", "split_prefill",
    "eagle_draft", "eagle_draft_extend", "frozen_kv_draft",
)
SUPPORTED_ACTIVITIES = ("CPU", "GPU", "MEM", "CUDA_PROFILER")


def add_profile_cli_args(parser):
    """Pure-argparse; safe to call unconditionally (no torch, no SGLang import)."""
    group = parser.add_argument_group("profiling (opt-in; default off)")
    group.add_argument("--profile", action="store_true",
                       help="enable profiling; the run is then NOT a performance measurement")
    group.add_argument("--profile-start-step", type=int, default=20,
                       help="first measured decode iteration inside the torch.profiler window")
    group.add_argument("--profile-num-steps", type=int, default=5,
                       help="number of measured decode iterations in the window")
    group.add_argument("--profile-activities", default="CPU,GPU",
                       help=f"comma list from {','.join(SUPPORTED_ACTIVITIES)}")
    group.add_argument("--profile-ranks", default="0",
                       help="comma list of TP ranks that emit traces, or 'all'")
    group.add_argument("--profile-dir", default=None, help="default <result-dir>/profile")
    group.add_argument("--profile-with-stack", action="store_true",
                       help="python stacks in the trace (large and slow); default off")
    group.add_argument("--profile-record-shapes", action="store_true", default=True)
    group.add_argument("--no-profile-record-shapes", dest="profile_record_shapes", action="store_false")
    group.add_argument("--device-timer", action="store_true", default=True,
                       help="attach DeviceTimer to target and draft runners (default on with --profile)")
    group.add_argument("--no-device-timer", dest="device_timer", action="store_false")
    group.add_argument("--profile-top-n", type=int, default=10)
    group.add_argument("--profile-graph-capture-trace", action="store_true",
                       help="set SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1 (capture-time inventory)")
    return parser


def validate_profile_args(args):
    if not args.profile:
        return
    if args.profile_start_step < 0 or args.profile_num_steps <= 0:
        raise ValueError("profile-start-step must be nonnegative and profile-num-steps positive")
    if args.profile_top_n <= 0:
        raise ValueError("profile-top-n must be positive")
    unknown = set(parse_activities(args.profile_activities)) - set(SUPPORTED_ACTIVITIES)
    if unknown:
        raise ValueError(f"Unsupported --profile-activities {sorted(unknown)}; supported: {list(SUPPORTED_ACTIVITIES)}")
    parse_ranks(args.profile_ranks, args.tp_size)
    window_end = args.profile_start_step + args.profile_num_steps
    if args.max_steps and args.max_steps < window_end:
        raise ValueError(f"--max-steps {args.max_steps} ends before the profile window closes at {window_end}")


def parse_activities(text):
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def parse_ranks(text, tp_size):
    if text.strip() == "all":
        return list(range(tp_size))
    ranks = sorted({int(item) for item in text.split(",") if item.strip()})
    if not ranks or any(not 0 <= rank < tp_size for rank in ranks):
        raise ValueError(f"--profile-ranks {text!r} is outside [0, {tp_size})")
    return ranks


def profile_dir_for(args):
    return Path(args.profile_dir) if args.profile_dir else Path(args.result_dir) / "profile"


def configure_profile_env(args):
    """Set the SGLang profiling env in the *parent* before ranks are spawned.

    ``SGLANG_TORCH_PROFILER_DIR`` has to be in place before CUDA-graph capture, which
    happens inside each spawned rank; children inherit this process's environment.
    Returns the variables actually set, for the metadata file.
    """
    directory = profile_dir_for(args)
    directory.mkdir(parents=True, exist_ok=True)
    applied = {"SGLANG_TORCH_PROFILER_DIR": str(directory)}
    if args.profile_graph_capture_trace:
        applied["SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE"] = "1"
    os.environ.update(applied)
    return applied


def env_snapshot():
    """Every SGLANG_*/DEBUG_CLR_* variable actually in effect, for the metadata."""
    return {name: value for name, value in sorted(os.environ.items())
            if name.startswith(("SGLANG_", "DEBUG_CLR_", "HIP_", "AITER_", "TORCH_"))}


class _DeviceTimerReport:
    """Accumulates ``{category: {seconds, count}}`` from DeviceTimer callbacks."""

    def __init__(self):
        self.by_category = {}
        self.by_runner = {}
        self._runner_label = None

    def reporter_for(self, runner_label):
        def report(t, category=None, **_kwargs):
            key = category if category is not None else "unlabelled"
            self._accumulate(self.by_category, key, t)
            self._accumulate(self.by_runner.setdefault(runner_label, {}), key, t)
        return report

    @staticmethod
    def _accumulate(table, key, seconds):
        entry = table.setdefault(key, {"seconds": 0.0, "count": 0, "min_seconds": None, "max_seconds": None})
        entry["seconds"] += seconds
        entry["count"] += 1
        entry["min_seconds"] = seconds if entry["min_seconds"] is None else min(entry["min_seconds"], seconds)
        entry["max_seconds"] = seconds if entry["max_seconds"] is None else max(entry["max_seconds"], seconds)

    def snapshot(self):
        return {key: {"seconds": value["seconds"], "count": value["count"]}
                for key, value in self.by_category.items()}

    @staticmethod
    def delta(before, after):
        result = {}
        for key, value in after.items():
            base = before.get(key, {"seconds": 0.0, "count": 0})
            result[key] = {"seconds": value["seconds"] - base["seconds"],
                           "count": value["count"] - base["count"]}
        return result


class ProfileSession:
    """Owns the torch profiler window and the DeviceTimer attachment for one rank."""

    @staticmethod
    def create(args, rank, log):
        if not args.profile:
            return None
        return ProfileSession(args, rank, log)

    def __init__(self, args, rank, log):
        self.args = args
        self.rank = rank
        self.log = log
        self.activities = parse_activities(args.profile_activities)
        self.emitting = rank in parse_ranks(args.profile_ranks, args.tp_size)
        self.directory = profile_dir_for(args)
        self.window = (args.profile_start_step, args.profile_start_step + args.profile_num_steps)
        self.profile_id = os.environ.get("SGLANG_BENCH_PROFILE_ID") or f"{time.time():.6f}"
        self.torch_profiler = None
        self.started_at_step = None
        self.stopped_at_step = None
        self.report = _DeviceTimerReport() if args.device_timer else None
        self.window_snapshot = None
        self.timers = {}
        self._timed_runners = []
        self._sort_key_used = None
        self._notes = []

    # ---------------------------------------------------------------- device timer

    def attach_device_timer(self, runners):
        """Attach one DeviceTimer per runner (``{label: model_runner}``).

        Per-runner instances, not the Scheduler's single shared timer: ``DeviceTimer.wrap``
        asserts it is not re-entrant, and here the target's ``decode`` wrap and the draft's
        ``eagle_draft`` wrap are driven by one process with no Scheduler in between. Separate
        instances make a nested wrap impossible by construction. Categories are disjoint
        across runners anyway, so the accumulated totals are unaffected.

        Called after warmup so no ``torch.cuda.Event`` is ever recorded during graph capture.
        """
        if self.report is None:
            return {}
        from sglang.srt.utils.device_timer import DeviceTimer
        attached = {}
        for label, runner in runners.items():
            if runner is None:
                continue
            if getattr(runner, "device_timer", "missing") == "missing":
                raise RuntimeError(f"{label} runner has no device_timer attribute; pinned SGLang changed")
            if runner.device_timer is not None:
                raise RuntimeError(f"{label} runner already has a device_timer; refusing to overwrite")
            timer = DeviceTimer(reporter=self.report.reporter_for(label))
            runner.device_timer = timer
            self.timers[label] = timer
            self._timed_runners.append((label, runner))
            attached[label] = type(runner).__name__
        self.log(self.rank, f"profile: DeviceTimer attached to {sorted(attached)}")
        return attached

    def detach_device_timer(self):
        import torch
        if self.report is None:
            return
        torch.cuda.synchronize()  # every end_event has completed, so nothing is dropped
        for timer in self.timers.values():
            timer._report()
        for _label, runner in self._timed_runners:
            runner.device_timer = None
        self._timed_runners = []

    # ---------------------------------------------------------------- torch profiler

    def step_boundary(self, completed_steps):
        """Called once per measured iteration, *outside* the per-step timing bracket.

        ``completed_steps`` is ``accounting.verify_ct``, i.e. iterations already recorded.
        """
        if self.stopped_at_step is not None:
            return  # the window is closed for good; the profiler object is kept for export
        start, end = self.window
        if self.torch_profiler is None and self.started_at_step is None and completed_steps == start:
            self._start(completed_steps)
        elif self.torch_profiler is not None and completed_steps >= end:
            self._stop(completed_steps)

    def _start(self, completed_steps):
        import torch
        if not self.emitting:
            self._notes.append(f"rank {self.rank} not in --profile-ranks; torch.profiler not started")
            self.started_at_step = completed_steps
            self.stopped_at_step = completed_steps  # never reopen the window
            return
        activity_map = {"CPU": torch.profiler.ProfilerActivity.CPU,
                        "GPU": torch.profiler.ProfilerActivity.CUDA}
        selected = [activity_map[name] for name in self.activities if name in activity_map]
        if self.report is not None:
            self.window_snapshot = self.report.snapshot()
        if "MEM" in self.activities:
            from sglang.srt.environ import envs
            torch.cuda.memory._record_memory_history(max_entries=envs.SGLANG_MEM_PROFILE_MAX_ENTRIES.get())
        if "CUDA_PROFILER" in self.activities:
            torch.cuda.cudart().cudaProfilerStart()
        if selected:
            self.torch_profiler = torch.profiler.profile(
                activities=selected,
                with_stack=self.args.profile_with_stack,
                record_shapes=self.args.profile_record_shapes,
            )
            self.torch_profiler.start()
        self.started_at_step = completed_steps
        self.log(self.rank, f"profile: window opened at measured iteration {completed_steps} "
                            f"(activities={self.activities})")

    def _stop(self, completed_steps):
        import torch
        self.torch_profiler.stop()
        self.stopped_at_step = completed_steps
        if "CUDA_PROFILER" in self.activities:
            torch.cuda.cudart().cudaProfilerStop()
        if "MEM" in self.activities:
            self.directory.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(self.directory / f"memory_rank_{self.rank}_yihou.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
        if self.report is not None:
            self.window_snapshot = _DeviceTimerReport.delta(self.window_snapshot or {}, self.report.snapshot())
        self.log(self.rank, f"profile: window closed at measured iteration {completed_steps}")

    # ---------------------------------------------------------------- outputs

    def trace_filename(self, ps):
        """Reproduces ``_ProfilerTorch.stop()``'s convention (profile_utils.py)."""
        parts = [self.profile_id, f"TP-{ps.tp_rank}"]
        if ps.dp_size > 1:
            parts.append(f"DP-{ps.dp_rank}")
        if ps.pp_size > 1:
            parts.append(f"PP-{ps.pp_rank}")
        if ps.moe_ep_size > 1:
            parts.append(f"EP-{ps.moe_ep_rank}")
        return "decode-" + "-".join(parts) + ".trace.json.gz"

    def finish(self, ps, extra_meta):
        """Export trace + top-N tables + DeviceTimer report. Returns the metadata dict."""
        if self.torch_profiler is not None and self.stopped_at_step is None:
            # The loop ended before the window closed (e.g. accounting.complete).
            self._notes.append("decode loop ended before the profile window closed; stopping at loop end")
            self._stop(self.started_at_step)
        self.detach_device_timer()
        self.directory.mkdir(parents=True, exist_ok=True)
        written = []
        top_ops = None
        if self.torch_profiler is not None:
            path = self.directory / self.trace_filename(ps)
            self.torch_profiler.export_chrome_trace(str(path))
            written.append(str(path))
            top_ops = self._write_top_ops()
            written.extend(top_ops.pop("_written"))
            self.torch_profiler = None
        if self.report is not None:
            path = self.directory / f"device_timer_rank_{self.rank}_yihou.json"
            path.write_text(json.dumps(self._device_timer_payload(), indent=2, sort_keys=True) + "\n")
            written.append(str(path))
        meta = self._meta(ps, extra_meta, written, top_ops)
        (self.directory / f"profile_meta_rank_{self.rank}_yihou.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
        if self.rank == 0:
            (self.directory / "profile_meta_yihou.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
        return meta

    def _device_timer_payload(self):
        total = sum(entry["seconds"] for entry in self.report.by_category.values())
        return {
            "mechanism": "sglang.srt.utils.device_timer.DeviceTimer attached by the bench",
            "scope": "whole measured decode loop",
            "total_seconds": total,
            "by_category": {key: dict(value, share=value["seconds"] / total if total else 0.0)
                            for key, value in sorted(self.report.by_category.items())},
            "by_runner": {runner: dict(sorted(table.items())) for runner, table in sorted(self.report.by_runner.items())},
            "categories_observed": sorted(self.report.by_category),
            "categories_known_to_pinned_sglang": list(KNOWN_DEVICE_TIMER_CATEGORIES),
            "torch_profiler_window_delta": self.window_snapshot,
            "note": "CUDA events bracket the graph replay from outside, so these totals are "
                    "valid with CUDA graphs both enabled and disabled.",
        }

    def _write_top_ops(self):
        # torch splits one name into a CPU-side row and a DeviceType.CUDA row, and a
        # record_function span gets a CUDA row too, so device_type alone cannot tell a
        # real kernel from an annotation. FunctionEvent (not FunctionEventAvg) carries
        # is_user_annotation; collect those names and exclude them from the kernel table.
        annotations = {str(event.key) for event in self.torch_profiler.events()
                       if getattr(event, "is_user_annotation", False)}
        self._notes.append(f"user_annotation keys seen: {sorted(annotations)}")
        averages = list(self.torch_profiler.key_averages())
        rows = [self._row(event, annotations) for event in averages]
        kernels = [row for row in rows if row["row_kind"] == "device_kernel"]
        device_total = sum(row["self_device_time_us"] for row in rows)
        kernel_total = sum(row["self_device_time_us"] for row in kernels)
        cpu_total = sum(row["self_cpu_time_us"] for row in rows)
        limit = self.args.profile_top_n
        by_device = sorted(rows, key=lambda row: row["self_device_time_us"], reverse=True)[:limit]
        by_kernel = sorted(kernels, key=lambda row: row["self_device_time_us"], reverse=True)[:limit]
        by_cpu = sorted(rows, key=lambda row: row["cpu_time_us"], reverse=True)[:limit]
        payload = {
            "rank": self.rank,
            "profile_window": {"start_step": self.started_at_step, "stop_step": self.stopped_at_step},
            "num_distinct_ops": len(rows),
            "num_distinct_device_kernels": len(kernels),
            "self_device_time_us_total": device_total,
            "self_device_time_us_total_device_kernels_only": kernel_total,
            "self_cpu_time_us_total": cpu_total,
            "top_by_self_device_time_device_kernels_only": by_kernel,
            "top_by_self_device_time": by_device,
            "top_by_cpu_time_total": by_cpu,
            "how_to_read": "rank GPU cost with top_by_self_device_time_device_kernels_only; "
                           "top_by_self_device_time also contains record_function spans whose "
                           "device time overlaps the kernels beneath them.",
            "time_basis": "microseconds, summed over the whole profile window (all iterations in it)",
            "caveat": "under CUDA-graph replay a whole graph can appear as one launch entry; "
                      "compare against the --disable-cuda-graph run before reading per-kernel shares.",
        }
        json_path = self.directory / f"top_ops_rank_{self.rank}_yihou.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        text_path = self.directory / f"top_ops_rank_{self.rank}_yihou.txt"
        text_path.write_text(self._render_tables(limit))
        payload["_written"] = [str(json_path), str(text_path)]
        return payload

    @staticmethod
    def _row(event, annotations):
        def value(*names):
            for name in names:
                found = getattr(event, name, None)
                if found is not None:
                    return float(found)
            return 0.0
        device_type = str(getattr(event, "device_type", ""))
        key = str(event.key)
        # A `user_annotation` row (e.g. `step[TARGET_VERIFY bs=1]`) reports the device time
        # of everything inside the range: it OVERLAPS the kernels beneath it and must never
        # be summed with them. Only `device_kernel` rows are additive.
        if key in annotations:
            row_kind = "user_annotation"
        elif device_type.endswith("CUDA"):
            row_kind = "device_kernel"
        else:
            row_kind = "cpu_side"
        return {
            "key": key,
            "count": int(event.count),
            "row_kind": row_kind,
            "device_type": device_type,
            "self_device_time_us": value("self_device_time_total", "self_cuda_time_total"),
            "device_time_us": value("device_time_total", "cuda_time_total"),
            "self_cpu_time_us": value("self_cpu_time_total"),
            "cpu_time_us": value("cpu_time_total"),
            "input_shapes": getattr(event, "input_shapes", None) or None,
        }

    def _render_tables(self, limit):
        chunks = []
        for label, sort_keys in (("self CUDA (device) time", ("self_device_time_total", "self_cuda_time_total")),
                                 ("total CPU time", ("cpu_time_total",))):
            table, used = None, None
            for key in sort_keys:
                try:
                    table = self.torch_profiler.key_averages().table(sort_by=key, row_limit=limit)
                    used = key
                    break
                except (KeyError, ValueError, AssertionError, RuntimeError) as error:
                    self._notes.append(f"key_averages().table(sort_by={key!r}) failed: {error!r}")
            if table is None:
                chunks.append(f"==== top {limit} by {label}: unavailable (see notes in profile_meta) ====\n")
                continue
            self._sort_key_used = used
            chunks.append(f"==== top {limit} by {label} (sort_by={used!r}) ====\n{table}\n")
        return "\n".join(chunks)

    def _meta(self, ps, extra_meta, written, top_ops):
        meta = {
            "is_performance_measurement": False,
            "why": "profiling perturbs the decode loop; TPOT from a profiled run is not a "
                   "performance number. Use the non-profiled packup runs for performance.",
            "rank": self.rank,
            "emitting_rank": self.emitting,
            "cuda_graph_enabled": not self.args.disable_cuda_graph,
            "profile_window_requested": {"start_step": self.window[0], "num_steps": self.args.profile_num_steps},
            "profile_window_actual": {"start_step": self.started_at_step, "stop_step": self.stopped_at_step},
            "profile_args": {key: getattr(self.args, key) for key in sorted(vars(self.args))
                             if key == "profile" or key.startswith("profile_") or key == "device_timer"},
            "activities": self.activities,
            "profile_id": self.profile_id,
            "table_sort_key_used": self._sort_key_used,
            "env": env_snapshot(),
            "files": written,
            "notes": self._notes,
            "cuda_graph_capture_profile_caveat":
                "--enable-profile-cuda-graph and SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE profile the "
                "graph CAPTURE pass. Kernels are recorded there, not executed: treat that output as a "
                "kernel identity/shape inventory, never as timing.",
        }
        if top_ops is not None:
            meta["top_ops_summary"] = {
                "self_device_time_us_total": top_ops["self_device_time_us_total"],
                "self_device_time_us_total_device_kernels_only":
                    top_ops["self_device_time_us_total_device_kernels_only"],
                "top_by_self_device_time_device_kernels_only":
                    top_ops["top_by_self_device_time_device_kernels_only"],
            }
        if self.report is not None:
            meta["device_timer"] = self._device_timer_payload()
        meta.update(extra_meta)
        return meta
