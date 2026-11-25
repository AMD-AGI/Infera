"""Opt-in profiling for the scheduler-free vLLM whole-step decode bench (port of GLM-5.2 tool 3).

Nothing here runs unless ``--profile`` is passed: ``profile_decode.py`` imports this module for its
argparse definitions only (stdlib at import time, torch lazily), and constructs a
:class:`ProfileSession` solely on the ``--profile`` branch.

Three complementary mechanisms, as in the GLM-5.2 tool:

A. ``torch.profiler`` over a window of *measured* decode iterations (graph ON and OFF).
B. A device timer attached by us. vLLM has no DeviceTimer (the SGLang mechanism); the bench brackets,
   with CUDA events from OUTSIDE, the target forward (``CudaGraphManager.run_fullgraph`` replay, or the
   eager ``model.forward``), target sampling + synthetic rejection (``GPUModelRunner.sample``) and the
   DSpark drafter (``speculator.propose``, which replays the drafter graph itself). Valid in both graph
   modes; categories are ``target_forward`` / ``target_sample`` / ``draft_propose``.
C. ``--profile-graph-capture-trace``: a torch.profiler trace around ``GPUModelRunner.capture_model``
   (the engine's own CUDA-graph capture at start-up). Kernels are *recorded, not executed* there, so
   that output is a kernel identity/shape inventory and NOT performance data (vLLM has no
   ``--enable-profile-cuda-graph``; this replaces SGLang's ``SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE``).

The engine's own worker profiler (``--profiler-config``) is not used: it is driven by the EngineCore
``profile`` RPC and gathers on every rank; the window here must open and close on measured iterations of
selected ranks only.
"""
import json
import os
from pathlib import Path
import time

# Categories of the bench's own stage timer (vLLM has no DeviceTimer).
KNOWN_DEVICE_TIMER_CATEGORIES = ("target_forward", "target_sample", "draft_propose")
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
                       help="comma list of ranks (DP x TP world) that emit traces, or 'all'")
    group.add_argument("--profile-dir", default=None, help="default <result-dir>/profile")
    group.add_argument("--profile-with-stack", action="store_true",
                       help="python stacks in the trace (large and slow); default off")
    group.add_argument("--profile-record-shapes", action="store_true", default=True)
    group.add_argument("--no-profile-record-shapes", dest="profile_record_shapes", action="store_false")
    group.add_argument("--device-timer", action="store_true", default=True,
                       help="attach the CUDA-event stage timer (target forward / sample / draft) (default on with --profile)")
    group.add_argument("--no-device-timer", dest="device_timer", action="store_false")
    group.add_argument("--profile-top-n", type=int, default=10)
    group.add_argument("--profile-graph-capture-trace", action="store_true",
                       help="torch.profiler trace of the engine's CUDA-graph capture (inventory, not timing)")
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
    parse_ranks(args.profile_ranks, _world(args))
    window_end = args.profile_start_step + args.profile_num_steps
    if args.max_steps and args.max_steps < window_end:
        raise ValueError(f"--max-steps {args.max_steps} ends before the profile window closes at {window_end}")


def parse_activities(text):
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def _world(args):
    """Every DP x TP process is a rank here (GLM: TP ranks)."""
    return getattr(args, "tp_size", 1) * getattr(args, "dp_size", 1)


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

    The profile directory has to exist before the ranks are spawned (capture tracing writes into it
    from inside each rank during engine construction). Returns the variables set, for the metadata.
    """
    directory = profile_dir_for(args)
    directory.mkdir(parents=True, exist_ok=True)
    applied = {"BENCH_PROFILE_DIR": str(directory)}
    if args.profile_graph_capture_trace:
        applied["BENCH_PROFILE_GRAPH_CAPTURE_TRACE"] = "1"
    os.environ.update(applied)
    return applied


def env_snapshot():
    """Every VLLM_*/DEBUG_CLR_*/HIP/AITER/TORCH variable actually in effect, for the metadata."""
    return {name: value for name, value in sorted(os.environ.items())
            if name.startswith(("VLLM_", "DEBUG_CLR_", "HIP_", "AITER_", "TORCH_", "BENCH_"))}


class _DeviceTimerReport:
    """Accumulates ``{category: {seconds, count}}`` from stage-timer callbacks."""

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
    """Owns the torch profiler window and the stage-timer attachment for one rank."""

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
        self.emitting = rank in parse_ranks(args.profile_ranks, _world(args))
        self.directory = profile_dir_for(args)
        self.window = (args.profile_start_step, args.profile_start_step + args.profile_num_steps)
        self.profile_id = os.environ.get("BENCH_PROFILE_ID") or f"{time.time():.6f}"
        self.torch_profiler = None
        self.started_at_step = None
        self.stopped_at_step = None
        self.report = _DeviceTimerReport() if args.device_timer else None
        self.window_snapshot = None
        self.timers = {}
        self._timed_runners = []
        self._pending = []
        self._sort_key_used = None
        self._notes = []

    # ---------------------------------------------------------------- device timer

    def attach_device_timer(self, runners):
        """Attach the CUDA-event stage timer to ``{label: (owner, method_name)}`` call sites.

        Each wrapped call records a start and an end event on the current stream around the call; the
        events are resolved lazily (``_drain``) so the hot loop never blocks on them. Called after warmup,
        i.e. after every graph is captured, so no event is ever recorded inside a capture.
        """
        if self.report is None:
            return {}
        import torch
        attached = {}
        for label, (owner, method) in runners.items():
            if owner is None:
                continue
            original = getattr(owner, method)
            reporter = self.report.reporter_for(label)
            pending = self._pending

            def timed(*args, _original=original, _reporter=reporter, _label=label, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = _original(*args, **kwargs)
                end.record()
                pending.append((start, end, _reporter, _label))
                return result

            setattr(owner, method, timed)
            self._timed_runners.append((owner, method, original))
            attached[label] = f"{type(owner).__name__}.{method}"
        self.log(self.rank, f"profile: stage timer attached to {attached}")
        return attached

    def drain(self):
        """Resolve recorded event pairs (call after a synchronize)."""
        while self._pending:
            start, end, reporter, label = self._pending.pop(0)
            reporter(start.elapsed_time(end) / 1000.0, category=label)

    def detach_device_timer(self):
        import torch
        if self.report is None:
            return
        torch.cuda.synchronize()
        self.drain()
        for owner, method, original in self._timed_runners:
            setattr(owner, method, original)
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
            torch.cuda.synchronize()
            self.drain()
            self.window_snapshot = self.report.snapshot()
        if "MEM" in self.activities:
            torch.cuda.memory._record_memory_history(max_entries=int(os.environ.get("BENCH_MEM_PROFILE_MAX_ENTRIES", "1000000")))
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
            torch.cuda.synchronize()
            self.drain()
            self.window_snapshot = _DeviceTimerReport.delta(self.window_snapshot or {}, self.report.snapshot())
        self.log(self.rank, f"profile: window closed at measured iteration {completed_steps}")

    # ---------------------------------------------------------------- outputs

    def trace_filename(self, ps):
        """``decode-<id>-DP-<d>-TP-<t>[-EP-<e>].trace.json.gz`` (GLM convention, vLLM ranks)."""
        parts = [self.profile_id, f"DP-{ps['dp_rank']}", f"TP-{ps['tp_rank']}"]
        if ps.get("moe_ep_size", 1) > 1:
            parts.append(f"EP-{ps['rank']}")
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
            "mechanism": "CUDA-event stage timer attached by the bench (vLLM has no DeviceTimer)",
            "scope": "whole measured decode loop",
            "total_seconds": total,
            "by_category": {key: dict(value, share=value["seconds"] / total if total else 0.0)
                            for key, value in sorted(self.report.by_category.items())},
            "by_stage": {runner: dict(sorted(table.items())) for runner, table in sorted(self.report.by_runner.items())},
            "categories_observed": sorted(self.report.by_category),
            "categories_known_to_bench": list(KNOWN_DEVICE_TIMER_CATEGORIES),
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
                "--profile-graph-capture-trace profiles the engine's "
                "graph CAPTURE pass (GPUModelRunner.capture_model). Kernels are recorded there, not executed: treat that output as a "
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


def install_capture_trace(rank, log):
    """Wrap ``GPUModelRunner.capture_model`` in a torch.profiler trace (mechanism C).

    Must run in the rank process BEFORE the engine is constructed (capture happens at start-up).
    """
    if os.environ.get("BENCH_PROFILE_GRAPH_CAPTURE_TRACE") != "1":
        return None
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    directory = Path(os.environ["BENCH_PROFILE_DIR"])
    original = GPUModelRunner.capture_model
    written = []

    def traced(self, *args, **kwargs):
        if kwargs.get("profile_only"):
            return original(self, *args, **kwargs)
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA],
                                          record_shapes=True)
        profiler.start()
        try:
            return original(self, *args, **kwargs)
        finally:
            profiler.stop()
            path = directory / f"capture_rank_{rank}_yihou.trace.json.gz"
            profiler.export_chrome_trace(str(path))
            table = profiler.key_averages().table(sort_by="cpu_time_total", row_limit=50)
            (directory / f"capture_rank_{rank}_yihou.txt").write_text(
                "CAPTURE-TIME INVENTORY: kernels are recorded, not executed; never read as timing.\n" + table)
            written.append(str(path))
            log(rank, f"profile: capture inventory written to {path}")

    GPUModelRunner.capture_model = traced
    return written
