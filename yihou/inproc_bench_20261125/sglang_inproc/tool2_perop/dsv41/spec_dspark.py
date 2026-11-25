"""DSPARK (DeepSeek-V4.1 MTP) path of tool 2 -- the counterpart of glm52's EAGLE path.

glm52 (`model_bench._build_spec_worker`, `seed_draft_input`, `run_step`, `spec_phases`, `draft_ops`) builds
sglang's own EAGLEWorkerV2 on the target TpModelWorker and seeds a synthetic draft input over random KV.
DSv4.1 serves DSPARK (block 5), whose worker keeps draft KV injected from the target hidden states and a
DFlashDraftInputV2 per batch; a hand-made draft input is not physically valid there, so the steady state
is reached the way the server reaches it: a real prefill of the synthetic prompts through
`DSparkWorkerV2.forward_batch_generation`, then decode steps, each committed like the scheduler does.

Construction order, bootstrap prefill, the non-overlap run_batch branch and the commit are adapted from
this kit's tool 3 (`sglang_inproc/tool3_spec_step/bench/profile_decode.py`, Engine / bootstrap / forward /
decode_step, 2026-11-25), which cites sglang 6be0d75ae managers/scheduler.py for each step.
"""

import dataclasses

import torch


class Spec:
    """The objects the Scheduler owns for this rank: target TpModelWorker, DSPARK worker, pools, tree cache."""

    def __init__(self, server_args, port_args, gpu_id, tp_rank):
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import bootstrap
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.mem_cache import kv_cache_builder
        from sglang.srt.runtime_context import get_device, get_model, get_schedule, get_spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.model_config = ModelConfig.from_server_args(server_args)
        bootstrap.init_parallel_runtime(server_args=server_args, device=get_device().device, dist_port=port_args.nccl_port)
        bootstrap.init_layer_runtime(model_config=self.model_config)
        self.spec_algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
        if not self.spec_algorithm.is_dspark():
            raise SystemExit(f"tool 2's MTP path drives DSPARK (DSv4.1's MTP); got {get_spec().speculative_algorithm}")
        self.target = TpModelWorker(server_args=server_args, gpu_id=gpu_id, nccl_port=port_args.nccl_port)
        if get_model().is_startup_weight_load_overlap:
            self.target.start_startup_weight_load()
        worker_cls = self.spec_algorithm.create_worker(server_args)
        self.worker = worker_cls(server_args=server_args, gpu_id=gpu_id, nccl_port=port_args.nccl_port, target_worker=self.target)
        runner = self.target.model_runner
        runner.account_preloaded_weights(self.target.preloaded_weights_bytes + self.worker.preloaded_weights_bytes)
        self.target.alloc_memory_pool(draft_parallel_layout=self.worker.draft_parallel_layout)
        kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.target)
        pool, allocator = self.target.get_memory_pool()
        self.worker.alloc_memory_pool(memory_pool_config=runner.memory_pool_config, req_to_token_pool=pool,
                                      token_to_kv_pool_allocator=allocator)
        self.worker.init_hicache_draft_plan()
        self.target.init_attention_backends()
        self.worker.init_attention_backends()
        with torch.no_grad():
            self.target.init_cuda_graphs()
            self.worker.init_cuda_graphs()
        self.worker.prewarm_sampling()
        if runner.token_to_kv_pool.post_capture_active:
            runner.post_capture_resize_kv_pool(draft_runners=self.worker._draft_model_runners())
        kv_cache_builder.agree_on_stage_request_capacity(tp_worker=self.target)
        if get_model().is_startup_weight_load_overlap:
            self.target.finalize_startup_weight_load()
        self.page_size = get_schedule().page_size
        built = kv_cache_builder.build_kv_cache(
            server_args=server_args, model_config=self.model_config, tp_worker=self.target, page_size=self.page_size,
            spec_algorithm=self.spec_algorithm, enable_metrics=False, enable_kv_cache_events=False,
            enable_hierarchical_cache=False, hicache_draft_plan=getattr(self.worker, "hicache_draft_plan", None),
            draft_window_reaches=getattr(self.worker, "draft_window_reaches", None))
        self.tree_cache = built.tree_cache
        self.req_to_token_pool = built.req_to_token_pool
        self.allocator = built.token_to_kv_pool_allocator

    # glm52 names
    @property
    def target_worker(self):
        return self.target

    @property
    def draft_runner(self):
        return self.worker.draft_model_runner


def build(server_args, port_args, gpu_id, tp_rank):
    return Spec(server_args, port_args, gpu_id, tp_rank)


def runners(spec) -> dict:
    return {"target": spec.target.model_runner, "draft": spec.worker.draft_model_runner}


def models(spec) -> dict:
    return {k: r.model for k, r in runners(spec).items()}


def draft_model(spec):
    return spec.worker.draft_model


def draft_layers(spec):
    return list(getattr(draft_model(spec), "stages", []) or [])


def draft_ops(spec) -> dict:
    """The draft model's hookable modules, keyed `draft|<name>` (stage i ops as `draft|s<i>.<op>`)."""
    import model_bench

    model = draft_model(spec)
    ops = {}
    for i, stage in enumerate(draft_layers(spec)):
        for name, mod in model_bench.layer_ops(stage).items():
            if isinstance(mod, model_bench._FnOp):
                continue
            ops[f"draft|s{i}.{name}"] = mod
        for extra in ("main_proj", "main_norm", "norm"):
            mod = getattr(stage, extra, None)
            if isinstance(mod, torch.nn.Module):
                ops[f"draft|s{i}.{extra}"] = mod
    for name in ("embed_tokens", "markov_head", "confidence_head"):
        mod = getattr(model, name, None)
        if isinstance(mod, torch.nn.Module):
            ops[f"draft|{name}"] = mod
    return ops


def phases(spec) -> list:
    """The phases of one DSPARK step and the model forwards inside them (glm52 spec_phases)."""
    if spec is None:
        return []
    w = spec.worker
    m = models(spec)
    return [
        ("draft (propose)", getattr(w, "_proposer", None), "propose"),
        ("verify (compact)", getattr(w, "_verify_executor", None), "run_compact"),
        ("verify (non-compact)", getattr(w, "_verify_executor", None), "run_non_compact"),
        ("accept+finalize", getattr(w, "_verify_executor", None), "accept_and_finalize"),
        ("  forward:draft", m["draft"], "forward"),
        ("  forward:target", m["target"], "forward"),
    ]


def note(server_args) -> str:
    from sglang_bench import cfg

    sa = cfg(server_args)
    block = getattr(sa, "speculative_dspark_block_size", None)
    import os

    acc = os.environ.get("SGLANG_SIMULATE_ACC_LEN")
    meth = os.environ.get("SGLANG_SIMULATE_ACC_METHOD")
    return (
        f"{sa.speculative_algorithm}, block {block}, {sa.speculative_num_draft_tokens} verify tokens;"
        f" simulated acceptance {acc or 'off'}" + (f" ({meth})" if meth else "") + ". One step is the draft"
        " proposal, the target verify and the accept/commit -- the per-op rows below are the verify's share,"
        " measured on the target stack only; the draft model is one row."
    )


def forward_note(spec) -> str:
    return (
        "every model forward of one DSPARK step (draft proposal forward(s) and the target verify), each timed"
        " in place under its own graph and summed."
    )


def _sync_fn(mr):
    from sglang.srt.layers.dp_attention import require_mlp_sync, require_mlp_tp_gather
    from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.runtime_context import get_schedule

    if not require_mlp_sync():
        return lambda batch: batch

    def no_idle():
        raise RuntimeError("tool 2 never runs an idle DP rank")

    def sync(batch):
        return prepare_mlp_sync_batch_raw(
            batch, model_runner=mr, get_idle_batch=no_idle, disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(),
            disable_overlap_schedule=get_schedule().disable_overlap_schedule, offload_tags=set())

    return sync


def _forward(spec, batch):
    """Scheduler.run_batch non-overlap spec-v2 branch (via tool 3)."""
    from sglang.srt.managers.overlap_utils import resolve_forward_inputs

    if getattr(batch, "prefill_input_ids_cpu", None) is not None:
        resolve_forward_inputs(batch, None)
    # Scheduler._on_step_launch stamps these before every forward; DSPARK decode reads forward_iter.
    spec.forward_ct = getattr(spec, "forward_ct", 0) + 1
    batch.forward_iter = spec.forward_ct
    batch.launch_ts = __import__("time").monotonic()
    snap = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
    if batch.sampling_info is not None:
        batch.sampling_info = batch.sampling_info.copy_for_forward()
    try:
        result = spec.worker.forward_batch_generation(batch)
    finally:
        for name, value in snap.items():
            setattr(batch, name, value)
    batch.spec_info = result.next_draft_input
    new = result.new_seq_lens
    if new is not None and new is not batch.seq_lens:
        batch.seq_lens = new
        if batch.seq_lens_cpu is not None:
            batch.seq_lens_cpu = new.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
    batch.input_ids = None
    return result


class Stepper:
    """A running DSPARK decode batch of the bench shape: real prefill bootstrap, then one step per prepare()."""

    def __init__(self, args, mr, spec):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang_bench import local_conc, parallel_state, synthetic_reqs

        self.spec, self.mr = spec, mr
        self.sync = _sync_fn(mr)
        self.pending = None
        n = local_conc(args, mr)
        rank = parallel_state(mr).attn_dp_rank or 0
        # max_new_tokens bounds how far a request may run; the bench steps are few.
        reqs = synthetic_reqs(n, args.ISL, prefix=f"mtp-dp{rank}")
        for r in reqs:
            r.sampling_params.max_new_tokens = 1 << 20
            r.sampling_params.ignore_eos = True
        budget = max(args.ISL, int(getattr(model_bench_cfg(mr), "chunked_prefill_size", 0) or args.ISL))
        groups, cur, tok = [], [], 0
        for r in reqs:
            if cur and tok + args.ISL > budget:
                groups.append(cur)
                cur, tok = [], 0
            cur.append(r)
            tok += args.ISL
        groups.append(cur)
        batch = None
        for g in groups:
            part = ScheduleBatch.init_new(g, spec.req_to_token_pool, spec.allocator, spec.tree_cache,
                                          spec.model_config, False, spec.spec_algorithm)
            part.prepare_for_extend()
            self.sync(part)
            result = _forward(spec, part)
            torch.cuda.synchronize()
            for req, tok_id in zip(g, result.next_token_ids.cpu().tolist()):
                req.output_ids.append(int(tok_id))
            batch = part if batch is None else (batch.merge_batch(part) or batch)
        self.batch = batch
        self.n_prefill_groups = len(groups)

    def _commit(self):
        if self.pending is None:
            return
        result = self.pending
        self.pending = None
        accept = [int(v) for v in result.accept_lens.cpu().tolist()]
        tokens = result.next_token_ids.cpu().tolist()
        stride = int(result.speculative_num_draft_tokens or 6)
        for i, (req, count) in enumerate(zip(self.batch.reqs, accept)):
            req.kv.commit_decode(count)
            req.output_ids.extend(int(t) for t in tokens[i * stride: i * stride + count])
        self.last_accept = accept

    def prepare(self):
        """Commit the previous step, prepare the next one (outside any clock); return its run()."""
        self._commit()
        self.batch.prepare_for_decode()
        self.sync(self.batch)

        def run():
            self.pending = _forward(self.spec, self.batch)
            return self.pending

        return run

    def close(self):
        from sglang.srt.mem_cache.common import release_kv_cache

        torch.cuda.synchronize()
        self._commit()
        for req in self.batch.reqs:
            if req.kv.holds_kv:
                release_kv_cache(req, self.spec.tree_cache, checkpoint=False)
        self.spec.worker.clear_cache_pool()


def model_bench_cfg(mr):
    from sglang_bench import cfg

    return cfg(mr.server_args)
