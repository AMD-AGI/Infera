#!/usr/bin/env python3
"""CPU check of the PP prefill scheduling overlay (scripts/make_ppsched_overlay.py). Run in the vLLM image with the
four overlay files mounted over the image's own, so `import vllm` picks them up:
  test_ppsched.py off   switches unset: v0.34.0 behaviour (lone PP prefill uncapped, no idle-step gating, no cap)
  test_ppsched.py on    VPD_PP_ADMIT=1 VPD_PP_LONE_THRESHOLD=1 VPD_PP_MICROBATCH_MIN_TOKENS=64: ports of the upstream
                        tests of #68614 (idle steps, microbatch cap incl. the prefix-cached backlog, progress,
                        free-slot admission, stop_on_add) and #68560 (lone PP prefill chunked)
PP is emulated at the scheduler level, as upstream's tests do (pipeline_parallel_size > 1 needs that many GPUs).
Without GPUs set VLLM_TARGET_DEVICE=cpu.
"""
import os
import sys

MODE = sys.argv[1]
SWITCHES = {"VPD_PP_ADMIT": "1", "VPD_PP_LONE_THRESHOLD": "1", "VPD_PP_MICROBATCH_MIN_TOKENS": "64"}
for k, v in SWITCHES.items():
    if MODE == "on":
        os.environ[k] = v
    else:
        os.environ.pop(k, None)

import json  # noqa: E402
import queue  # noqa: E402
import tempfile  # noqa: E402
from collections import deque  # noqa: E402
from concurrent.futures import Future  # noqa: E402
from functools import partial  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import torch  # noqa: E402

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched import scheduler as sched_mod  # noqa: E402
from vllm.v1.core.sched.async_scheduler import AsyncScheduler  # noqa: E402
from vllm.v1.core.sched.scheduler import PPMicrobatchCap  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs  # noqa: E402
from vllm.v1.engine import EngineCoreRequestType  # noqa: E402
from vllm.v1.engine import core as engine_core  # noqa: E402
from vllm.v1.engine.core import EngineCore, EngineCoreProc  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

if os.environ.get("VLLM_TARGET_DEVICE") == "cpu":
    # No GPU in the test container (the node's GPUs belong to a running job); the CPU platform's kernel config check
    # wants triton-cpu, which the scheduler does not use.
    from vllm.platforms import current_platform

    type(current_platform).check_and_update_config = classmethod(lambda cls, vllm_config: None)

ADD, ABORT, WAKEUP = EngineCoreRequestType.ADD, EngineCoreRequestType.ABORT, EngineCoreRequestType.WAKEUP
MODEL_DIR = tempfile.mkdtemp(prefix="vpd-tiny-opt-")
json.dump({"architectures": ["OPTForCausalLM"], "model_type": "opt", "hidden_size": 64, "word_embed_proj_dim": 64,
           "num_attention_heads": 2, "num_hidden_layers": 2, "ffn_dim": 128, "vocab_size": 1000,
           "max_position_embeddings": 4096, "do_layer_norm_before": True, "torch_dtype": "float16"},
          open(os.path.join(MODEL_DIR, "config.json"), "w"))
init_none_hash(sha256)
HASHER = get_request_block_hasher(16, sha256)


def create_scheduler(max_num_batched_tokens=512, long_prefill_token_threshold=0, num_blocks=1000,
                     enable_prefix_caching=False):
    """tests/v1/core/utils.py create_scheduler (v0.34.0), async scheduling, one full-attention group, block 16."""
    model_config = ModelConfig(model=MODEL_DIR, trust_remote_code=True, dtype="float16", seed=42,
                               skip_tokenizer_init=True, max_model_len=4096)
    scheduler_config = SchedulerConfig(
        max_num_seqs=16, max_num_batched_tokens=max_num_batched_tokens, max_model_len=4096,
        long_prefill_token_threshold=long_prefill_token_threshold, enable_chunked_prefill=True,
        async_scheduling=True, is_encoder_decoder=model_config.is_encoder_decoder, watermark=0.0)
    cache_config = CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=enable_prefix_caching)
    vllm_config = VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                             parallel_config=ParallelConfig())
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32)
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                                    kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return AsyncScheduler(vllm_config=vllm_config, kv_cache_config=kv_cache_config, block_size=16, log_stats=True,
                          structured_output_manager=StructuredOutputManager(vllm_config))


def pp_scheduler(pp_size=4, cap=True, **kwargs):
    """_create_async_pp_scheduler of the upstream tests: PP emulated at the scheduler level."""
    s = create_scheduler(**kwargs)
    s.decode_stagger = pp_size
    s.use_pp = pp_size > 1
    s.skip_idle_steps = sched_mod._VPD_PP_ADMIT and s.use_pp
    s.pp_microbatch_cap = PPMicrobatchCap.create(s, pp_size) if cap else None
    return s


def create_requests(num_requests, num_tokens, max_tokens=16, req_ids=None, same_prompt=False):
    sp = SamplingParams(ignore_eos=True, max_tokens=max_tokens)
    sp.update_from_generation_config({}, 50256)
    req_ids = req_ids or [str(i) for i in range(num_requests)]
    return [Request(request_id=req_ids[i], prompt_token_ids=([0] if same_prompt else [i]) * num_tokens,
                    sampling_params=sp, pooling_params=None, mm_features=None, block_hasher=HASHER)
            for i in range(num_requests)]


def runner_output(so):
    req_ids = list(so.num_scheduled_tokens.keys())
    return ModelRunnerOutput(req_ids=req_ids, sampled_token_ids=[[i] for i in range(len(req_ids))], logprobs=None,
                             prompt_logprobs_dict={}, pooler_output=[])


class PollableFuture(Future):
    """tests/v1/engine/test_inputs_during_step.py: done on the `ready_after`-th poll; on_poll is called first."""

    def __init__(self, ready_after, on_poll=None):
        super().__init__()
        self.ready_after, self.on_poll, self.polls, self.waits = ready_after, on_poll, 0, 0

    def poll(self):
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if self.polls >= self.ready_after and not self.done():
            self.set_result("output")
        return self.done()

    def wait_briefly(self, pending, deadline):
        self.waits += 1

    def result(self, timeout=None):
        if not self.done():
            self.set_result("output")
        return super().result(timeout)


def make_engine():
    engine = object.__new__(EngineCoreProc)
    engine.input_queue = queue.Queue()
    engine._deferred_inputs = deque()
    engine.has_work = lambda: True
    handled = []
    engine._handle_client_request = lambda t, r: handled.append((t, r))
    return engine, handled


def check_wait_for_output_unchanged():
    """v0.34.0's test_adds_and_aborts_are_handled_while_waiting, through the refactored _wait_for_output."""
    engine, handled = make_engine()

    def on_poll(n):
        if n == 2:
            engine.input_queue.put((ADD, "a"))
        elif n == 4:
            engine.input_queue.put((ABORT, ["a"]))
            engine.input_queue.put((WAKEUP, None))

    future = PollableFuture(ready_after=10, on_poll=on_poll)
    assert engine._wait_for_output(future) == "output"
    assert future.polls == 10 and handled == [(ADD, "a"), (ABORT, ["a"]), (WAKEUP, None)], (future.polls, handled)


def check_off():
    assert not (sched_mod._VPD_PP_ADMIT or sched_mod._VPD_PP_LONE_THRESHOLD or sched_mod._VPD_PP_MB_MIN)
    assert not engine_core._VPD_PP_ADMIT
    s = pp_scheduler(pp_size=4)
    assert not s.skip_idle_steps and s.pp_microbatch_cap is None
    # lone PP prefill: v0.34.0 ignores the threshold when alone
    s = pp_scheduler(pp_size=2, max_num_batched_tokens=1024, long_prefill_token_threshold=400)
    s.add_request(create_requests(1, 1000)[0])
    assert s.schedule().total_num_scheduled_tokens == 1000
    # with two requests the threshold applies, as before
    s = pp_scheduler(pp_size=2, max_num_batched_tokens=1024, long_prefill_token_threshold=400)
    for r in create_requests(2, 1000):
        s.add_request(r)
    assert s.schedule().num_scheduled_tokens == {"0": 400, "1": 400}
    # no idle-step gating: should_schedule() == has_requests()
    s = pp_scheduler(pp_size=4)
    s.add_request(create_requests(1, 8, max_tokens=1)[0])
    s.schedule()
    assert s.schedule().total_num_scheduled_tokens == 0
    assert s.should_schedule() and s._idle_step is None
    assert not EngineCore._wait_for_new_request(SimpleNamespace(_pp_admission=False, batch_queue_size=4), deque([1]))
    check_wait_for_output_unchanged()
    print("off: lone prefill uncapped (1000), two requests capped (400 + 400), no idle gating, no cap, "
          "_wait_for_output unchanged")


def check_idle_steps(max_tokens):
    """#68614 test_should_schedule_skips_idle_steps."""
    s = pp_scheduler(pp_size=4)
    (request,) = create_requests(1, 8, max_tokens=max_tokens)
    s.add_request(request)
    prefill = s.schedule()
    assert prefill.num_scheduled_tokens[request.request_id] == 8
    assert s.should_schedule()
    idle = s.schedule()
    assert idle.total_num_scheduled_tokens == 0
    assert s.should_schedule() == (max_tokens > 1), (max_tokens, s._idle_step)
    if max_tokens == 1:
        (other,) = create_requests(1, 8, max_tokens=1, req_ids=["other"])
        s.add_request(other)
        assert s.should_schedule()
        assert s.schedule().num_scheduled_tokens == {"other": 8}
        assert not s.schedule().total_num_scheduled_tokens
        assert not s.should_schedule()
        s.update_from_output(prefill, runner_output(prefill))
        assert s.should_schedule()


def check_cap(arrivals, expected):
    """#68614 test_pp_microbatch_cap (budget 512, pp 4, floor 64)."""
    s = pp_scheduler(pp_size=4, num_blocks=1000)
    assert s.pp_microbatch_cap.min_tokens == 64
    steps = []
    for i, lens in enumerate(arrivals):
        for j, n in enumerate(lens):
            s.add_request(create_requests(1, n, max_tokens=1, req_ids=[f"{i}-{j}"])[0])
        steps.append(s.schedule().total_num_scheduled_tokens)
    while len(steps) < len(expected):
        steps.append(s.schedule().total_num_scheduled_tokens)
    assert steps == expected, (arrivals, steps, expected)


def check_cap_prefix_cached():
    """The backlog counts a waiting request's uncached tokens (kv_cache_manager.extend_cached_prefix overlay)."""
    s = pp_scheduler(pp_size=4, num_blocks=1000, enable_prefix_caching=True)
    a, b = create_requests(2, 400, max_tokens=1, req_ids=["a", "b"], same_prompt=True)
    cap, s.pp_microbatch_cap = s.pp_microbatch_cap, None
    s.add_request(a)
    assert s.schedule().num_scheduled_tokens == {"a": 400}
    s.pp_microbatch_cap = cap
    s.add_request(b)
    assert s.kv_cache_manager.get_num_cached_tokens(b) == 384
    assert s.kv_cache_manager.extend_cached_prefix(b, 0) == 384
    # 400 - 384 = 16 uncached -> floor 64 (400 uncached would give ceil(400 / 4) = 100)
    assert s.pp_microbatch_cap.apply(512) == 64, s.pp_microbatch_cap.apply(512)


def check_cap_progress():
    """#68614 test_pp_microbatch_cap_progress: no zero-token steps with floor 1 and an MTP lookahead of 4."""
    s = pp_scheduler(pp_size=4, num_blocks=1000)
    s.pp_microbatch_cap.min_tokens = 1
    s.num_prefill_lookahead = 4
    s.add_request(create_requests(1, 400, max_tokens=1)[0])
    steps = []
    while sum(steps) < 400 and 0 not in steps:
        steps.append(s.schedule().total_num_scheduled_tokens)
    assert sum(steps) == 400 and 0 not in steps, steps
    return steps


def check_lone_threshold():
    """#68560 test_long_prefill_threshold_applies_to_lone_request_with_pp (cap off, so only the threshold acts)."""
    s = pp_scheduler(pp_size=2, cap=False, max_num_batched_tokens=1024, long_prefill_token_threshold=400)
    s.add_request(create_requests(1, 1000)[0])
    chunks = [s.schedule().total_num_scheduled_tokens for _ in range(3)]
    assert chunks == [400, 400, 200], chunks


def check_free_slot_admission(pp_admission, num_tokens, should_schedule, waits):
    """#68614 test_free_slot_admission on v0.34.0's step_with_batch_queue."""
    core = MagicMock()
    core.batch_queue_size = 5
    core.batch_queue = deque(maxlen=core.batch_queue_size)
    core._pp_admission = pp_admission
    core.is_mm_encoder_only = False
    core.is_pooling_model = False
    core._should_throttle_prefills.return_value = False
    core._wait_for_new_request = partial(EngineCore._wait_for_new_request, core)
    core.scheduler.should_schedule.side_effect = [True, should_schedule]
    core.scheduler.schedule.return_value = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens, pending_structured_output_tokens=False,
        has_structured_output_spec_decode=False)
    core.model_executor.execute_model.return_value = Future()
    core.model_executor.sample_tokens.return_value = Future()
    core._handle_inputs_while_waiting.return_value = True
    core._wait_for_output.side_effect = RuntimeError("blocked")
    if not (waits or should_schedule or num_tokens):
        try:
            EngineCore.step_with_batch_queue(core)
        except RuntimeError as e:
            assert "blocked" in str(e)
        else:
            raise AssertionError("expected to block on the oldest batch")
        return
    assert EngineCore.step_with_batch_queue(core) == (None, num_tokens > 0)
    assert len(core.batch_queue) == 1
    if waits:
        core._handle_inputs_while_waiting.assert_called_once_with(core.batch_queue[-1][0], stop_on_add=True)
    else:
        core._handle_inputs_while_waiting.assert_not_called()


def check_stop_on_add():
    """#68614 test_stop_on_add_returns_at_first_add."""
    engine, handled = make_engine()

    def on_poll(n):
        if n == 2:
            engine.input_queue.put((ABORT, ["x"]))
        elif n == 4:
            engine.input_queue.put((ADD, "a"))
            engine.input_queue.put((ADD, "b"))

    future = PollableFuture(ready_after=1000, on_poll=on_poll)
    assert engine._handle_inputs_while_waiting(future, stop_on_add=True)
    assert not future.done() and handled == [(ABORT, ["x"]), (ADD, "a")] and engine.input_queue.qsize() == 1
    engine, handled = make_engine()

    def on_late_poll(n):
        if n == 5:
            engine.input_queue.put((ADD, "c"))

    future = PollableFuture(ready_after=1000, on_poll=on_late_poll)
    with mock.patch.object(engine_core, "_INPUT_SPIN_DURATION_S", 0.0):
        assert engine._handle_inputs_while_waiting(future, stop_on_add=True)
    assert future.waits == 0 and handled == [(ADD, "c")], (future.waits, handled)
    engine, handled = make_engine()
    assert not engine._handle_inputs_while_waiting(PollableFuture(ready_after=3), stop_on_add=True)


def check_on():
    assert sched_mod._VPD_PP_ADMIT and sched_mod._VPD_PP_LONE_THRESHOLD and sched_mod._VPD_PP_MB_MIN == 64
    assert engine_core._VPD_PP_ADMIT
    for max_tokens in (1, 4):
        check_idle_steps(max_tokens)
    cases = [([[400]], [100, 75, 64]), ([[400, 400]], [200, 150, 113]), ([[400] * 8], [512]), ([[40]], [40]),
             ([[40], [400]], [40, 100])]
    for arrivals, expected in cases:
        check_cap(arrivals, expected)
    check_cap_prefix_cached()
    steps = check_cap_progress()
    check_lone_threshold()
    for args in [(True, 0, False, True), (True, 0, True, False), (False, 0, False, False), (True, 100, False, False)]:
        check_free_slot_admission(*args)
    check_stop_on_add()
    check_wait_for_output_unchanged()
    print(f"on: idle steps gated (max_tokens 1 / 4), cap {[e for _, e in cases]}, prefix-cached backlog 64, "
          f"progress {steps}, lone PP prefill [400, 400, 200], free-slot admission (4 cases), stop_on_add")


if __name__ == "__main__":
    {"on": check_on, "off": check_off}[MODE]()
    print(f"ppsched overlay test ({MODE}) passed")
