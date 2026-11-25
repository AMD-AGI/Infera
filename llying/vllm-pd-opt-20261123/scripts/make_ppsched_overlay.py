#!/usr/bin/env python3
"""Build the PP prefill scheduling overlay (backport of vLLM #68614 and #68560 to v0.34.0) from the image's own files.

The GLM-5.2 P4D4 prefill is PP4 x TP1, DP1, async scheduling, a batch queue of 4. Three switches, all off by default
(every switch unset = v0.34.0's behaviour):
  VPD_PP_ADMIT=1                    #68614 admission: at most one in-flight empty step that no request needs (they used
                                    to fill the batch queue, so a request arriving meanwhile waited for a whole PP pass);
                                    with a free batch slot, wait on the oldest batch but return as soon as a request is
                                    added, so it is scheduled into the slot.
  VPD_PP_MICROBATCH_MIN_TOKENS=N    #68614 microbatch cap: step budget = min(budget, max(N, lookahead,
                                    ceil(uncached prefill backlog / pp)) + running decode tokens).
  VPD_PP_LONE_THRESHOLD=1           #68560: apply long_prefill_token_threshold to a lone request under PP too, so its
                                    chunks overlap across the stages.
Differences from upstream main: switches are env vars instead of --pp-microbatch-min-tokens (default off, upstream
4096); v0.34.0's "+ 2" in place of main's `1 + _min_in_flight_output_tokens`; PPMicrobatchCap is appended to
scheduler.py instead of a new pp_utils.py (one mount less); v0.34.0 keeps the deferred-sampling step local, so the
admission wait is skipped while one is pending (upstream: `deferred_step is None`).
  make_ppsched_overlay.py SITE_PACKAGES OUT_DIR   -> OUT_DIR/{scheduler,interface,kv_cache_manager,core}.py
"""
import os
import sys

SCHED_ENV = (
    "from vllm.v1.core.sched.utils import check_stop, remove_all\n",
    "from vllm.v1.core.sched.utils import check_stop, remove_all\n"
    "import os as _vpd_os\n"
    "\n"
    "_VPD_PP_ADMIT = _vpd_os.environ.get(\"VPD_PP_ADMIT\", \"0\") == \"1\"\n"
    "_VPD_PP_LONE_THRESHOLD = _vpd_os.environ.get(\"VPD_PP_LONE_THRESHOLD\", \"0\") == \"1\"\n"
    "_VPD_PP_MB_MIN = int(_vpd_os.environ.get(\"VPD_PP_MICROBATCH_MIN_TOKENS\") or 0)\n",
)

SCHED_INIT_IDLE = (
    "        self.use_pp = self.parallel_config.pipeline_parallel_size > 1\n"
    "        self._pending_encoder_input_ids",
    "        self.use_pp = self.parallel_config.pipeline_parallel_size > 1\n"
    "        # PP, not with DP, whose busy loop runs a dummy batch on idle steps.\n"
    "        self.skip_idle_steps = (\n"
    "            _VPD_PP_ADMIT\n"
    "            and self.use_pp\n"
    "            and self.parallel_config.data_parallel_size == 1\n"
    "        )\n"
    "        # In-flight empty step that no request needs; blocks should_schedule().\n"
    "        self._idle_step: SchedulerOutput | None = None\n"
    "        if self.use_pp and (_VPD_PP_ADMIT or _VPD_PP_LONE_THRESHOLD or _VPD_PP_MB_MIN):\n"
    "            logger.info(\n"
    "                \"[vpd] PP scheduling overlay: admit=%s lone_threshold=%s microbatch_min=%d\",\n"
    "                self.skip_idle_steps, _VPD_PP_LONE_THRESHOLD, _VPD_PP_MB_MIN,\n"
    "            )\n"
    "        self._pending_encoder_input_ids",
)

SCHED_INIT_CAP = (
    "        self.need_mamba_block_aligned_split = (\n"
    "            self.has_mamba_layers and self.cache_config.mamba_cache_mode == \"align\"\n"
    "        )\n",
    "        self.need_mamba_block_aligned_split = (\n"
    "            self.has_mamba_layers and self.cache_config.mamba_cache_mode == \"align\"\n"
    "        )\n"
    "        self.pp_microbatch_cap = PPMicrobatchCap.create(\n"
    "            self, self.parallel_config.pipeline_parallel_size\n"
    "        )\n",
)

SCHED_BUDGET = (
    "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    "            # Do not schedule any requests when paused.\n"
    "            token_budget = 0\n",
    "        if self._pause_state == PauseState.PAUSED_ALL:\n"
    "            # Do not schedule any requests when paused.\n"
    "            token_budget = 0\n"
    "        if self.pp_microbatch_cap is not None and token_budget > 0:\n"
    "            token_budget = self.pp_microbatch_cap.apply(token_budget)\n",
)

SCHED_THRESHOLD = (
    "        long_prefill_token_threshold = (\n"
    "            self.scheduler_config.long_prefill_token_threshold\n"
    "            if self.scheduler_config.enable_chunked_prefill and num_eligible_reqs > 1\n"
    "            else 0\n"
    "        )\n"
    "        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:\n",
    "        long_prefill_token_threshold = (\n"
    "            self.scheduler_config.long_prefill_token_threshold\n"
    "            if self.scheduler_config.enable_chunked_prefill\n"
    "            and (num_eligible_reqs > 1 or (_VPD_PP_LONE_THRESHOLD and self.use_pp))\n"
    "            else 0\n"
    "        )\n"
    "        # A lone PP prefill is capped only to pipeline its chunks; not when its\n"
    "        # multimodal items must stay whole (the chunk could never progress).\n"
    "        lone_pp_cap = long_prefill_token_threshold > 0 and num_eligible_reqs <= 1\n"
    "        if (\n"
    "            long_prefill_token_threshold > 0\n"
    "            and self.adaptive_long_prefill_threshold\n"
    "            and num_eligible_reqs > 1\n"
    "        ):\n",
)


def cap_site(indent):
    a, b = " " * indent, " " * (indent + 4)
    return (
        f"\n{a}if 0 < long_prefill_token_threshold < num_new_tokens:\n{b}num_new_tokens = long_prefill_token_threshold\n",
        f"\n{a}if 0 < long_prefill_token_threshold < num_new_tokens and not (\n"
        f"{b}lone_pp_cap and self._keeps_mm_items_whole(request)\n"
        f"{a}):\n{b}num_new_tokens = long_prefill_token_threshold\n",
    )


SCHED_IDLE_MARK = (
    "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
    "            self._update_after_schedule(scheduler_output)\n"
    "        return scheduler_output\n",
    "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
    "            self._update_after_schedule(scheduler_output)\n"
    "        if (\n"
    "            self.skip_idle_steps\n"
    "            and not total_num_scheduled_tokens\n"
    "            and not self._has_decode_awaiting_cadence()\n"
    "        ):\n"
    "            self._idle_step = scheduler_output\n"
    "        return scheduler_output\n",
)

SCHED_IDLE_CLEAR = (
    "        model_runner_output: ModelRunnerOutput,\n"
    "    ) -> dict[int, EngineCoreOutputs]:\n"
    "        sampled_token_ids = model_runner_output.sampled_token_ids\n",
    "        model_runner_output: ModelRunnerOutput,\n"
    "    ) -> dict[int, EngineCoreOutputs]:\n"
    "        # A step with tokens can make schedule() produce work again.\n"
    "        if (\n"
    "            scheduler_output.total_num_scheduled_tokens\n"
    "            or scheduler_output is self._idle_step\n"
    "        ):\n"
    "            self._idle_step = None\n"
    "        sampled_token_ids = model_runner_output.sampled_token_ids\n",
)

SCHED_ADD = (
    "    def add_request(self, request: Request) -> None:\n"
    "        existing = self.requests.get(request.request_id)\n",
    "    def add_request(self, request: Request) -> None:\n"
    "        self._idle_step = None\n"
    "        existing = self.requests.get(request.request_id)\n",
)

SCHED_METHODS = (
    "    def has_requests(self) -> bool:\n"
    "        # Override the interface default to also keep the engine alive while a\n",
    "    def _needs_no_more_steps(self, request: Request) -> bool:\n"
    "        return (\n"
    "            request.num_output_placeholders > 0\n"
    "            # Placeholders are included in num_computed_tokens (v0.34.0's form of\n"
    "            # main's 1 + _min_in_flight_output_tokens).\n"
    "            and request.num_computed_tokens + 2 - request.num_output_placeholders\n"
    "            >= request.num_prompt_tokens + request.max_tokens\n"
    "        )\n"
    "\n"
    "    def _has_decode_awaiting_cadence(self) -> bool:\n"
    "        \"\"\"Whether a running request still needs a decode but awaits the cadence.\"\"\"\n"
    "        return any(\n"
    "            self.current_step < r.next_decode_eligible_step\n"
    "            and not self._needs_no_more_steps(r)\n"
    "            for r in self.running\n"
    "        )\n"
    "\n"
    "    def should_schedule(self) -> bool:\n"
    "        return self._idle_step is None and self.has_requests()\n"
    "\n"
    "    def _keeps_mm_items_whole(self, request: Request) -> bool:\n"
    "        \"\"\"Whether the request has multimodal items that must not be split\n"
    "        across prefill chunks (`disable_chunked_mm_input`).\"\"\"\n"
    "        return (\n"
    "            self.scheduler_config.disable_chunked_mm_input\n"
    "            and not self.is_encoder_decoder\n"
    "            and bool(request.mm_features)\n"
    "        )\n"
    "\n"
    "    def has_requests(self) -> bool:\n"
    "        # Override the interface default to also keep the engine alive while a\n",
)

PP_CAP_CLASS = '''

import itertools as _vpd_itertools
import weakref as _vpd_weakref

from vllm.utils.math_utils import cdiv as _vpd_cdiv


class PPMicrobatchCap:
    """Caps each PP step's token budget at the uncached prefill backlog /
    pp_size (at least `min_tokens`) plus running decode tokens, so a light
    prefill load is split into chunks that fill every pipeline stage
    (vLLM #68614, vllm/v1/core/sched/pp_utils.py)."""

    def __init__(self, scheduler: "Scheduler", pp_size: int, min_tokens: int):
        self.scheduler = scheduler
        self.pp_size = pp_size
        self.min_tokens = min_tokens
        # (num_tokens, prefix-cache hit, probed cached prefix) of waiting requests.
        self._cached_tokens = _vpd_weakref.WeakKeyDictionary()

    @classmethod
    def create(cls, scheduler: "Scheduler", pp_size: int) -> "PPMicrobatchCap | None":
        """None without PP, when disabled, or when prefills can't be freely
        chunked."""
        config = scheduler.scheduler_config
        if (
            pp_size == 1
            or not _VPD_PP_MB_MIN
            or not config.enable_chunked_prefill
            or config.disable_chunked_mm_input
            or scheduler.need_mamba_block_aligned_split
        ):
            return None
        return cls(scheduler, pp_size, _VPD_PP_MB_MIN)

    def apply(self, budget: int) -> int:
        scheduler = self.scheduler
        waiting_queues = (
            scheduler.kv_holding_waiting,
            scheduler.waiting,
            scheduler.async_kv_waiting,
        )
        # Parked async KV loads are counted with the waiting requests.
        prefill = sum(
            max(0, request.num_tokens - request.num_computed_tokens)
            for request in scheduler._inflight_prefills
            if request.status == RequestStatus.RUNNING
        )
        if not prefill and not any(waiting_queues):
            return budget  # No prefill to split.
        decode = sum(
            1 + len(request.spec_token_ids)
            for request in scheduler.running
            if request.num_computed_tokens >= request.num_prompt_tokens
            and not scheduler._needs_no_more_steps(request)
        )
        for request in _vpd_itertools.chain(*waiting_queues):
            if prefill >= budget * self.pp_size:
                break  # cap already at the full budget
            hit = self._num_cached_tokens(request)
            prefill += max(0, request.num_tokens - hit)
        # num_prefill_lookahead: the MTP lookahead reservation must leave a token.
        cap = max(
            self.min_tokens,
            scheduler.num_prefill_lookahead,
            _vpd_cdiv(prefill, self.pp_size),
        )
        return min(budget, cap + decode)

    def _num_cached_tokens(self, request: Request) -> int:
        """`get_num_cached_tokens`, redone only when num_tokens changes
        (preemption) or more of the request's prefix has been cached since."""
        kv_cache_manager = self.scheduler.kv_cache_manager
        cached = self._cached_tokens.get(request)
        if cached is not None and cached[0] == request.num_tokens:
            _, num_cached, prefix = cached
            if kv_cache_manager.extend_cached_prefix(request, prefix) == prefix:
                return num_cached
        num_cached = kv_cache_manager.get_num_cached_tokens(request)
        prefix = kv_cache_manager.extend_cached_prefix(request, num_cached)
        self._cached_tokens[request] = (request.num_tokens, num_cached, prefix)
        return num_cached
'''

IFACE = (
    "    def has_requests(self) -> bool:\n"
    "        \"\"\"Returns True if there are unfinished requests, or finished requests\n"
    "        not yet returned in SchedulerOutputs.\"\"\"\n"
    "        return self.has_unfinished_requests() or self.has_finished_requests()\n",
    "    def has_requests(self) -> bool:\n"
    "        \"\"\"Returns True if there are unfinished requests, or finished requests\n"
    "        not yet returned in SchedulerOutputs.\"\"\"\n"
    "        return self.has_unfinished_requests() or self.has_finished_requests()\n"
    "\n"
    "    def should_schedule(self) -> bool:\n"
    "        \"\"\"Whether the engine should schedule another step now, rather than wait\n"
    "        for an in-flight one.\"\"\"\n"
    "        return self.has_requests()\n",
)

KVM_IMPORT = (
    "from vllm.v1.core.single_type_kv_cache_manager import MambaManager\n",
    "from vllm.v1.core.single_type_kv_cache_manager import (\n"
    "    FullAttentionManager,\n"
    "    MambaManager,\n"
    ")\n",
)

KVM_INIT = (
    "        self.kv_cache_config = kv_cache_config\n\n"
    "        # Watermark: minimum number of KV cache blocks to keep free when\n",
    "        self.kv_cache_config = kv_cache_config\n"
    "        # Full-attention groups hashed per block: their cached prefix of a\n"
    "        # request grows one hash block at a time.\n"
    "        self.prefix_probe_group_ids = [\n"
    "            manager.kv_cache_group_id\n"
    "            for manager in self.coordinator.single_type_managers\n"
    "            if isinstance(manager, FullAttentionManager)\n"
    "            and manager.block_size == self.block_pool.hash_block_size\n"
    "        ]\n\n"
    "        # Watermark: minimum number of KV cache blocks to keep free when\n",
)

KVM_EXTEND = (
    "        return num_computed_tokens\n\n"
    "    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:\n",
    "        return num_computed_tokens\n\n"
    "    def extend_cached_prefix(self, request: Request, num_tokens: int) -> int:\n"
    "        \"\"\"Extends `num_tokens`, a cached prefix length of the request, by the\n"
    "        full-attention blocks cached after it (if any such groups).\"\"\"\n"
    "        if not self.prefix_probe_group_ids or not self.prefix_cache_lookup_enabled(\n"
    "            request\n"
    "        ):\n"
    "            return num_tokens\n"
    "        block_size = self.block_pool.hash_block_size\n"
    "        block_hashes = request.block_hashes\n"
    "        # The last token is always recomputed.\n"
    "        max_blocks = min(len(block_hashes), (request.num_tokens - 1) // block_size)\n"
    "        num_blocks = num_tokens // block_size\n"
    "        while num_blocks < max_blocks and self.block_pool.get_cached_block(\n"
    "            block_hashes[num_blocks], self.prefix_probe_group_ids\n"
    "        ):\n"
    "            num_blocks += 1\n"
    "        return num_blocks * block_size\n\n"
    "    def record_prefix_cache_stats(self, request: Request, num_hits: int) -> None:\n",
)

CORE_ENV = (
    "import msgspec\n",
    "import msgspec\n"
    "\n"
    "_VPD_PP_ADMIT = os.environ.get(\"VPD_PP_ADMIT\", \"0\") == \"1\"\n",
)

CORE_INIT = (
    "        if self.batch_queue_size > 1:\n"
    "            logger.debug(\"Batch queue is enabled with size %d\", self.batch_queue_size)\n"
    "            self.batch_queue = deque(maxlen=self.batch_queue_size)\n",
    "        if self.batch_queue_size > 1:\n"
    "            logger.debug(\"Batch queue is enabled with size %d\", self.batch_queue_size)\n"
    "            self.batch_queue = deque(maxlen=self.batch_queue_size)\n"
    "        # Not with DP, whose busy loop runs a dummy batch on idle steps.\n"
    "        self._pp_admission = (\n"
    "            _VPD_PP_ADMIT\n"
    "            and self.batch_queue_size > 1\n"
    "            and vllm_config.parallel_config.pipeline_parallel_size > 1\n"
    "            and vllm_config.parallel_config.data_parallel_size == 1\n"
    "        )\n",
)

CORE_SCHED1 = (
    "        deferred_scheduler_output = None\n"
    "        if self.scheduler.has_requests():\n",
    "        deferred_scheduler_output = None\n"
    "        if self.scheduler.should_schedule():\n",
)

CORE_SCHED2 = (
    "                if len(batch_queue) < self.batch_queue_size and (\n"
    "                    model_executed or self.scheduler.has_requests()\n"
    "                ):\n",
    "                if len(batch_queue) < self.batch_queue_size and (\n"
    "                    model_executed or self.scheduler.should_schedule()\n"
    "                ):\n",
)

CORE_WAIT_NEW = (
    "        # Block until the next result is available.\n"
    "        future, scheduler_output, exec_model_fut = batch_queue.pop()\n",
    "        if deferred_scheduler_output is None and self._wait_for_new_request(\n"
    "            batch_queue\n"
    "        ):\n"
    "            # Schedule the newly arrived request into the free slot.\n"
    "            return None, model_executed\n"
    "\n"
    "        # Block until the next result is available.\n"
    "        future, scheduler_output, exec_model_fut = batch_queue.pop()\n",
)

CORE_BASE = (
    "    def _wait_for_output(self, future: Future[Any]) -> Any:\n"
    "        \"\"\"Block until a step's model output is ready.\"\"\"\n"
    "        return future.result()\n",
    "    def _wait_for_new_request(self, batch_queue: deque) -> bool:\n"
    "        \"\"\"PP, with a free batch slot: wait on the oldest batch, but return True\n"
    "        as soon as a request is added, to schedule it into the slot.\"\"\"\n"
    "        return (\n"
    "            self._pp_admission\n"
    "            and len(batch_queue) < self.batch_queue_size\n"
    "            and self._handle_inputs_while_waiting(batch_queue[-1][0], stop_on_add=True)\n"
    "        )\n"
    "\n"
    "    def _wait_for_output(self, future: Future[Any]) -> Any:\n"
    "        \"\"\"Block until a step's model output is ready.\"\"\"\n"
    "        return future.result()\n"
    "\n"
    "    def _handle_inputs_while_waiting(\n"
    "        self, future: Future[Any], stop_on_add: bool = False\n"
    "    ) -> bool:\n"
    "        \"\"\"Handle inputs until the output is ready. Returns True if `stop_on_add`\n"
    "        and a request was added first.\"\"\"\n"
    "        return False\n",
)

CORE_POLL_CONST = (
    "_INPUT_SPIN_DURATION_S = 1.0\n",
    "_INPUT_SPIN_DURATION_S = 1.0\n"
    "# With a free batch slot, how often to check for new requests after spinning.\n"
    "_INPUT_POLL_INTERVAL_S = 0.001\n",
)

CORE_PROC_WAIT = (
    "    def _wait_for_output(self, future: Future[Any]) -> Any:\n"
    "        \"\"\"Wait for a step's model output, meanwhile handling new requests and\n"
    "        aborts as they arrive.\"\"\"\n"
    "        poll: Callable[[], bool] | None = getattr(future, \"poll\", None)\n"
    "        wait_briefly: Callable[[Sized, float], None] | None = getattr(\n"
    "            future, \"wait_briefly\", None\n"
    "        )\n"
    "        if poll is not None and wait_briefly is not None:\n"
    "            pending_inputs = self.input_queue.queue\n"
    "            spin_deadline = time.monotonic() + _INPUT_SPIN_DURATION_S\n"
    "            while not self._deferred_inputs and not poll():\n"
    "                if pending_inputs:\n"
    "                    req = self.input_queue.get_nowait()\n"
    "                    # Some kinds of request must wait until the step completes.\n"
    "                    if req[0] in _INPUTS_HANDLED_DURING_STEP:\n"
    "                        self._handle_client_request(*req)\n"
    "                    else:\n"
    "                        self._deferred_inputs.append(req)\n"
    "                elif time.monotonic() < spin_deadline:\n"
    "                    # Spin on something cheaper than poll(), to leave the GIL\n"
    "                    # free more of the time for the input and output threads.\n"
    "                    wait_briefly(pending_inputs, spin_deadline)\n"
    "                else:\n"
    "                    # Long-running step: stop busy-looping and block until the\n"
    "                    # output arrives. Later inputs are handled after the step.\n"
    "                    break\n"
    "\n"
    "        return future.result()\n",
    "    def _wait_for_output(self, future: Future[Any]) -> Any:\n"
    "        \"\"\"Wait for a step's model output, meanwhile handling new requests and\n"
    "        aborts as they arrive.\"\"\"\n"
    "        self._handle_inputs_while_waiting(future)\n"
    "        return future.result()\n"
    "\n"
    "    def _handle_inputs_while_waiting(\n"
    "        self, future: Future[Any], stop_on_add: bool = False\n"
    "    ) -> bool:\n"
    "        \"\"\"Handle new requests and aborts while waiting for a step's output.\"\"\"\n"
    "        poll: Callable[[], bool] | None = getattr(future, \"poll\", None)\n"
    "        wait_briefly: Callable[[Sized, float], None] | None = getattr(\n"
    "            future, \"wait_briefly\", None\n"
    "        )\n"
    "        if poll is not None and wait_briefly is not None:\n"
    "            pending_inputs = self.input_queue.queue\n"
    "            spin_deadline = time.monotonic() + _INPUT_SPIN_DURATION_S\n"
    "            while not self._deferred_inputs and not poll():\n"
    "                if pending_inputs:\n"
    "                    req = self.input_queue.get_nowait()\n"
    "                    # Some kinds of request must wait until the step completes.\n"
    "                    if req[0] in _INPUTS_HANDLED_DURING_STEP:\n"
    "                        self._handle_client_request(*req)\n"
    "                        if stop_on_add and req[0] == EngineCoreRequestType.ADD:\n"
    "                            return True\n"
    "                    else:\n"
    "                        self._deferred_inputs.append(req)\n"
    "                elif time.monotonic() < spin_deadline:\n"
    "                    # Spin on something cheaper than poll(), to leave the GIL\n"
    "                    # free more of the time for the input and output threads.\n"
    "                    wait_briefly(pending_inputs, spin_deadline)\n"
    "                elif stop_on_add:\n"
    "                    # A new request could use the free batch slot, so keep\n"
    "                    # checking, but without holding the GIL.\n"
    "                    time.sleep(_INPUT_POLL_INTERVAL_S)\n"
    "                else:\n"
    "                    # Long-running step: stop busy-looping and block until the\n"
    "                    # output arrives. Later inputs are handled after the step.\n"
    "                    break\n"
    "        return False\n",
)

FILES = {
    "scheduler.py": (
        "vllm/v1/core/sched/scheduler.py",
        [SCHED_ENV, SCHED_INIT_IDLE, SCHED_INIT_CAP, SCHED_BUDGET, SCHED_THRESHOLD, cap_site(12), cap_site(20),
         SCHED_IDLE_MARK, SCHED_IDLE_CLEAR, SCHED_ADD, SCHED_METHODS],
        PP_CAP_CLASS,
    ),
    "interface.py": ("vllm/v1/core/sched/interface.py", [IFACE], ""),
    "kv_cache_manager.py": ("vllm/v1/core/kv_cache_manager.py", [KVM_IMPORT, KVM_INIT, KVM_EXTEND], ""),
    "core.py": (
        "vllm/v1/engine/core.py",
        [CORE_ENV, CORE_INIT, CORE_SCHED1, CORE_SCHED2, CORE_WAIT_NEW, CORE_BASE, CORE_POLL_CONST, CORE_PROC_WAIT],
        "",
    ),
}


def main():
    sp, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name, (rel, edits, tail) in FILES.items():
        src = os.path.join(sp, rel)
        s = open(src).read()
        for old, new in edits:
            n = s.count(old)
            if n != 1:
                sys.exit(f"expected exactly one {old.strip()[:80]!r} in {src}, found {n}")
            s = s.replace(old, new)
        s += tail
        open(os.path.join(out, name), "w").write(s)
        print(f"wrote {out}/{name} (from {rel})")


if __name__ == "__main__":
    main()
