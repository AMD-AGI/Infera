"""CPU-only accounting contracts; acceptance lengths include the bonus token.

vLLM port of the GLM-5.2 tool-3 `batch_state.py`. Two differences, both forced by vLLM's synthetic
rejection sampler (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py, SYNTHETIC_MODE):

- acceptance is drawn PER REQUEST (u = rand(seed_req, position) < rate_i), not one scalar per iteration
  broadcast over the batch, so requests progress at different speeds;
- a request that reached its output length stays in the batch (fixed batch C for the whole run) and its
  further tokens are paid but not counted (`terminal_extra_tokens`), the GLM terminal policy applied per
  request. The run ends when EVERY request has emitted `output_len` useful tokens.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path


def validate_worker_progress(previous_lens, accept_lens, new_lens):
    if not (len(previous_lens) == len(accept_lens) == len(new_lens)):
        raise ValueError("Worker sequence-length vectors have different batch sizes")
    expected = [length + count for length, count in zip(previous_lens, accept_lens)]
    if list(new_lens) != expected:
        raise ValueError(f"Worker sequence progress mismatch: expected {expected}, got {list(new_lens)}")


# ---------------------------------------------------------------- KV layout / fill policy

# Element sizes the physical initializer knows how to fill with finite values.
# 1-byte storage: random bytes in [0, 0x7C). Read as e4m3/e5m2 fp8, e8m0 scale, two fp4 e2m1 nibbles,
# or as the high byte of a little-endian bf16/fp16/fp32 word, every such byte is finite (an all-ones
# exponent needs a top byte >= 0x7C), so the fill is finite under every packed interpretation.
KV_INIT_MODES = ("random", "decode-bench", "zeros")


def kv_fill_rule(dtype_name, element_size, is_floating_point, circular_buffer):
    """Return the fill rule for one KV tensor, or raise on an unknown layout (never skip silently).

    `random` (default, GLM-5.2 rule): 1-byte storage -> bytes in [0, 0x7C); floating storage wider than one
    byte -> uniform[-1, 1]. Circular-buffer groups (DSv4.1 compressor state rings) are zeroed, as vLLM's
    own DecodeBenchConnector does ("may pack non-floating metadata alongside their floating-point state").
    """
    if circular_buffer:
        return "zeros"
    if element_size == 1:
        return "bytes_0_0x7c"
    if is_floating_point:
        return "uniform_-1_1"
    raise ValueError(f"Unknown KV storage dtype {dtype_name} (element size {element_size}); "
                     "refusing to leave it unfilled")


def count_graph_executions(managers, method="run_fullgraph"):
    """Attach host-side counters after graph capture; preserve results.

    `managers` maps a label to an object whose `method` replays a captured graph (vLLM's
    CudaGraphManager.run_fullgraph for the target, DFlashCudaGraphManager.run_fullgraph for the DSpark
    drafter). Availability of a captured graph is not evidence of replay; these counters are.
    """
    counts = {name: 0 for name in managers}
    for name, manager in managers.items():
        if manager is None:
            continue
        original = getattr(manager, method)

        def counted(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            counts[_name] += 1
            return result

        setattr(manager, method, counted)
    return counts


def module_source_paths(modules):
    paths = {}
    for module in modules:
        path = getattr(module, "__file__", None)
        if not path:
            raise ValueError(f"Module {module.__name__} has no concrete source path")
        paths[module.__name__] = str(Path(path).resolve())
    return paths


def accept_bounds(conditional_rates):
    """(min, max) bonus-inclusive tokens one verify step can emit under synthetic rates.

    The kernel accepts draft i iff every earlier draft was accepted and u_i < rate_i, so a rate of exactly
    1.0 always accepts and 0.0 never does.
    """
    rates = list(conditional_rates)
    if not rates or any(not 0.0 <= r <= 1.0 for r in rates):
        raise ValueError(f"conditional rates must be in [0, 1]: {rates}")
    minimum = 1
    for rate in rates:
        if rate < 1.0:
            break
        minimum += 1
    maximum = 1
    for rate in rates:
        if rate <= 0.0:
            break
        maximum += 1
    return minimum, maximum


def unconditional_to_conditional(rates):
    """Mirror of vLLM's unconditional_to_conditional_rates for the bound computation (CPU)."""
    out, previous = [], 1.0
    for rate in rates:
        out.append(0.0 if previous == 0.0 else min(1.0, rate / previous))
        previous = rate
    return out


def length_to_rates(length, num_spec):
    """Mirror of SpeculativeConfig._acceptance_length_to_rates (minimum-variance schedule)."""
    if not 1.0 <= length <= num_spec + 1:
        raise ValueError(f"accept length must be in [1, {num_spec + 1}], got {length}")
    drafts = length - 1
    full = int(drafts)
    return ([1.0] * full + [drafts - full] + [0.0] * (num_spec - full - 1))[:num_spec]


def required_tokens_per_request(input_len, output_len, min_accept, max_accept, num_spec):
    """Upper bound of KV positions one request touches (reserve once, never allocate in the loop).

    The loop runs until the SLOWEST request has emitted output_len tokens: at most ceil(OSL / min_accept)
    verify steps, during which any request emits at most max_accept tokens per step. Each step writes the
    verify window (1 + K) past the committed context and the drafter its own (1 + K) query rows.
    """
    if min(input_len, output_len, min_accept, max_accept) <= 0 or num_spec < 0 or min_accept > max_accept:
        raise ValueError("lengths and accept bounds must be positive, min <= max")
    steps = -(-output_len // min_accept)
    return input_len + max_accept * steps + 2 * (1 + num_spec)


def blocks_for_request(kind, block_size, computed, total_tokens, *, window=None, num_spec_blocks=0):
    """(leading null blocks, live blocks) one request needs in one KV cache group.

    Mirrors vllm/benchmarks/forward/batch.py::_num_blocks for the manager kinds a DSv4.1 config uses:
    full attention, sliding window (blocks wholly before the window at admission are the null block),
    circular buffer (one ring block per request) and mamba-style state (1 + speculative blocks).
    """
    if block_size <= 0 or total_tokens <= 0 or computed < 0:
        raise ValueError("block size and token counts must be positive")
    if kind == "circular":
        return 0, 1
    if kind == "mamba":
        return 0, 1 + num_spec_blocks
    total = -(-total_tokens // block_size)
    skipped = 0
    if kind == "sliding":
        if not window:
            raise ValueError("sliding-window group needs its window")
        skipped = max(0, computed - window + 1) // block_size
    elif kind != "full":
        raise ValueError(f"Unknown KV manager kind {kind!r}; refusing to size it")
    return skipped, total - skipped


@dataclass
class DecodeAccounting:
    batch_size: int
    input_lens: list  # one entry per request; a bare int is broadcast (uniform-ISL callers)
    output_len: int
    max_accept_len: int = 6
    emitted: list = field(init=False)
    raw: list = field(init=False)
    verify_ct: int = 0
    raw_accept_tokens: int = 0
    num_correct_drafts: int = 0
    accept_histogram: Counter = field(default_factory=Counter)
    iteration_seconds: list = field(default_factory=list)
    completed_at: list = field(init=False)

    def __post_init__(self):
        if isinstance(self.input_lens, int):
            self.input_lens = [self.input_lens] * self.batch_size
        self.input_lens = list(self.input_lens)
        if min(self.batch_size, self.output_len, self.max_accept_len) <= 0:
            raise ValueError("batch and lengths must be positive")
        if len(self.input_lens) != self.batch_size:
            raise ValueError(f"input_lens has {len(self.input_lens)} entries for batch_size {self.batch_size}")
        if any(not isinstance(length, int) or length <= 0 for length in self.input_lens):
            raise ValueError(f"every input length must be a positive int: {self.input_lens}")
        self.emitted = [0] * self.batch_size
        self.raw = [0] * self.batch_size
        self.completed_at = [None] * self.batch_size

    @property
    def input_len_uniform(self):
        return len(set(self.input_lens)) == 1

    @property
    def seq_lens(self):
        """Committed context per request (what the worker holds): ISL + every accepted token."""
        return [base + count for base, count in zip(self.input_lens, self.raw)]

    @property
    def complete(self):
        return all(count == self.output_len for count in self.emitted)

    def record(self, accept_lens, seconds):
        if self.complete:
            raise ValueError("Cannot record another iteration after output completion")
        if len(accept_lens) != self.batch_size or any(
            not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= self.max_accept_len
            for count in accept_lens
        ):
            raise ValueError("accept_lens must have one valid bonus-inclusive integer per request")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("iteration time must be finite and nonnegative")
        useful = [min(count, self.output_len - done) for count, done in zip(accept_lens, self.emitted)]
        self.emitted = [done + count for done, count in zip(self.emitted, useful)]
        self.raw = [done + count for done, count in zip(self.raw, accept_lens)]
        self.verify_ct += 1
        for index, done in enumerate(self.emitted):
            if done == self.output_len and self.completed_at[index] is None:
                self.completed_at[index] = self.verify_ct
        self.raw_accept_tokens += sum(accept_lens)
        self.num_correct_drafts += sum(count - 1 for count in accept_lens)
        self.accept_histogram.update(accept_lens)
        self.iteration_seconds.append(seconds)
        return {"iteration": self.verify_ct, "accept_lens": list(accept_lens),
                "useful_accept_lens": useful, "seq_lens": self.seq_lens,
                "seconds": seconds, "complete": self.complete}

    def summary(self, elapsed_seconds):
        useful = sum(self.emitted)
        done = [step for step in self.completed_at if step is not None]
        return {
            "complete": self.complete,
            "batch_size": self.batch_size,
            # None, never a mean or a max: downstream reads this field as "the ISL of the run".
            "input_len": self.input_lens[0] if self.input_len_uniform else None,
            "input_lens": list(self.input_lens),
            "input_len_uniform": self.input_len_uniform,
            "output_len": self.output_len,
            "emitted_per_request": self.emitted[:],
            "final_seq_lens": self.seq_lens,
            "verify_iterations": self.verify_ct,
            "useful_output_tokens": useful,
            "raw_accept_tokens": self.raw_accept_tokens,
            "num_correct_drafts": self.num_correct_drafts,
            "terminal_extra_tokens": self.raw_accept_tokens - useful,
            "realized_accept_length": self.raw_accept_tokens / (self.batch_size * self.verify_ct) if self.verify_ct else 0.0,
            "accept_histogram": dict(sorted(self.accept_histogram.items())),
            "request_completion_iteration_min": min(done) if done else None,
            "request_completion_iteration_max": max(done) if done else None,
            "elapsed_seconds": elapsed_seconds,
            "output_tokens_per_second": useful / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "effective_token_latency_ms_per_user": elapsed_seconds * 1000 * self.batch_size / useful if useful else 0.0,
        }
