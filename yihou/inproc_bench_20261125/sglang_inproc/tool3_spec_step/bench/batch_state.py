"""CPU-only accounting contracts; acceptance lengths include the bonus token.

Port of the GLM-5.2 tool's batch_state.py. Unchanged: DecodeAccounting, validate_worker_progress,
count_graph_executions, module_source_paths, required_token_capacity. Adapted: the accept bound is
the verify window (DSPARK block 5 -> gamma 5 -> window 6, the same 6 as EAGLE 5/6/1), and the KV
layout check validates the DeepSeek-V4.1 pool inventory instead of the MLA nope+rope layout.
"""
from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path

# DeepSeek-V4.1 target pool classes this bench knows (sglang 6be0d75ae,
# srt/mem_cache/deepseek_v4_memory_pool.py). Anything else fails loudly.
KNOWN_TARGET_POOLS = ("DeepSeekV4TokenToKVPool",)


def validate_worker_progress(previous_lens, accept_lens, new_lens):
    if not (len(previous_lens) == len(accept_lens) == len(new_lens)):
        raise ValueError("Worker sequence-length vectors have different batch sizes")
    expected = [length + count for length, count in zip(previous_lens, accept_lens)]
    if list(new_lens) != expected:
        raise ValueError(f"Worker sequence progress mismatch: expected {expected}, got {new_lens}")


def validate_kv_layout(pool_class_name, num_kv_buffers, known=KNOWN_TARGET_POOLS):
    """Refuse an unknown target KV pool instead of silently measuring something else.

    GLM-5.2 checked the raw MLA nope+rope byte layout because it wrote random bytes into it.
    Here the prefix KV is written by the engine's own prefill kernels (see profile_decode.py), so
    what must hold is that the pool is the DeepSeek-V4 hybrid pool the bench was written for and
    that it has physical buffers.
    """
    if pool_class_name not in known:
        raise ValueError(f"Unknown target KV pool {pool_class_name!r}; this bench knows {list(known)}")
    if num_kv_buffers <= 0:
        raise ValueError(f"{pool_class_name} exposes no physical KV buffers; refusing metadata-only KV")


def count_graph_executions(runners):
    """Attach host-side counters after graph capture; preserve execute results."""
    counts = {name: 0 for name in runners}
    for name, runner in runners.items():
        if runner is None:
            continue
        execute = runner.execute

        def counted(*args, _name=name, _execute=execute, **kwargs):
            result = _execute(*args, **kwargs)
            counts[_name] += 1
            return result

        runner.execute = counted
    return counts


def module_source_paths(modules):
    paths = {}
    for module in modules:
        path = getattr(module, "__file__", None)
        if not path:
            raise ValueError(f"Module {module.__name__} has no concrete source path")
        paths[module.__name__] = str(Path(path).resolve())
    return paths


def required_token_capacity(batch_size, input_len, output_len, page_size, reserve):
    if min(batch_size, input_len, output_len, page_size) <= 0 or reserve < 0:
        raise ValueError("lengths and batch/page sizes must be positive; reserve nonnegative")
    per_request = input_len + output_len + reserve
    return batch_size * ((per_request + page_size - 1) // page_size * page_size)


@dataclass
class DecodeAccounting:
    batch_size: int
    input_lens: list  # one entry per request; a bare int is broadcast (uniform-ISL callers)
    output_len: int
    max_accept_len: int = 6
    emitted: list = field(init=False)
    verify_ct: int = 0
    raw_accept_tokens: int = 0
    num_correct_drafts: int = 0
    accept_histogram: Counter = field(default_factory=Counter)
    iteration_seconds: list = field(default_factory=list)

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

    @property
    def input_len_uniform(self):
        return len(set(self.input_lens)) == 1

    @property
    def seq_lens(self):
        return [base + count for base, count in zip(self.input_lens, self.emitted)]

    @property
    def complete(self):
        return all(count == self.output_len for count in self.emitted)

    def record(self, accept_lens, seconds):
        if self.complete:
            raise ValueError("Cannot record another iteration after output completion")
        if len(accept_lens) != self.batch_size or any(
            not isinstance(count, int) or not 1 <= count <= self.max_accept_len
            for count in accept_lens
        ):
            raise ValueError("accept_lens must have one valid bonus-inclusive integer per request")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("iteration time must be finite and nonnegative")
        useful = [min(count, self.output_len - done) for count, done in zip(accept_lens, self.emitted)]
        self.emitted = [done + count for done, count in zip(self.emitted, useful)]
        self.verify_ct += 1
        self.raw_accept_tokens += sum(accept_lens)
        self.num_correct_drafts += sum(count - 1 for count in accept_lens)
        self.accept_histogram.update(accept_lens)
        self.iteration_seconds.append(seconds)
        return {"iteration": self.verify_ct, "accept_lens": list(accept_lens),
                "useful_accept_lens": useful, "seq_lens": self.seq_lens,
                "seconds": seconds, "complete": self.complete}

    def summary(self, elapsed_seconds):
        useful = sum(self.emitted)
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
            "elapsed_seconds": elapsed_seconds,
            "output_tokens_per_second": useful / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "effective_token_latency_ms_per_user": elapsed_seconds * 1000 * self.batch_size / useful if useful else 0.0,
        }
