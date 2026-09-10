"""CPU-only accounting contracts; acceptance lengths include the bonus token."""
from collections import Counter
from dataclasses import dataclass, field
import math


def required_token_capacity(batch_size, input_len, output_len, page_size, reserve):
    if min(batch_size, input_len, output_len, page_size) <= 0 or reserve < 0:
        raise ValueError("lengths and batch/page sizes must be positive; reserve nonnegative")
    per_request = input_len + output_len + reserve
    return batch_size * ((per_request + page_size - 1) // page_size * page_size)


@dataclass
class DecodeAccounting:
    batch_size: int
    input_len: int
    output_len: int
    max_accept_len: int = 6
    emitted: list = field(init=False)
    verify_ct: int = 0
    raw_accept_tokens: int = 0
    num_correct_drafts: int = 0
    accept_histogram: Counter = field(default_factory=Counter)
    iteration_seconds: list = field(default_factory=list)

    def __post_init__(self):
        if min(self.batch_size, self.input_len, self.output_len, self.max_accept_len) <= 0:
            raise ValueError("batch and lengths must be positive")
        self.emitted = [0] * self.batch_size

    @property
    def seq_lens(self):
        return [self.input_len + count for count in self.emitted]

    @property
    def complete(self):
        return all(count == self.output_len for count in self.emitted)

    def record(self, accept_lens, seconds):
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
            "input_len": self.input_len,
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
