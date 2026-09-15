"""CPU-only input-length specifications for heterogeneous decode batches.

Plan 1: every attention-DP rank receives the SAME ISL multiset in the SAME order. generate()
is therefore a pure function of (spec, count, seed) and takes no rank input — each rank calls
it independently and must arrive at an elementwise-identical vector, which is what keeps the
harness's cross-rank equality check provable instead of deleted. `count` is per-rank
(local_batch_size = batch_size // dp_size), never the global batch.

Two constraints follow, and neither is incidental:

- Stdlib only. profile_decode.py imports this before spawning the TP ranks, so nothing here
  may pull in torch/numpy/sglang, at module level or lazily inside a function.
- normal-mode draws come from a PRIVATE random.Random(seed). The benchmark's acceptance coins
  are drawn from the module-level `random` stream; consuming even one value from it here would
  shift that sequence and change `realized_accept_length`, which is the correctness gate for
  the whole harness. The private RNG is the mechanism, not a stylistic preference.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TypeVar
import math
import random

_T = TypeVar("_T")

# profile_decode.py bootstraps a populated prefix of len-1 before the first verify step,
# so a request shorter than 2 has no room for it.
MIN_ISL = 2

MODES = ("uniform", "bimodal", "normal", "list")


@dataclass(frozen=True)
class IslSpec:
    mode: str
    raw: str
    value: int | None = None                            # uniform
    value_a: int | None = None                          # bimodal, the side `ratio` describes
    value_b: int | None = None
    ratio: float | None = None
    ratio_exact: Fraction | None = field(default=None, repr=False, compare=False)
    mean: float | None = None                           # normal
    std: float | None = None
    lo: int | None = None
    hi: int | None = None                               # None means no upper clamp
    values: tuple[int, ...] | None = None               # list
    # Observability only, filled in by the most recent normal-mode generate(); see _normal_draw.
    # None here is load-bearing and distinct from 0: it means generate() has not run yet, so
    # describe() is reporting "unknown", not "the clamp ate nothing".
    clamped_low: int | None = field(default=None, repr=False, compare=False)
    clamped_high: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown ISL mode {self.mode!r}; expected one of {', '.join(MODES)}")
        if self.mode == "uniform":
            _require_isl(self.value, "uniform value")
        elif self.mode == "bimodal":
            _require_isl(self.value_a, "bimodal first value")
            _require_isl(self.value_b, "bimodal second value")
            if self.ratio_exact is None or not 0 < self.ratio_exact < 1:
                raise ValueError(f"bimodal ratio must satisfy 0 < ratio < 1, got {self.ratio!r}")
        elif self.mode == "normal":
            if not (isinstance(self.mean, float) and math.isfinite(self.mean)):
                raise ValueError(f"normal mean must be a finite float, got {self.mean!r}")
            if not (isinstance(self.std, float) and math.isfinite(self.std)) or self.std < 0:
                raise ValueError(f"normal stddev must be a finite float >= 0, got {self.std!r}")
            lo = _require_isl(self.lo, "normal lower clamp")
            if self.hi is not None and _require_isl(self.hi, "normal upper clamp") < lo:
                raise ValueError(f"normal clamp is empty: lo={self.lo} > hi={self.hi}")
        else:
            if not self.values:
                raise ValueError("list: spec has no values")
            for index, value in enumerate(self.values):
                _require_isl(value, f"list value #{index}")

    def generate(self, count, seed):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"ISL count must be a positive int, got {count!r}")
        if self.mode == "uniform":
            result = [self.value] * count
        elif self.mode == "bimodal":
            n_a, n_b = self._bimodal_split(count)
            # Not shuffled: cross-rank determinism is cheapest to guarantee when nothing moves.
            # A caller who wants the sides interleaved can spell it out with list:.
            result = [self.value_a] * n_a + [self.value_b] * n_b
        elif self.mode == "normal":
            result = self._normal_draw(count, seed)
        else:
            values = _set_by_post_init(self.values, "list values")
            if len(values) != count:
                raise ValueError(
                    f"list: spec has {len(values)} values but this batch needs {count}. "
                    f"The ISL spec is PER ATTENTION-DP RANK, so the count to match is "
                    f"local_batch_size = batch_size // dp_size = {count}; a global batch of "
                    f"{count} * dp_size requests still takes {count} values here."
                )
            result = list(values)
        if len(result) != count:
            raise ValueError(f"Internal error: generated {len(result)} lengths for count={count}")
        for value in result:
            if not isinstance(value, int) or isinstance(value, bool) or value < MIN_ISL:
                raise ValueError(f"Generated ISL {value!r} is not an int >= {MIN_ISL}")
        return result

    def describe(self):
        described: dict = {"mode": self.mode, "raw": self.raw}
        if self.mode == "uniform":
            described["value"] = self.value
        elif self.mode == "bimodal":
            described.update(value_a=self.value_a, value_b=self.value_b, ratio=self.ratio)
        elif self.mode == "normal":
            described.update(mean=self.mean, std=self.std, lo=self.lo, hi=self.hi,
                             clamped_low=self.clamped_low, clamped_high=self.clamped_high)
        else:
            described["values"] = list(_set_by_post_init(self.values, "list values"))
        return described

    def _bimodal_split(self, count):
        # Exact rational arithmetic rather than float: "10% of 70" has to be 7, but
        # 70 * 0.1 is 7.000000000000001 in IEEE and would ceil to 8.
        share = _set_by_post_init(self.ratio_exact, "bimodal ratio")
        if share < Fraction(1, 2):
            n_a = math.ceil(count * share)              # side a is the minority; it rounds UP
            n_b = count - n_a
        elif share > Fraction(1, 2):
            n_b = math.ceil(count * (1 - share))        # mirror: side b is the minority
            n_a = count - n_b
        else:
            n_a = math.ceil(Fraction(count, 2))         # no minority; the tie breaks toward a
            n_b = count - n_a
        if n_a < 1 or n_b < 1:
            raise ValueError(
                f"bimodal:{self.value_a},{self.value_b},{self.ratio} degenerates at count={count}: "
                f"{n_a} request(s) of {self.value_a} and {n_b} of {self.value_b}. A bimodal batch "
                f"that holds only one length is a uniform batch reported under a false label; "
                f"raise count or move the ratio toward 0.5."
            )
        return n_a, n_b

    def _normal_draw(self, count, seed):
        # Private stream. Drawing from the module-level random.* would shift the benchmark's
        # acceptance coin sequence and change realized_accept_length, the correctness gate.
        rng = random.Random(seed)
        mean = _set_by_post_init(self.mean, "normal mean")
        std = _set_by_post_init(self.std, "normal stddev")
        lo = _set_by_post_init(self.lo, "normal lower clamp")
        drawn, low_hits, high_hits = [], 0, 0
        for _ in range(count):
            value = round(rng.gauss(mean, std))
            # Saturate rather than resample: resampling makes the number of draws depend on the
            # values drawn, so any later change to the clamp reshuffles the whole sequence.
            # Saturation is reproducible, and describe() reports how much of the tail it ate.
            if value < lo:
                value, low_hits = lo, low_hits + 1
            elif self.hi is not None and value > self.hi:
                value, high_hits = self.hi, high_hits + 1
            drawn.append(int(value))
        object.__setattr__(self, "clamped_low", low_hits)
        object.__setattr__(self, "clamped_high", high_hits)
        return drawn


def parse_isl_spec(spec):
    if not isinstance(spec, str):
        raise ValueError(f"ISL spec must be a string, got {type(spec).__name__}")
    text = spec.strip()
    if not text:
        raise ValueError("ISL spec is empty; expected an integer or uniform:/bimodal:/normal:/list:")
    if ":" not in text:
        # A bare integer keeps --input-len 70000 meaning exactly what it always meant.
        return IslSpec(mode="uniform", raw=spec, value=_parse_int(text, "uniform value", spec))
    mode, _, body = text.partition(":")
    mode, body = mode.strip().lower(), body.strip()
    if mode == "uniform":
        return IslSpec(mode="uniform", raw=spec, value=_parse_int(body, "uniform value", spec))
    if mode == "bimodal":
        parts = _split_fields(body, spec, "bimodal", "<a>,<b>,<ratio>", (3,))
        ratio = _parse_ratio(parts[2], spec)
        return IslSpec(mode="bimodal", raw=spec,
                       value_a=_parse_int(parts[0], "bimodal first value", spec),
                       value_b=_parse_int(parts[1], "bimodal second value", spec),
                       ratio=float(ratio), ratio_exact=ratio)
    if mode == "normal":
        parts = _split_fields(body, spec, "normal", "<mean>,<std>[,<lo>,<hi>]", (2, 4))
        lo = _parse_int(parts[2], "normal lower clamp", spec) if len(parts) == 4 else MIN_ISL
        hi = _parse_int(parts[3], "normal upper clamp", spec) if len(parts) == 4 else None
        return IslSpec(mode="normal", raw=spec,
                       mean=_parse_float(parts[0], "normal mean", spec),
                       std=_parse_float(parts[1], "normal stddev", spec), lo=lo, hi=hi)
    if mode == "list":
        values = _read_list_file(body[1:].strip(), spec) if body.startswith("@") else \
            [_parse_int(part, f"list value #{index}", spec)
             for index, part in enumerate(_split_fields(body, spec, "list", "<v>,<v>,...", None))]
        return IslSpec(mode="list", raw=spec, values=tuple(values))
    raise ValueError(f"Unknown ISL mode {mode!r} in {spec!r}; expected one of {', '.join(MODES)}")


def _require_isl(value, what) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < MIN_ISL:
        raise ValueError(f"{what} must be an int >= {MIN_ISL} (the bootstrap needs a prefix), got {value!r}")
    return value


def _set_by_post_init(value: _T | None, what: str) -> _T:
    # Every mode-specific field is Optional because one dataclass covers four modes, but
    # __post_init__ has already proven the fields this mode needs are populated. This asserts
    # that at the point of use, which both documents the invariant and lets a type checker
    # follow it across the mode dispatch.
    if value is None:
        raise ValueError(f"Internal error: {what} is unset on an already-validated IslSpec")
    return value


def _parse_int(text, what, spec):
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        raise ValueError(f"{what} must be an integer, got {text!r} in ISL spec {spec!r}") from None


def _parse_float(text, what, spec):
    try:
        value = float(text.strip())
    except (AttributeError, ValueError):
        raise ValueError(f"{what} must be a number, got {text!r} in ISL spec {spec!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {text!r} in ISL spec {spec!r}")
    return value


def _parse_ratio(text, spec):
    try:
        ratio = Fraction(text.strip())
    except (AttributeError, ValueError, ZeroDivisionError):
        raise ValueError(f"bimodal ratio must be a number, got {text!r} in ISL spec {spec!r}") from None
    if not 0 < ratio < 1:
        raise ValueError(f"bimodal ratio must satisfy 0 < ratio < 1, got {text!r} in ISL spec {spec!r}")
    return ratio


def _split_fields(body, spec, mode, shape, allowed):
    parts = [part.strip() for part in body.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"ISL spec {spec!r} has an empty field; expected {mode}:{shape}")
    if allowed is not None and len(parts) not in allowed:
        expected = " or ".join(str(count) for count in allowed)
        raise ValueError(f"ISL spec {spec!r} has {len(parts)} field(s), expected {expected}: {mode}:{shape}")
    return parts


def _read_list_file(path_text, spec):
    if not path_text:
        raise ValueError(f"ISL spec {spec!r} is missing a path after '@'")
    path = Path(path_text)
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"Cannot read ISL list file {str(path)!r} from spec {spec!r}: {error}") from None
    values = []
    for number, line in enumerate(lines, start=1):
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            values.append(_parse_int(stripped, f"{path}:{number}", spec))
    if not values:
        raise ValueError(f"ISL list file {str(path)!r} contains no values (spec {spec!r})")
    return values
