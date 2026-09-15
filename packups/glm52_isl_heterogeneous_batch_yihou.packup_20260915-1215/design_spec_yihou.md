# Design spec — heterogeneous ISL, Plan 1

Authoritative interface contract. Teammates implement against **this**, not against each other.
File ownership is disjoint by construction; nobody edits a file another person owns.

| owner | files |
|---|---|
| teammate `isl-spec` | `bench/isl_spec.py` (new), `tests/test_isl_spec_yihou.py` (new) |
| teammate `harness-core` | `bench/batch_state.py`, `bench/topology.py`, `tests/test_batch_state_yihou.py`, `tests/test_topology_yihou.py` |
| leader (yihou) | `bench/profile_decode.py`, `scripts/*`, `README.md`, integration, smoke run |

---

## 0. The invariant that governs everything

**Plan 1: every attention-DP rank receives the same ISL multiset, in the same order.**

The generator is a pure function of `(spec, count, seed)` with **no rank input**, so all 8 ranks
compute an elementwise-identical list independently. Cross-rank equality is therefore still
provable, and we keep the harness's strongest check instead of deleting it.

Two consequences that are requirements, not side effects:

- The generator **must not consume from any global RNG** (`random`, `numpy.random`, `torch`). It
  owns a private `random.Random(seed)`. Perturbing a global stream would shift the acceptance
  coins and change `realized_accept_length`, which is the correctness gate.
- `local_batch_size = batch_size // dp_size` is the count the spec describes. **The spec is
  per-rank, not global.** Error messages must say so with the arithmetic spelled out.

---

## 1. `bench/isl_spec.py` — teammate `isl-spec`

Stdlib only. No `torch`, no `numpy`, no `sglang` import, at module level or inside functions.
`profile_decode.py` imports this before spawning ranks, so it must stay import-cheap.

### Grammar

```
uniform:<int>
bimodal:<int>,<int>,<float>
normal:<float>,<float>[,<int>,<int>]      # mean, stddev [, lo, hi]
list:<int>,<int>,...
list:@<path>                              # one integer per line; blank lines and #-comments skipped
```

A bare integer (`"70000"`) is accepted and means `uniform:70000`. This is what keeps
`--input-len 70000` working.

### API

```python
@dataclass(frozen=True)
class IslSpec:
    mode: str            # "uniform" | "bimodal" | "normal" | "list"
    raw: str             # the original spec string, verbatim
    # mode-specific fields, all Optional

    def generate(self, count: int, seed: int) -> list[int]: ...
    def describe(self) -> dict: ...   # JSON-serialisable; goes into config_yihou.json

def parse_isl_spec(spec: str) -> IslSpec: ...
```

### Semantics per mode

**`uniform:<v>`** — `[v] * count`. Consumes no randomness.

**`bimodal:<a>,<b>,<ratio>`** — `ratio` is the share of requests that get value `a`;
`1 - ratio` get `b`. `0 < ratio < 1`.

> **The minority side rounds UP.** Fixed by the user. If `ratio < 0.5`, side `a` is the minority
> and `n_a = ceil(count * ratio)`, `n_b = count - n_a`. If `ratio > 0.5`, side `b` is the minority
> and `n_b = ceil(count * (1 - ratio))`, `n_a = count - n_b`. At exactly `ratio == 0.5` there is no
> minority; break the tie toward `a`: `n_a = ceil(count / 2)`.

Result ordering is **deterministic and not shuffled**: all `a` first, then all `b`. Do not
randomise the order — determinism across ranks is cheaper to guarantee when nothing is shuffled,
and a caller who wants interleaving can use `list:`. Consumes no randomness.

Guard: `n_a >= 1` and `n_b >= 1` after rounding, else raise with the computed numbers in the
message — a "bimodal" batch that is actually uniform is a silent lie about what was measured.

**`normal:<mean>,<std>[,<lo>,<hi>]`** — `count` draws from `random.Random(seed).gauss(mean, std)`,
each `round()`ed to int then clamped to `[lo, hi]`. Defaults `lo = 2`, `hi = None` (no upper
clamp). `lo` must be `>= 2`. `std >= 0`; `std == 0` is legal and degenerates to uniform.

Clamp by **saturation, not resampling.** Resampling would make the draw count depend on the values
drawn, which makes the sequence fragile to any future change. Saturation is reproducible and its
distortion is visible: `describe()` must report `clamped_low` / `clamped_high` counts so a caller
can see when the clamp ate the tail.

Sorted output? **No — emit in draw order.** Sorting would hide the draw sequence.

**`list:<v>,<v>,...`** / **`list:@<path>`** — used verbatim. `len(values)` must equal `count`;
otherwise raise, and the message must contain both numbers and the
`batch_size // dp_size` arithmetic that produced `count`. Consumes no randomness.

### Universal post-conditions, enforced inside `generate()`

- `len(result) == count`
- every value `>= 2` (`profile_decode.py:73-74` requires a populated-prefix bootstrap)
- every value is a Python `int`, not `float`/`numpy` scalar
- `generate(count, seed)` called twice returns equal lists (self-check this in tests)

### `describe()`

Returns a dict recorded verbatim into `config_yihou.json`. Must include at minimum:
`mode`, `raw`, and for `normal` also `mean`, `std`, `lo`, `hi`, `clamped_low`, `clamped_high`.
The leader additionally records the **realized list** separately — `describe()` need not carry it.

### Tests (`tests/test_isl_spec_yihou.py`)

Cover, at least: bare-int back-compat; each mode's happy path; the ceil rule at
`count=32, ratio=0.1 -> 4/28` and at `ratio=0.5 -> 16/16`; the `ratio>0.5` mirror case; the
degenerate-bimodal guard; `normal` determinism across two calls and across two fresh
`IslSpec` objects; clamp saturation counts; `list:` length mismatch message content; `list:@file`
parsing incl. comments/blanks; rejection of every malformed spec you can think of;
**and an explicit assertion that `generate()` does not perturb `random.random()` /
`random.getstate()` globally.**

---

## 2. `bench/batch_state.py` — teammate `harness-core`

### `required_token_capacity` — DO NOT CHANGE

It stays scalar. The caller (leader) passes `max(input_lens)`, because allocation is uniform-max
by design (mission F3). Leaving this function untouched removes it from the regression surface.

### `validate_worker_progress` — DO NOT CHANGE

Already elementwise (mission F4).

### `DecodeAccounting` — the only change in this file

`input_len: int` becomes `input_lens: list[int]`.

```python
@dataclass
class DecodeAccounting:
    batch_size: int
    input_lens: list          # one entry per request, len == batch_size
    output_len: int
    max_accept_len: int = 6
```

- `__post_init__`: validate `len(input_lens) == batch_size`, every entry `>= 1`, and the existing
  positivity checks on `batch_size` / `output_len` / `max_accept_len`.
- `seq_lens` property becomes `[base + count for base, count in zip(self.input_lens, self.emitted)]`.
- `complete`, `record` — **unchanged**. Acceptance is per-request already and ISL does not enter.
- `summary()`:
  - add `"input_lens": list(self.input_lens)`
  - keep `"input_len"` for backward compatibility, set to the scalar **iff all entries are equal**,
    otherwise `None`. Do not emit a mean or a max there — a single number in that field is read by
    downstream tooling as "this is the ISL", and for a mixed batch that would be a false statement.
  - add `"input_len_uniform": bool`

Everything else in `summary()` is untouched.

---

## 3. `bench/topology.py` — teammate `harness-core`

### `rank_progress_signature` — the load-bearing change

Current line 69 asserts two different things at once:

```python
valid = valid and len(set(accept_lens)) == 1 and len(set(previous_lens)) == 1
```

**Keep the `accept_lens` half. Delete only the `previous_lens` half.** Mission F2: the acceptance
half is true by upstream construction (`spec_utils.py:411-419` draws one scalar and broadcasts it)
and it is what makes `realized_accept_length` bit-reproducible. Weakening it would remove a real
check for no reason.

Replace the positional length fields with an order-sensitive digest, so the signature covers the
whole vector rather than element 0:

```python
_DIGEST_MOD = (1 << 61) - 1

def lens_digest(values):
    """Order-sensitive rolling hash; fits int64. Divergence in ANY position or in
    ordering changes the digest, which element-0 comparison could not detect."""
    digest = 0
    for value in values:
        digest = (digest * 1000003 + int(value) + 1) % _DIGEST_MOD
    return digest
```

Field 3 becomes `lens_digest(previous_lens)`, field 5 becomes `lens_digest(new_lens)`. Field 4
stays `accept_lens[0]` — they are asserted equal. The signature stays 7 int64s, so
`progress_buffers` in `profile_decode.py` needs no resize (leader will confirm).

#### AMENDMENT 1 (leader, 2026-09-15 11:2x) — `final_len` becomes `final_lens`

Gap in the original spec, found while wiring `profile_decode.py`. Field 6 is the completion flag:

```python
int(bool(new_lens) and all(length >= final_len for length in new_lens))
```

With a ragged batch each request finishes at its own `isl_i + output_len`, so a single scalar
cannot express it. **Change the parameter `final_len` to `final_lens: list[int]`** and compare
elementwise:

```python
int(bool(new_lens) and all(new >= final for new, final in zip(new_lens, final_lens)))
```

Validate `len(final_lens) == len(new_lens)` as part of the `valid` flag rather than raising — the
never-raise-before-the-collective contract still applies. The signature stays 7 int64s. The leader
updates both call sites in `profile_decode.py`.

Keep the docstring contract: **never raise locally before the collective.** A local raise on one
rank while the others enter `all_gather` is a hang, not an error.

Order-sensitivity is deliberate and correct here: Plan 1 guarantees identical order, so an
ordering divergence is a real bug we want caught.

### `validate_rank_progress` — unchanged

### `aggregate_rank_summaries`

The `fields` equality tuple currently contains `"input_len"`. Under Plan 1 the ranks hold identical
ISL vectors, so:

- replace `"input_len"` with `"input_lens"` in `fields` — still an equality check, now stronger
- add `"input_len_uniform"` to `fields`
- leave `final_seq_lens` and `emitted_per_request` exactly as they are: still equality-checked
  across ranks, still concatenated across replicas for the global view

`input_len` (the back-compat scalar) is inherited from `first` via `dict(first)` and needs no
special handling.

### Tests

`tests/test_topology_yihou.py`: a heterogeneous-but-identical-across-ranks case must PASS; a case
where one rank's vector differs in a **non-zero position** must FAIL (this is precisely what the
old element-0 check would have missed — assert it explicitly); a case where two ranks hold the same
multiset in **different order** must FAIL; non-uniform `accept_lens` must still FAIL.

`tests/test_batch_state_yihou.py`: `DecodeAccounting` with a mixed vector; `seq_lens` tracks per
request; `summary()["input_len"]` is `None` when mixed and the scalar when uniform; length-mismatch
rejection.

---

## 4. `bench/profile_decode.py` — leader

Recorded here so teammates know the integration shape; **do not edit this file.**

- New `--input-len-spec STRING`. `--input-len INT` stays and keeps its default. Supplying both a
  non-default `--input-len` and `--input-len-spec` is an error.
- `allocate_batch`: `required_token_capacity(..., max(isl), ...)`; the truncation loop at 231-235
  becomes per-request (`zip(reqs, isl)`); the `context_len` guard uses `max(isl)`.
- `bootstrap`: `torch.tensor(isl)` instead of `torch.full`; `prefix_lens = [v - 1 for v in isl]`;
  `seq_lens_sum = sum(isl)`; `out_cache_loc` gathers with a per-row column index.
- `run_rank`: `final_len` per request at both `check_rank_progress` call sites; the progress log
  line stops printing `seq_lens[0]` as if it were "the" context.
- `config_yihou.json` gains `input_len_spec` (`describe()`) and `input_lens_realized`.
- Back-compat guard: with `--input-len 70000` and no spec, `config_yihou.json` must stay
  byte-compatible with the published packup.

---

## 5. Definition of done

1. `pytest tests/` green, including the new files.
2. A uniform-mode run reproduces `realized_accept_length = 3.6134393063583814` exactly. This is the
   gate; a run that misses it is measuring something else.
3. A heterogeneous smoke run (small ISL, `--max-steps`) completes and its per-request
   `final_seq_lens` show the expected spread.
4. Every claim in the final report is traceable to a log line or a code read.
