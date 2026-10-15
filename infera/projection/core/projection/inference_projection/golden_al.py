###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Speculative acceptance, read from measured acceptance-length files.

How many drafts a target keeps is never made up here: it comes from a file of
measured acceptance lengths (AL, tokens committed per verify step including
the bonus token) keyed by draft depth. Two sets of files exist.

``configs/speculative/golden_al`` holds InferenceX's golden curves
(``golden_al_distribution``), copied verbatim. Each was measured with vLLM's
own acceptance counters on SPEED-Bench's Qualitative coding category at
temperature 1.0 and 4096 output tokens, per thinking mode, and is the value
InferenceX pins its throughput runs to. ``INFERASIM_GOLDEN_AL_DIR`` points the
loader at a live checkout instead.

``configs/speculative/defaults`` holds one curve per method, derived from the
golden files by ``derive_defaults`` (``python -m <this module>
--write-defaults`` regenerates them). They stand in for a model nobody has
measured.

A model names its own curves in its model YAML, under ``speculative``::

    speculative:
      default: dspark                # method served when none is asked for
      dspark:
        layers: 3                    # drafter shape, see speculative.drafter_for
        acceptance:                  # a file, or one per serving engine
          default: kimik3_dspark_probabilistic_...yaml
          atom: kimik3_dspark.yaml

A request resolves to a curve in this order: an explicit curve file, the
model's curve for the method, the model's curve for the sibling method (an
MTP head and an EAGLE3 head are the same autoregressive one-layer drafter
trained on the target's hidden states, so either stands in for the other),
then the method's default curve. A depth the curve does not cover is
interpolated between measured depths, or carried past the last one by a chain
fitted to the whole curve.
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from .speculative import AR_MAX, acceptance_length, canonical_method, chain_rates

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "speculative"
GOLDEN_DIR = _CONFIG_DIR / "golden_al"
DEFAULTS_DIR = _CONFIG_DIR / "defaults"
SPEED_BENCH_FILE = _CONFIG_DIR / "speed_bench_table1.yaml"

# The depths the default curves span: the sweep InferenceX runs.
DEFAULT_DEPTHS = tuple(range(1, 9))

THINKING_MODES = ("on", "off")

_SIBLING = {"mtp": "eagle3", "eagle3": "mtp"}

# Methods with golden files; the rest are derived from SPEED-Bench Table 1.
_GOLDEN_METHODS = ("mtp", "eagle3", "dspark")


def golden_dir() -> Path:
    env = os.environ.get("INFERASIM_GOLDEN_AL_DIR")
    return Path(env) if env else GOLDEN_DIR


@dataclass(frozen=True)
class Curve:
    """One acceptance-length file: AL by draft depth, per thinking mode."""

    name: str
    model: str
    method: str | None
    modes: dict  # "on" / "off" -> {depth: AL}

    def points(self, thinking: str = "on") -> tuple[dict, str]:
        """The depth sweep for ``thinking``, or the other mode if it was not run."""
        want = normalize_thinking(thinking)
        if self.modes.get(want):
            return self.modes[want], want
        for mode in THINKING_MODES:
            if self.modes.get(mode):
                return self.modes[mode], mode
        raise ValueError(f"acceptance file {self.name} has no depth sweep")


def normalize_thinking(value) -> str:
    if value is None or value == "":
        return "on"
    if isinstance(value, bool):
        return "on" if value else "off"
    key = str(value).strip().lower()
    if key in ("on", "true", "1", "thinking_on", "yes"):
        return "on"
    if key in ("off", "false", "0", "thinking_off", "no"):
        return "off"
    raise ValueError(f"speculative thinking mode must be on or off, got {value!r}")


def method_of_file(name: str) -> str | None:
    """The method a golden file measures, from the name InferenceX gives it."""
    stem = Path(name).stem.lower()
    for token in ("dspark", "eagle3", "mtp", "ngram", "draft_model"):
        if token in stem:
            return token
    return None


def _parse(path: Path) -> Curve:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path}: expected one top-level model key")
    ((model, body),) = data.items()
    modes = {}
    for mode in THINKING_MODES:
        sweep = (body or {}).get(f"thinking_{mode}")
        if sweep:
            modes[mode] = {int(k): float(v) for k, v in sweep.items()}
    if not modes:
        raise ValueError(f"{path}: no thinking_on / thinking_off sweep")
    return Curve(path.name, str(model), method_of_file(path.name), modes)


@cache
def _load(path: str) -> Curve:
    return _parse(Path(path))


def find_file(ref: str) -> Path:
    """A curve file by path, or by name in the golden or the defaults directory."""
    p = Path(str(ref)).expanduser()
    if p.is_file():
        return p
    name = p.name if p.suffix in (".yaml", ".yml") else f"{p.name}.yaml"
    for base in (golden_dir(), GOLDEN_DIR, DEFAULTS_DIR):
        cand = base / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no acceptance-length file {ref!r} in {golden_dir()} or {DEFAULTS_DIR}"
    )


def load_curve(ref: str) -> Curve:
    return _load(str(find_file(ref).resolve()))


def golden_curves(directory: str | Path | None = None) -> list[Curve]:
    base = Path(directory) if directory else golden_dir()
    return [_load(str(p.resolve())) for p in sorted(base.glob("*.yaml"))]


def default_curve(method: str) -> Curve:
    m = canonical_method(method) or "mtp"
    return _load(str((DEFAULTS_DIR / f"{m}.yaml").resolve()))


# ---- depth interpolation and extrapolation ----------------------------------


def _sse(points: dict, a: float, d: float) -> float:
    return sum((acceptance_length(chain_rates(a, d, k)) - al) ** 2 for k, al in points.items())


def _best_a(points: dict, d: float) -> float:
    """Golden-section search on ``a`` at fixed decay; the SSE is unimodal in it."""
    lo, hi = 0.0, AR_MAX
    g = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = _sse(points, x1, d), _sse(points, x2, d)
    for _ in range(40):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = _sse(points, x1, d)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = _sse(points, x2, d)
    return 0.5 * (lo + hi)


def _fit(items: tuple) -> tuple[float, float]:
    points = dict(items)
    best = (float("inf"), 0.0, 1.0)
    for i in range(101):
        d = 0.5 + i * 0.005
        a = _best_a(points, d)
        err = _sse(points, a, d)
        if err < best[0]:
            best = (err, a, d)
    centre = best[2]
    for i in range(-10, 11):
        d = centre + i * 0.0005
        if not 0.0 < d <= 1.0:
            continue
        a = _best_a(points, d)
        err = _sse(points, a, d)
        if err < best[0]:
            best = (err, a, d)
    return best[1], best[2]


_FIT_CACHE: dict = {}


def fit_chain(points: dict) -> tuple[float, float]:
    """Least-squares ``(a, decay)`` of a chain ``AR_j = a * decay**(j-1)`` through a sweep.

    A single measured depth fixes only the level, so it keeps the default
    MTP curve's decay.
    """
    items = tuple(sorted((int(k), float(v)) for k, v in points.items()))
    if items in _FIT_CACHE:
        return _FIT_CACHE[items]
    if len(items) >= 2:
        out = _fit(items)
    else:
        d = fit_chain(default_curve("mtp").points("on")[0])[1]
        k, al = items[0]
        out = (invert_rate(al, k, d), d)
    _FIT_CACHE[items] = out
    return out


def invert_rate(al: float, k: int, decay: float) -> float:
    """The first-position rate whose chain commits ``al`` tokens at depth ``k``."""
    if k <= 0 or al <= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    if acceptance_length(chain_rates(hi, decay, k)) <= al:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if acceptance_length(chain_rates(mid, decay, k)) < al:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def al_at(points: dict, k: int) -> float:
    """Acceptance length at depth ``k`` from a measured sweep.

    Exact at a measured depth; linear between two measured depths (and between
    depth 0, which commits only the bonus token, and the first one); past the
    last measured depth the fitted chain's gain is added to the last
    measurement, so the curve continues from what was measured rather than
    from the fit's own level. Never more than the ``k + 1`` a step can commit.
    """
    k = int(k)
    if k <= 0:
        return 1.0
    pts = {int(d): float(v) for d, v in points.items()}
    if k in pts:
        return pts[k]
    depths = sorted(pts)
    if k < depths[-1]:
        lo = max([0] + [d for d in depths if d < k])
        hi = min(d for d in depths if d > k)
        v_lo = 1.0 if lo == 0 else pts[lo]
        w = (k - lo) / (hi - lo)
        return min(float(k + 1), v_lo + w * (pts[hi] - v_lo))
    last = depths[-1]
    a, d = fit_chain(pts)
    gain = acceptance_length(chain_rates(a, d, k)) - acceptance_length(chain_rates(a, d, last))
    return min(float(k + 1), pts[last] + max(0.0, gain))


# ---- what a model serves -----------------------------------------------------


def _plain(obj):
    """A config block as plain dicts: model YAMLs arrive as ``SimpleNamespace``."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float)):
        return {str(k): _plain(v) for k, v in vars(obj).items()}
    return obj


def model_speculative(model_config) -> dict:
    """The model YAML's ``speculative`` block, with method names canonicalised."""
    raw = _plain(getattr(model_config, "speculative", None)) or {}
    out: dict = {}
    for key, val in raw.items():
        if key == "default":
            out["default"] = canonical_method(val)
            continue
        m = canonical_method(key)
        if m is not None and isinstance(val, dict):
            out[m] = val
    return out


def default_method(model_config) -> str:
    """The drafter a model is served with when the request does not name one.

    The model YAML's ``speculative.default``; else its own MTP head when the
    checkpoint ships one; else the one method the YAML gives a curve for; else
    an EAGLE3 head, the drafter trained for checkpoints that ship none.
    """
    spec = model_speculative(model_config)
    if spec.get("default"):
        return spec["default"]
    if int(getattr(model_config, "mtp_num_layers", 0) or 0) > 0:
        return "mtp"
    declared = [m for m in spec if m != "default"]
    if len(declared) == 1:
        return declared[0]
    return "eagle3"


def _model_file(spec: dict, method: str, engine: str | None) -> str | None:
    entry = (spec.get(method) or {}).get("acceptance")
    if isinstance(entry, dict):
        eng = str(engine or "").strip().lower()
        return entry.get(eng) or entry.get("default")
    return entry or None


@dataclass(frozen=True)
class Acceptance:
    """The curve a projection's acceptance is read from, and where it came from."""

    points: dict
    source: str

    def at(self, k: int) -> float:
        return al_at(self.points, k)

    @property
    def decay(self) -> float:
        return fit_chain(self.points)[1]


def resolve(
    method: str,
    *,
    model_config=None,
    engine: str | None = None,
    thinking="on",
    curve: str | None = None,
) -> Acceptance:
    """The acceptance curve for ``method`` on this model, engine and thinking mode."""
    m = canonical_method(method) or "mtp"
    spec = model_speculative(model_config) if model_config is not None else {}
    candidates = []
    if curve:
        candidates.append(("file", curve))
    own = _model_file(spec, m, engine)
    if own:
        candidates.append(("model", own))
    sib = _SIBLING.get(m)
    if sib:
        sib_file = _model_file(spec, sib, engine)
        if sib_file:
            candidates.append((f"model {sib} as {m}", sib_file))
    for kind, ref in candidates:
        c = load_curve(ref)
        pts, mode = c.points(thinking)
        return Acceptance(pts, f"{kind}:{c.name}[thinking_{mode}]")
    c = default_curve(m)
    pts, mode = c.points(thinking)
    return Acceptance(pts, f"default:{c.name}[thinking_{mode}]")


# ---- the default curves ------------------------------------------------------


def _extended(points: dict) -> dict:
    return {k: al_at(points, k) for k in DEFAULT_DEPTHS}


def _monotone(sweep: dict) -> dict:
    out, run = {}, 1.0
    for k in sorted(sweep):
        run = max(run, sweep[k])
        out[k] = round(run, 2)
    return out


def _speed_bench() -> dict:
    return yaml.safe_load(SPEED_BENCH_FILE.read_text())


def speed_bench_al(method: str, domains) -> list[float]:
    """SPEED-Bench Table 1 acceptance lengths for ``method`` over ``domains``."""
    table = _speed_bench()["acceptance_length"][method]
    return [float(al) for dom in domains for al in table[dom]]


def derive_defaults(directory: str | Path | None = None) -> dict:
    """One default curve per method, from every golden file.

    For a method with golden files: each file's sweep is carried to depths
    1..8 (``al_at``), the files of one model are averaged so a model measured
    twice counts once, and the median across models is taken per depth and
    thinking mode, held non-decreasing. A mode no model measured takes the
    other mode's curve.

    The vanilla draft model and n-gram lookup have no golden files. Their
    level is SPEED-Bench Table 1's coding acceptance at its draft length of 3
    (the golden files are coding too), and the depth shape is the chain decay
    of the pooled MTP and EAGLE3 defaults, carried from that one point.
    """
    by_method: dict = {}
    for c in golden_curves(directory):
        if c.method not in _GOLDEN_METHODS:
            continue
        for mode in THINKING_MODES:
            if c.modes.get(mode):
                by_method.setdefault(c.method, {}).setdefault(mode, {}).setdefault(
                    c.model, []
                ).append(_extended(c.modes[mode]))
    out: dict = {}
    for m in _GOLDEN_METHODS:
        modes = by_method.get(m, {})
        curves = {}
        for mode, models in modes.items():
            per_model = [
                {k: statistics.fmean(s[k] for s in sweeps) for k in DEFAULT_DEPTHS}
                for sweeps in models.values()
            ]
            curves[mode] = _monotone(
                {k: statistics.median(s[k] for s in per_model) for k in DEFAULT_DEPTHS}
            )
        for mode in THINKING_MODES:
            if mode not in curves and curves:
                curves[mode] = dict(next(iter(curves.values())))
        out[m] = {
            "curves": curves,
            "models": {mode: sorted(models) for mode, models in modes.items()},
        }
    pooled = {
        k: statistics.fmean(out[m]["curves"]["on"][k] for m in ("mtp", "eagle3"))
        for k in DEFAULT_DEPTHS
    }
    decay = fit_chain(pooled)[1]
    dl = int(_speed_bench()["draft_length"])
    for m in ("draft_model", "ngram"):
        level = statistics.fmean(speed_bench_al(m, ["Coding"]))
        a = invert_rate(level, dl, decay)
        sweep = _monotone({k: acceptance_length(chain_rates(a, decay, k)) for k in DEFAULT_DEPTHS})
        out[m] = {"curves": {"on": sweep, "off": dict(sweep)}, "level": level, "decay": decay}
    return out


def render_default(method: str, entry: dict) -> str:
    """A default curve as a YAML file in the golden files' own layout."""
    lines = [
        f"# Default acceptance length for {method}, for a model with no measured curve.",
        "# Generated by golden_al.derive_defaults; regenerate with",
        "#   python -m infera.projection.core.projection.inference_projection.golden_al"
        " --write-defaults",
    ]
    if "models" in entry:
        lines.append(
            "# Median across models of the InferenceX golden curves (SPEED-Bench coding,"
            " temperature 1.0),"
        )
        lines.append("# each carried to depths 1-8, a model's own files averaged first.")
        for mode, models in entry["models"].items():
            lines.append(f"# thinking_{mode}: {', '.join(models)}")
    else:
        lines.append(
            f"# Level: SPEED-Bench Table 1 coding AL at draft length 3 ({entry['level']:.3f});"
        )
        lines.append(
            f"# depth shape: chain decay {entry['decay']:.4f} of the pooled MTP/EAGLE3 defaults."
        )
    lines.append("#")
    lines.append("# key = num_speculative_tokens; value = AL including the verified token")
    lines.append(f"default-{method}:")
    for mode in THINKING_MODES:
        sweep = entry["curves"].get(mode)
        if not sweep:
            continue
        lines.append(f"  thinking_{mode}:")
        lines.extend(f"    {k}: {v:.2f}" for k, v in sorted(sweep.items()))
    return "\n".join(lines) + "\n"


def write_defaults(directory: str | Path | None = None) -> list[Path]:
    DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for m, entry in derive_defaults(directory).items():
        path = DEFAULTS_DIR / f"{m}.yaml"
        path.write_text(render_default(m, entry))
        written.append(path)
    _load.cache_clear()
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-defaults", action="store_true", help="regenerate the default curves")
    ap.add_argument("--golden-dir", default=None, help="golden files to derive from")
    ns = ap.parse_args()
    if ns.write_defaults:
        for p in write_defaults(ns.golden_dir):
            print(p)
    else:
        for m, entry in derive_defaults(ns.golden_dir).items():
            print(render_default(m, entry))
