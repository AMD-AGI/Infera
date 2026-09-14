#!/usr/bin/env python3
"""From a device kernel symbol to the source that declares it.

Magpie's `amd_kernel_finder` resolves by grepping names across whole
repositories, and on the sample profile that produced one hit out of five — and
the one was wrong: `kernels/ops/kvcache/triton_store_cache.py` for an
`add_rmsnorm_quant` kernel. Both failures have the same cause. A device symbol
carries three kinds of text mixed together, and only one of them identifies the
operator:

    _ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256E...
    ^^^^^^^^^^ mangling      ^^^^^^^^^^^^^^^^^^^^^^^ the name    ^^^^ template args

    mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32
    ^^^^^^^^^^^^^^^^^^ the name       ^^^^^^^^^^^^^^^^^^^^^^^^^ tile and tuning

Splitting on word boundaries and testing tokens individually destroys exactly
the part that discriminates: `quant` matches half of aiter, `add` matches all of
it. Keeping the **compound name contiguous** is what makes the search specific,
and on the sample profile it finds real sources for kernels the token search
missed entirely.

Measured on `aiter` and `sglang` as installed in
`lmsysorg/sglang:v0.5.18-rocm720-mi35x`:

| symbol | probe | file found |
|---|---|---|
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_...` | `mfma_moe1_silu_mul` | `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` |
| `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32...` | `gemm_a16_w16_kernel` | `aiter/ops/triton/gemm/basic/gemm_a16w16.py` |

Nothing here is certain, and the caller is told which rule fired. A JIT-generated
symbol such as TileLang's `main_kernel` carries no name at all and is reported as
unresolvable rather than matched to whatever happens to contain the word.
"""

from __future__ import annotations

import re
from pathlib import Path

#: A run of snake_case identifier characters. The unit the search keeps whole.
#:
#: **Nothing is stripped from the middle of a compound name**, and that was
#: measured. An earlier version removed tokens it classified as tuning, which
#: turned `gemm_a16_w16_kernel_BLOCK_SIZE_M` into `gemm_kernel` and matched
#: `aiter/tuned_gemm.py` — a dispatch table, not the kernel. `a16` and `w16` name
#: the operand precisions and are part of the operator's identity; no rule
#: separates them from `t32x128x256` reliably, because the difference is
#: semantic. Shortening from the right, stopping at the first match, gets the
#: same tuning suffixes off without having to classify anything.
_COMPOUND = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)+")

#: Names that identify no operator. `main_kernel` is what TileLang calls every
#: kernel it generates; matching it against source would find the wrong thing
#: with high confidence, which is worse than finding nothing.
GENERIC = frozenset({"main_kernel", "main", "kernel", "entry", "run", "_kernel"})


def demangle_head(symbol: str) -> str:
    """The readable head of an Itanium-mangled name, or the input unchanged.

    `_ZN5aiter24add_rmsnorm_quant_kernelI...` is `_ZN` followed by
    length-prefixed components. Reading the lengths is what separates
    `add_rmsnorm_quant_kernel` from the namespace and from the template
    arguments; a regex on word characters cannot, which is why
    `aiter24add_rmsnorm_quant_kernelIDF16b...` came out of the earlier attempt.
    """
    if not symbol.startswith("_ZN"):
        return symbol
    rest = symbol[3:]
    components = []
    while rest:
        match = re.match(r"(\d+)", rest)
        if not match:
            break
        length = int(match.group(1))
        start = match.end()
        component = rest[start : start + length]
        if len(component) < length:
            break
        components.append(component)
        rest = rest[start + length :]
    return components[-1] if components else symbol


def probe_of(symbol: str) -> str:
    """The contiguous compound name to search for, or `""` when there is none.

    Returns `""` for a symbol that names no operator — a TileLang `main_kernel`,
    or anything left with fewer than two underscore-joined parts. An empty probe
    means *this symbol cannot be resolved by name*, which is a usable answer.
    """
    text = demangle_head(symbol)
    text = re.sub(r"^(void\s+|\(anonymous namespace\)::)", "", text)
    text = re.split(r"[<(]", text)[0].strip()
    text = re.sub(r"^_+", "", text)

    candidates = [c for c in _COMPOUND.findall(text) if c.lower() not in GENERIC]
    if not candidates:
        return ""

    # **Leftmost, not longest.** The operator's name comes first and the tuning
    # parameters follow it — a Triton JIT symbol is
    # `<operator>_BLOCK_SIZE_M_32_..._cache_modifier_CG_activation_NONE_...`,
    # where the trailing run is longer than the name. Taking the longest
    # candidate turned `_gemm_a16_w16_kernel_BLOCK_SIZE_M_32_...` into a search
    # for `cache_modifier` and matched `aiter/ops/gemm_op_a8w8.py`, a different
    # operator entirely. Measured on the sample profile.
    name = candidates[0].rstrip("_")
    if name.lower() in GENERIC or name.count("_") < 1:
        return ""
    return name


def _probes(name: str) -> list[str]:
    """The probe and its progressively shorter prefixes, longest first.

    A symbol often carries a suffix the source does not — `_kernel` on a Triton
    JIT name whose Python function is the bare operator. Shortening stops at two
    underscore-joined parts, because a single word is not specific enough to
    corroborate anything.
    """
    parts = name.split("_")
    return ["_".join(parts[:n]) for n in range(len(parts), 1, -1)]


def find_source(
    symbol: str,
    repos: dict[str, Path],
    *,
    suffixes: tuple[str, ...] = (".py", ".cpp", ".cu", ".hip", ".h", ".hpp", ".cuh"),
    max_files: int = 60000,
) -> dict:
    """Search indexed repositories for the source that declares `symbol`.

    `repos` maps a repository name to its checkout root. Returns

        {"relative": "<repo-relative path>", "repo": "<name>",
         "probe": "<what was searched>", "why": "<how it was decided>"}

    with `relative` empty when nothing was found. Files whose text contains the
    probe **as a definition** — `def <probe>`, `class <probe>`, or `<probe>(` —
    outrank files that merely mention it, because a call site is not where an
    operator lives.
    """
    probe = probe_of(symbol)
    if not probe:
        return {
            "relative": "",
            "repo": "",
            "probe": "",
            "why": "symbol carries no operator name (generated or generic)",
        }

    for attempt in _probes(probe):
        definitions: list[tuple[str, str]] = []
        mentions: list[tuple[str, str]] = []
        defines = re.compile(
            rf"(?:^|\n)\s*(?:def|class)\s+{re.escape(attempt)}\b"
            rf"|(?:^|\n)[\w:<>,\s*&]*\b{re.escape(attempt)}\s*\("
        )
        scanned = 0
        for repo_name, root in repos.items():
            for path in sorted(Path(root).rglob("*")):
                if scanned >= max_files:
                    break
                if not path.is_file() or path.suffix not in suffixes:
                    continue
                scanned += 1
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if attempt not in text:
                    continue
                relative = str(path.relative_to(root))
                (definitions if defines.search(text) else mentions).append((repo_name, relative))

        for bucket, why in ((definitions, "defines"), (mentions, "mentions")):
            if bucket:
                # Shortest path first: an operator's own module outranks an
                # aggregating `__init__.py` or a dispatch table that lists it.
                repo_name, relative = min(bucket, key=lambda pair: (len(pair[1]), pair[1]))
                exact = " (exact)" if attempt == probe else f" (shortened from {probe!r})"
                return {
                    "relative": relative,
                    "repo": repo_name,
                    "probe": attempt,
                    "why": f"{why} {attempt!r}{exact}",
                }

    return {
        "relative": "",
        "repo": "",
        "probe": probe,
        "why": f"no indexed file contains {probe!r} or any prefix of it",
    }
