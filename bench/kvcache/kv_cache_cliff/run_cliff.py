###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KV Cache Cliff bench — mirrors LMCache's June 1 chart on MI300X.

Workload: shared 18K-token prefix + 2K-token per-client unique suffix
(ISL=20K), max_tokens=1 (prefill-only, OSL=1), warm cache. Sweep
concurrency. Two arms compared:

  - **Arm A (VRAM-only)**: vanilla vLLM, prefix-cache in VRAM only.
    Throughput cliff appears when concurrent KV exceeds VRAM budget;
    cached prefixes get evicted, every new request re-prefills.
  - **Arm B (VRAM + kvd-on-NVMe via v2)**: our chunked-fusion
    connector pointing at /mnt/nvme8. Evicted prefixes spill to
    NVMe; subsequent requests load from there in ~5 ms / 4.5 MiB
    chunk (per infera-kvd-l3-bench numbers — the connector round-trip
    bench that absorbed bench_packed_v2) instead of re-prefilling
    18K tokens.

Two arms must be benched separately (each requires a vLLM server
built differently). This script runs ONE arm at a time against a
pre-launched vLLM endpoint, then merges the two CSVs offline.

Usage:
  # Pre-launch vLLM separately in its own container.
  python -u -m bench.kvcache.kv_cache_cliff.run_cliff \\
      --endpoint http://localhost:8801 \\
      --model /PATH/TO/gpt-oss-120b \\
      --arm vram_only \\
      --isl 20000 --shared-prefix-tokens 18000 \\
      --concurrencies 1,2,4,8,16,32,48,64,80,100,128,160,200,250 \\
      --iters 3 --warmup-iters 1 \\
      --out results/cliff-vram-only.csv

  # Then re-run with the kvd-attached server:
  python -u -m bench.kvcache.kv_cache_cliff.run_cliff ... --arm kvd_v2 \\
      --out results/cliff-kvd-v2.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _ckpt(msg: str) -> None:
    print(f"[cliff] {msg}", flush=True)


# ---------------------------------------------------------------------
# Metrics scrape — vLLM Prometheus /metrics for prefix-cache hit rates
# ---------------------------------------------------------------------
#
# Prefix-cache counters live in vLLM's /metrics endpoint. Names vary
# slightly by build; we sum across all label sets (e.g. per-engine) and
# classify into L1 (in-VRAM GPU prefix cache) vs External (kvd/L3
# connector). Snapshotting before/after EACH timed iter yields the hit
# rate over exactly that measurement window — which lets us correlate
# arm B's bimodal latency variance with hit-rate drops (the 2026-06-04
# round-1 finding that the c=64 crossover is unstable).


def _parse_prometheus(text: str) -> dict[str, float]:
    """Sum Prometheus counter/gauge values by base metric name (labels
    stripped). Tolerant of comments, blank lines, and unparsable rows."""
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # format:  name{labels} value   OR   name value
        try:
            left, val = line.rsplit(" ", 1)
            value = float(val)
        except ValueError:
            continue
        name = left.split("{", 1)[0].strip()
        out[name] = out.get(name, 0.0) + value
    return out


def _classify_cache_counters(metrics: dict[str, float]) -> dict[str, float]:
    """Pull L1 (GPU prefix cache) and External (kvd/L3) query/hit counters
    out of a parsed /metrics dump. Sums every metric whose name ends in
    ``prefix_cache_queries_total`` / ``prefix_cache_hits_total`` (so it
    works across vllm:, vllm:gpu_, etc.); names containing ``external``
    go to the External bucket. Absent metrics → 0.0."""
    l1_q = l1_h = ext_q = ext_h = 0.0
    for name, val in metrics.items():
        is_ext = "external" in name
        if name.endswith("prefix_cache_queries_total"):
            if is_ext:
                ext_q += val
            else:
                l1_q += val
        elif name.endswith("prefix_cache_hits_total"):
            if is_ext:
                ext_h += val
            else:
                l1_h += val
    return {"l1_q": l1_q, "l1_h": l1_h, "ext_q": ext_q, "ext_h": ext_h}


async def _snap_cache(http_client, metrics_url: str) -> dict[str, float] | None:
    """GET /metrics and return classified cache counters, or None on
    failure (so a missing/erroring endpoint just blanks the columns)."""
    try:
        r = await http_client.get(metrics_url, timeout=15.0)
        if r.status_code != 200:
            return None
        return _classify_cache_counters(_parse_prometheus(r.text))
    except Exception:
        return None


def _window_rate(before: dict | None, after: dict | None, kq: str, kh: str) -> tuple[str, str, str]:
    """Delta query/hit counts and hit-rate % over a [before, after]
    snapshot window. Returns ("","","" ) when either snapshot is missing."""
    if before is None or after is None:
        return ("", "", "")
    dq = after[kq] - before[kq]
    dh = after[kh] - before[kh]
    rate = (100.0 * dh / dq) if dq > 0 else 0.0
    return (f"{dq:.0f}", f"{dh:.0f}", f"{rate:.1f}")


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Tokenizer — make "N tokens" actually mean N tokens.
#
# The old heuristic (words = N / 1.4) badly UNDER-shot: for this vocab
# the real ratio is ~1.0 token/word, so "60k" produced only ~44k tokens.
# When a tokenizer is available we build slightly-over-target text and
# truncate to EXACTLY the requested token count; the builders stay
# deterministic per client_id, so warm-cache (byte-identical prefix
# across runs) is preserved.
# ---------------------------------------------------------------------

_ACTIVE_TOKENIZER = None  # set once by set_active_tokenizer()
_TOKENIZER_TRIED = False

# tokens-per-word for this (plain-English) vocab. Default ~1.03 measured
# on gpt-oss-120b's tokenizer; refined at runtime by calibrate_word_ratio()
# against the live server's /tokenize endpoint (no client tokenizer needed).
_TOKENS_PER_WORD = 1.03


def set_active_tokenizer(model_path: str | None) -> None:
    """Load the model tokenizer once so prompt builders can hit exact
    token counts. Safe to call repeatedly; silently degrades to the
    word-ratio approximation if transformers / the tokenizer is
    unavailable."""
    global _ACTIVE_TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return
    _TOKENIZER_TRIED = True
    if not model_path:
        return
    try:
        from transformers import AutoTokenizer

        _ACTIVE_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _ckpt(f"tokenizer: loaded from {model_path} (exact token-count prompts enabled)")
    except Exception as exc:  # noqa: BLE001
        _ckpt(
            f"tokenizer: load failed ({type(exc).__name__}: {exc}); "
            f"falling back to ~1.0 word/token approximation"
        )
        _ACTIVE_TOKENIZER = None


def _approx_words_for_tokens(n_tokens: int) -> int:
    """Words needed to hit `n_tokens` tokens, using the calibrated
    tokens-per-word ratio for this vocab (~1.03; refined at runtime by
    calibrate_word_ratio()). A small +0.5%/+8 safety overshoot ensures
    we land AT-OR-ABOVE the target (so _fit_to_tokens can trim down to
    exact when a local tokenizer is present).

    The OLD code divided by a hard-coded 1.4 here, which under-shot to
    ~0.73x — that was the "60k → 44k tokens" bug."""
    return max(1, int(n_tokens / _TOKENS_PER_WORD * 1.005) + 8)


def _fit_to_tokens(text: str, n_tokens: int) -> str:
    """Truncate `text` to exactly `n_tokens` tokens using the active
    tokenizer. If no tokenizer is loaded, return `text` unchanged
    (it was already built to ~overshoot the target). Deterministic, so
    the result stays byte-stable for a given input → warm-cacheable."""
    tok = _ACTIVE_TOKENIZER
    if tok is None:
        return text
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= n_tokens:
        return text
    return tok.decode(ids[:n_tokens])


async def calibrate_word_ratio(http_client, base_url: str, model: str) -> None:
    """Measure this vocab's true tokens-per-word against the LIVE server
    (the served model's exact tokenizer) via the OpenAI /tokenize
    endpoint, and update the global ratio so `_approx_words_for_tokens`
    lands on the requested token count. No client-side tokenizer needed.

    Degrades silently to the default 1.03 if /tokenize is unavailable."""
    global _TOKENS_PER_WORD
    sample_words = 4000
    # Sample uses the SAME vocab the real prefixes use, so the measured
    # ratio matches production prompts.
    vocab = list(_PER_CLIENT_VOCAB)
    rng = random.Random(12345)
    rng.shuffle(vocab)
    out: list[str] = []
    while len(out) < sample_words:
        out.extend(vocab)
    sample = " ".join(out[:sample_words])
    try:
        resp = await http_client.post(
            f"{base_url.rstrip('/')}/tokenize",
            json={"model": model, "prompt": sample},
            timeout=30.0,
        )
        if resp.status_code != 200:
            _ckpt(
                f"calibrate: /tokenize http {resp.status_code}; "
                f"keeping tokens/word={_TOKENS_PER_WORD:.4f}"
            )
            return
        body = resp.json()
        ntok = int(body.get("count") or len(body.get("tokens") or []))
        if ntok <= 0:
            _ckpt("calibrate: /tokenize returned 0 tokens; keeping default")
            return
        _TOKENS_PER_WORD = ntok / sample_words
        _ckpt(
            f"calibrate: tokens/word={_TOKENS_PER_WORD:.4f} "
            f"({ntok} tokens / {sample_words} words) — "
            f"prompts will hit requested token counts"
        )
    except Exception as exc:  # noqa: BLE001
        _ckpt(
            f"calibrate: /tokenize failed ({type(exc).__name__}: {exc}); "
            f"keeping tokens/word={_TOKENS_PER_WORD:.4f}"
        )


_SHARED_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. Sphinx of black "
    "quartz, judge my vow. Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. The five boxing wizards "
    "jump quickly. Crazy Fredrick bought many very exquisite opal "
    "jewels. Jaded zombies acted quaintly but kept driving their oxen "
    "forward. "
)


# Larger vocabulary for per-client prefixes — real content diversity
# beyond the 50-word pangram, so each client's long prefix isn't just one
# short paragraph reshuffled. Keeps real English words so the
# tokens≈words/1.4 estimate stays stable.
_PER_CLIENT_VOCAB = (
    _SHARED_PARAGRAPH + "system user assistant context document session token cache memory "
    "vector matrix tensor kernel buffer stream latency bandwidth storage "
    "network protocol transfer offload prefix suffix sequence batch "
    "concurrency throughput pipeline scatter gather register allocate "
    "evict retain compute attention layer model engine worker request "
    "response payload header metadata checksum compress encode decode "
    "serialize partition replicate shard cluster node leader follower "
    "quorum consensus heartbeat timeout retry backoff garden mountain "
    "river forest desert ocean valley canyon glacier meadow copper silver "
    "golden iron bronze marble granite crystal amber pearl swift gentle "
    "fierce calm bright ancient modern hollow solid whisper thunder ripple "
    "cascade drift soar plunge wander linger vanish beacon harbor lantern "
    "compass voyage summit ember frost willow cedar maple birch"
).split()


def _build_shared_prefix(n_tokens: int) -> str:
    """Build a deterministic shared prefix targeting roughly n_tokens
    tokens. Identical bytes across all clients → cache-friendly."""
    word_target = _approx_words_for_tokens(n_tokens)
    out_words: list[str] = []
    para = _SHARED_PARAGRAPH.split()
    while len(out_words) < word_target:
        out_words.extend(para)
    return _fit_to_tokens(" ".join(out_words[:word_target]), n_tokens)


def _build_unique_suffix(client_id: int, run_id: int, n_tokens: int) -> str:
    """Build a deterministic-per-(client_id, run_id) unique suffix
    targeting roughly n_tokens tokens. Different for every request so
    the suffix DOESN'T hit cache (only the shared prefix should)."""
    word_target = _approx_words_for_tokens(n_tokens)
    rng = random.Random(client_id * 1_000_003 + run_id)
    # Use a deterministic vocabulary so prompts are stable across runs
    # of the same (client_id, run_id) — useful for reproducing results.
    vocab = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "yankee",
        "zulu",
    ]
    words = [rng.choice(vocab) for _ in range(word_target)]
    text = f"Client {client_id} run {run_id}: " + " ".join(words)
    return _fit_to_tokens(text, n_tokens)


def _build_per_client_prefix(client_id: int, n_tokens: int) -> str:
    """Build a per-client-but-stable prefix targeting roughly n_tokens
    tokens. Each `client_id` gets a DIFFERENT deterministic prefix,
    but the prefix is STABLE across runs of the same client_id so
    warmup + measure hit the same cache entry.

    Use case: multi-tenant cliff bench. Each "client" represents a
    user session with its own system prompt; the cliff appears when
    aggregate session-prefix bytes exceed VRAM cache budget and the
    engine starts evicting + re-prefilling per request.
    """
    word_target = _approx_words_for_tokens(n_tokens)
    rng = random.Random(client_id * 7919 + 13)
    vocab = list(_PER_CLIENT_VOCAB)
    # Stable-per-client prefix; deterministic shuffle of a large vocab.
    out: list[str] = []
    while len(out) < word_target:
        rng.shuffle(vocab)
        out.extend(vocab)
    body = " ".join(out[:word_target])
    # UNIQUE ANCHOR at the very front: guarantees the FIRST prefix-cache
    # block differs across clients regardless of how the vocab shuffles,
    # so there's no accidental cross-client prefix-cache (L1) sharing.
    anchor = f"{client_id}-{client_id * 99991 + 7}-session:"
    return _fit_to_tokens(anchor + " " + body, n_tokens)


def build_prompt(
    client_id: int,
    run_id: int,
    shared_prefix_tokens: int,
    unique_suffix_tokens: int,
    prefix_mode: str = "shared",
) -> str:
    """Compose prompt for a given (client_id, run_id).

    ``prefix_mode``:
      - ``"shared"`` (default): all clients share one identical
        prefix → vLLM's VRAM prefix cache hits trivially → no cliff
        in the tested range (verified on an MI355X node TP=1
        gpt-oss-120b: throughput climbs to 304K tok/s at c=250 with
        no inflection).
      - ``"per_client"``: each client_id gets its own DETERMINISTIC
        prefix (stable across runs so warmup → measure shares cache).
        K unique sessions → cliff appears when aggregate
        session-prefix KV exceeds VRAM budget. This mirrors the
        LMCache slide's "warm cache, multi-tenant" pattern.
    """
    if prefix_mode == "per_client":
        prefix = _build_per_client_prefix(client_id, shared_prefix_tokens)
    else:
        prefix = _build_shared_prefix(shared_prefix_tokens)
    suffix = _build_unique_suffix(client_id, run_id, unique_suffix_tokens)
    return prefix + "\n\nQ: " + suffix + "\nA: "


# ---------------------------------------------------------------------
# Client — emits prefill requests
# ---------------------------------------------------------------------


@dataclass
class ReqResult:
    client_id: int
    run_id: int
    issued_at: float
    finished_at: float
    prompt_chars: int
    prompt_tokens_reported: int  # from server response if available
    output_tokens: int
    error: str | None


async def _fire_one(
    http_client,
    base_url: str,
    model: str,
    prompt: str,
    client_id: int,
    run_id: int,
    max_tokens: int,
    request_timeout: float,
) -> ReqResult:
    issued = time.perf_counter()
    err: str | None = None
    prompt_tokens = 0
    out_tokens = 0
    try:
        resp = await http_client.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=request_timeout,
        )
        if resp.status_code != 200:
            err = f"http {resp.status_code}: {resp.text[:200]}"
        else:
            body = resp.json()
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            out_tokens = int(usage.get("completion_tokens") or 0)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    return ReqResult(
        client_id=client_id,
        run_id=run_id,
        issued_at=issued,
        finished_at=time.perf_counter(),
        prompt_chars=len(prompt),
        prompt_tokens_reported=prompt_tokens,
        output_tokens=out_tokens,
        error=err,
    )


async def _run_one_concurrency(
    http_client,
    base_url: str,
    model: str,
    concurrency: int,
    run_id: int,
    isl: int,
    shared_prefix_tokens: int,
    max_tokens: int,
    request_timeout: float,
    prefix_mode: str = "shared",
) -> tuple[float, list[ReqResult]]:
    """Fire `concurrency` concurrent requests, wait for all, return
    (wall_seconds, [results])."""
    unique_suffix_tokens = isl - shared_prefix_tokens
    prompts = [
        build_prompt(
            client_id=i,
            run_id=run_id,
            shared_prefix_tokens=shared_prefix_tokens,
            unique_suffix_tokens=unique_suffix_tokens,
            prefix_mode=prefix_mode,
        )
        for i in range(concurrency)
    ]
    t0 = time.perf_counter()
    coros = [
        _fire_one(
            http_client,
            base_url,
            model,
            prompts[i],
            client_id=i,
            run_id=run_id,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        for i in range(concurrency)
    ]
    results = await asyncio.gather(*coros)
    wall = time.perf_counter() - t0
    return wall, results


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


async def amain(args: argparse.Namespace) -> None:
    try:
        import httpx
    except ImportError:
        _ckpt("FAIL: httpx not installed; pip install httpx")
        sys.exit(1)

    concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    if not concurrencies:
        _ckpt("FAIL: --concurrencies empty")
        sys.exit(1)

    # Load the tokenizer so "N tokens" means exactly N tokens. Default
    # to the served model path; --tokenizer overrides. Degrades to the
    # word-ratio approximation if transformers/tokenizer unavailable.
    set_active_tokenizer(args.tokenizer or args.model)

    _ckpt(f"endpoint: {args.endpoint}")
    _ckpt(f"model: {args.model}")
    _ckpt(f"arm: {args.arm}")
    _ckpt(
        f"isl={args.isl} (prefix={args.shared_prefix_tokens} + "
        f"unique={args.isl - args.shared_prefix_tokens})  max_tokens={args.max_tokens}"
    )
    _ckpt(f"concurrencies: {concurrencies}")
    _ckpt(f"iters={args.iters} warmup_iters={args.warmup_iters}")

    scrape_metrics = not args.no_metrics
    metrics_base = (args.metrics_endpoint or args.endpoint).rstrip("/")
    metrics_url = f"{metrics_base}/metrics"
    if scrape_metrics:
        _ckpt(f"metrics: scraping prefix-cache hit rate from {metrics_url} (per timed iter)")
    else:
        _ckpt("metrics: disabled (--no-metrics); hit-rate columns blank")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Append-mode if user passes --append; default truncate.
    mode = "a" if args.append else "w"
    write_header = not (args.append and out_path.exists())

    async with httpx.AsyncClient(timeout=args.request_timeout) as http_client:
        # Ping the endpoint so we fail fast if it's not up.
        try:
            r = await http_client.get(f"{args.endpoint.rstrip('/')}/v1/models", timeout=10.0)
            if r.status_code != 200:
                _ckpt(
                    f"WARN: /v1/models returned {r.status_code}; "
                    f"continuing anyway — endpoint may still serve /v1/completions"
                )
        except Exception as exc:
            _ckpt(f"FAIL: cannot reach endpoint {args.endpoint}: {exc}")
            sys.exit(1)

        # Calibrate words→tokens against the live server so "N tokens"
        # actually means N tokens (no client-side tokenizer required).
        await calibrate_word_ratio(http_client, args.endpoint, args.model)

        # Warmup at the lowest concurrency so kvd hits the shared
        # prefix on its first eviction-and-reload cycle. For
        # per_client mode, we MUST warm at the same concurrency we'll
        # measure so all per-client prefixes get cached first; pass
        # `--warmup-at-each-c` to do that automatically.
        if args.warmup_iters > 0 and not args.warmup_at_each_c:
            warm_c = max(1, concurrencies[0])
            _ckpt(f"--- WARMUP {args.warmup_iters} iter(s) at c={warm_c} ---")
            for w in range(args.warmup_iters):
                wall, results = await _run_one_concurrency(
                    http_client,
                    args.endpoint,
                    args.model,
                    concurrency=warm_c,
                    run_id=10_000 + w,
                    isl=args.isl,
                    shared_prefix_tokens=args.shared_prefix_tokens,
                    max_tokens=args.max_tokens,
                    request_timeout=args.request_timeout,
                    prefix_mode=args.prefix_mode,
                )
                errs = [r for r in results if r.error]
                _ckpt(f"  warmup {w}: wall={wall:.2f}s  errors={len(errs)}")
                if errs:
                    _ckpt(f"  first error: {errs[0].error}")

        with out_path.open(mode, newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(
                    [
                        "arm",
                        "concurrency",
                        "iter",
                        "wall_s",
                        "ok_count",
                        "err_count",
                        "total_prompt_tokens",
                        "total_output_tokens",
                        "throughput_tok_s_total",
                        "throughput_tok_s_prompt",
                        "p50_latency_s",
                        "p95_latency_s",
                        # per-iter prefix-cache hit rates from /metrics (blank
                        # when --no-metrics or the scrape fails):
                        "l1_cache_queries",
                        "l1_cache_hits",
                        "l1_hit_rate_pct",
                        "ext_cache_queries",
                        "ext_cache_hits",
                        "ext_hit_rate_pct",
                    ]
                )

            for c in concurrencies:
                # Per-c warmup: send the SAME C clients once to prime
                # any caches (VRAM prefix cache for Arm A, kvd file
                # tier for Arm B). The measurement runs that follow
                # then see warm cache, mirroring the LMCache slide's
                # "ideal conditions, cache is warm" footnote.
                if args.warmup_at_each_c:
                    wall, results = await _run_one_concurrency(
                        http_client,
                        args.endpoint,
                        args.model,
                        concurrency=c,
                        run_id=20_000 + c,
                        isl=args.isl,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                        prefix_mode=args.prefix_mode,
                    )
                    errs = [r for r in results if r.error]
                    _ckpt(f"  c={c} per-c warmup: wall={wall:.2f}s errors={len(errs)}")
                    # Async-save connectors (kvd v2 chunked-fusion)
                    # return from wait_for_save before chunks land on
                    # disk; back-pressure from the warmup's tail of
                    # in-flight writes will contaminate iter 0's
                    # measurement. Sleep here to let the queue drain.
                    # See project_lmcache_mi300x_agentic_bench_2026_05
                    # for the comparison: LMCache doesn't have this
                    # either; they bench sustained load instead.
                    if args.post_warmup_sleep_s > 0:
                        _ckpt(
                            f"  c={c} sleeping {args.post_warmup_sleep_s}s "
                            f"for async save queue drain"
                        )
                        await asyncio.sleep(args.post_warmup_sleep_s)
                _ckpt(f"--- concurrency c={c} ({args.iters} iter(s)) ---")
                point_throughputs: list[float] = []
                for it in range(args.iters):
                    snap0 = await _snap_cache(http_client, metrics_url) if scrape_metrics else None
                    wall, results = await _run_one_concurrency(
                        http_client,
                        args.endpoint,
                        args.model,
                        concurrency=c,
                        run_id=(c * 100 + it),
                        isl=args.isl,
                        shared_prefix_tokens=args.shared_prefix_tokens,
                        max_tokens=args.max_tokens,
                        request_timeout=args.request_timeout,
                        prefix_mode=args.prefix_mode,
                    )
                    snap1 = await _snap_cache(http_client, metrics_url) if scrape_metrics else None
                    l1q, l1h, l1r = _window_rate(snap0, snap1, "l1_q", "l1_h")
                    exq, exh, exr = _window_rate(snap0, snap1, "ext_q", "ext_h")
                    ok = [r for r in results if r.error is None]
                    errs = [r for r in results if r.error is not None]
                    total_pt = sum(r.prompt_tokens_reported for r in ok)
                    total_ot = sum(r.output_tokens for r in ok)
                    if total_pt == 0:
                        # Server didn't report usage tokens — fall
                        # back to a char-based estimate (rough but
                        # comparable across arms since prompts are
                        # identical).
                        total_pt = sum(r.prompt_chars for r in ok) // 4
                    throughput_total = (total_pt + total_ot) / wall if wall > 0 else 0
                    throughput_prompt = total_pt / wall if wall > 0 else 0
                    latencies = [r.finished_at - r.issued_at for r in ok]
                    p50 = statistics.median(latencies) if latencies else 0.0
                    p95 = (
                        statistics.quantiles(latencies, n=20)[-1]
                        if len(latencies) >= 5
                        else (max(latencies) if latencies else 0.0)
                    )
                    writer.writerow(
                        [
                            args.arm,
                            c,
                            it,
                            f"{wall:.3f}",
                            len(ok),
                            len(errs),
                            total_pt,
                            total_ot,
                            f"{throughput_total:.1f}",
                            f"{throughput_prompt:.1f}",
                            f"{p50:.3f}",
                            f"{p95:.3f}",
                            l1q,
                            l1h,
                            l1r,
                            exq,
                            exh,
                            exr,
                        ]
                    )
                    fh.flush()
                    point_throughputs.append(throughput_total)
                    hit_str = ""
                    if scrape_metrics:
                        hit_str = f"  L1_hit={l1r or 'n/a'}%  ext_hit={exr or 'n/a'}%"
                    _ckpt(
                        f"  it={it}  wall={wall:.2f}s  ok={len(ok)}  err={len(errs)}  "
                        f"BW={throughput_total:7.0f} tok/s  p50={p50 * 1000:.0f}ms  p95={p95 * 1000:.0f}ms"
                        f"{hit_str}"
                    )
                    if errs and it == 0:
                        _ckpt(f"    first err: {errs[0].error}")
                med = statistics.median(point_throughputs)
                _ckpt(f"  median throughput at c={c}: {med:.0f} tok/s")

    _ckpt(f"done; results at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", required=True, help="vLLM OpenAI-compat endpoint, e.g. http://localhost:8801"
    )
    parser.add_argument(
        "--model", required=True, help="--model arg vLLM was launched with (path or HF id)"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer path/HF id for exact token-count "
        "prompts (default: --model). If it can't load, "
        "falls back to a word-ratio approximation.",
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=["vram_only", "vram_dram", "kvd_v2"],
        help="arm label written to the CSV (drives the plot legend)",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=20000,
        help="total input sequence length per request (default 20000)",
    )
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=18000,
        help="how many of the ISL tokens are the shared prefix "
        "(rest is per-client unique suffix; default 18000)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1, help="OSL (default 1; prefill-only measurement)"
    )
    parser.add_argument(
        "--concurrencies",
        default="1,2,4,8,16,32,48,64,80,100,128,160,200,250",
        help="comma-separated list of concurrency levels",
    )
    parser.add_argument(
        "--iters", type=int, default=3, help="iterations per concurrency level (median reported)"
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="warmup iterations at the lowest concurrency (excluded from CSV)",
    )
    parser.add_argument(
        "--warmup-at-each-c",
        action="store_true",
        help="warm cache at EACH concurrency level (one iter, "
        "before timed iters); required for per_client mode "
        "so all C unique prefixes get cached first",
    )
    parser.add_argument(
        "--post-warmup-sleep-s",
        type=float,
        default=0.0,
        help="seconds to sleep after each per-c warmup before "
        "starting timed iters. For async-save connectors "
        "(e.g. kvd v2 chunked-fusion) this lets the save "
        "queue drain so iter 0 isn't contaminated by "
        "in-flight writes from the warmup tail. 5-10 s "
        "is plenty for Kimi K2.5 c<=32 workloads.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=["shared", "per_client"],
        default="shared",
        help="shared: all clients share one prefix (cache-hit "
        "trivial); per_client: each client gets its own "
        "stable prefix (mirrors LMCache multi-tenant cliff)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=600.0, help="per-request timeout in seconds"
    )
    parser.add_argument(
        "--metrics-endpoint",
        default=None,
        help="endpoint to scrape Prometheus /metrics from "
        "(default: --endpoint). Captures L1 (GPU) + "
        "external (kvd/L3) prefix-cache hit rate per "
        "timed iter into the CSV.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="disable /metrics scraping; the six hit-rate columns are still written but left blank",
    )
    parser.add_argument("--out", required=True, help="CSV output path; created if missing")
    parser.add_argument(
        "--append", action="store_true", help="append to existing CSV instead of truncate"
    )
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
