#!/usr/bin/env python3
"""Add cache-tier fields/events to SGLang's existing request-stage tracing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/sgl-workspace/sglang/python/sglang")
SCHEDULE_BATCH = ROOT / "srt/managers/schedule_batch.py"
MARKER = "INFERA_C80_C112_REQUEST_TRACE_V1"


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{description}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = SCHEDULE_BATCH.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{SCHEDULE_BATCH}: trace patch already present")
        return 0

    text = replace_once(
        text,
        '''            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
''',
        f'''            f"cached_input_len={{self.cached_tokens}}, "
            f"cached_device={{self.cached_tokens_device}}, "
            f"cached_host={{self.cached_tokens_host}}, "
            f"cached_storage={{self.cached_tokens_storage}}, "
            f"routed_dp_rank={{self.disagg_prefill_dp_rank}}, "
            f"output_len={{len(self.output_ids)}}, "
            # {MARKER}
''',
        "request time stats cache-tier fields",
    )

    text = replace_once(
        text,
        '''                    req._cache_breakdown_computed = True

                req.already_computed = seq_len
''',
        f'''                    req._cache_breakdown_computed = True
                    req.time_stats.trace_ctx.trace_event(
                        "prefill_cache_lookup",
                        1,
                        attrs={{
                            "cache.device_tokens": int(req.cached_tokens_device),
                            "cache.host_tokens": int(req.cached_tokens_host),
                            "cache.storage_tokens": int(req.cached_tokens_storage),
                            "cache.miss_tokens": int(
                                max(
                                    0,
                                    len(req.origin_input_ids)
                                    - req.cached_tokens_device
                                    - req.cached_tokens_host
                                    - req.cached_tokens_storage,
                                )
                            ),
                            "routing.prefill_dp_rank": int(
                                req.disagg_prefill_dp_rank
                                if req.disagg_prefill_dp_rank is not None
                                else -1
                            ),
                        }},
                    )
                    # {MARKER}

                req.already_computed = seq_len
''',
        "per-request cache lookup trace event",
    )

    SCHEDULE_BATCH.write_text(text, encoding="utf-8")
    print(f"patched {SCHEDULE_BATCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
