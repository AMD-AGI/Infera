"""Compatibility for AIPerf 0.12 offline workers using local tokenizers.

Verbatim from examples/sglang_1p1d_glm5.2/engine/aiperf_compat/sitecustomize.py.
AIPerf resolves --tokenizer through the Hub snapshot path even when the value is
a directory that already exists, so an offline run fails on a tokenizer it is
looking straight at. Reached via PYTHONPATH so it loads before aiperf's CLI.
"""

from pathlib import Path

from aiperf.common.tokenizer import Tokenizer  # type: ignore[import-not-found]

_original_resolve_local_snapshot = Tokenizer._resolve_local_snapshot.__func__


@classmethod
def _resolve_local_snapshot(cls, name: str, revision: str) -> str:
    local_path = Path(name)
    if local_path.is_dir():
        return str(local_path.resolve())
    return _original_resolve_local_snapshot(cls, name, revision)


Tokenizer._resolve_local_snapshot = _resolve_local_snapshot
