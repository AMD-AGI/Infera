###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_tokenizer_path(path: str) -> str:
    """Return a local path, downloading tokenizer files if given an HF id."""
    if os.path.isfile(path) or os.path.isdir(path):
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError(f"{path!r} is not a local path and huggingface_hub is missing") from exc
    try:
        resolved = snapshot_download(repo_id=path, allow_patterns=["tokenizer*", "chat_template*"])
    except Exception as exc:
        raise ValueError(f"{path!r}: not a local path and HF resolution failed: {exc}") from exc
    logger.info("resolved HF id %r → %s", path, resolved)
    return resolved
