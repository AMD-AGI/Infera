###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.nats_bus import js_store_failed


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('nats: 503 err_code=10077 error opening msg block file [""]', True),
        ("JSStreamStoreFailedF", True),
        ("error opening msg block file", True),
        ("timeout talking to nats", False),
    ],
)
def test_js_store_failed_detects_filestore_errors(text, expected):
    """10077 is a broken JetStream FILE store, not a missing worker."""
    assert js_store_failed(RuntimeError(text)) is expected
