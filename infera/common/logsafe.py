###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Helpers for safely logging untrusted (client-supplied) values.

CodeQL's ``py/log-injection`` flags any flow from request data (headers,
body fields, query params) into a log record: a value containing CR/LF can
forge additional, attacker-controlled log lines. ``scrub`` coerces a value to
``str`` and escapes CR/LF/tab so a single logged value can never span more
than one line, and bounds its length so a client can't flood the log.
"""

from __future__ import annotations

_MAX_LEN = 256


def scrub(value: object, *, max_len: int = _MAX_LEN) -> str:
    """Return a single-line, length-bounded string for logging ``value``.

    Escapes backslash, CR, LF and tab so untrusted input can't inject new
    log lines, then truncates to ``max_len`` characters.
    """
    s = (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    if len(s) > max_len:
        s = s[:max_len] + "...(truncated)"
    return s
