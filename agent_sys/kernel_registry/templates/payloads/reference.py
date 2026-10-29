# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Naive mathematical reference: elementwise x squared plus one."""


def run(values):
    return [value**2 + 1 for value in values]
