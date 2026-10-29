# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Original CPU implementation for the catalog example."""


def run(values):
    result = []
    for value in values:
        result.append(value * value + 1)
    return result
