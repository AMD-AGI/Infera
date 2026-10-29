# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Illustrative candidate; no performance improvement has been measured."""


def run(values):
    return [value * value + 1 for value in values]
