###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Public HTTP API surfaces.

Currently:

- `anthropic` — Anthropic Messages API (`/v1/messages`) shim that
  translates incoming requests into the engine-friendly OpenAI Chat
  shape, then translates responses back to Anthropic SSE / JSON.
"""
