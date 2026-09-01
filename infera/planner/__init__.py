###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Capacity planning for Infera prefill/decode deployments.

``python -m infera.planner.profile`` measures a model deployment and writes its
capacity envelope. ``python -m infera.planner`` combines that envelope with
windowed server observations and reports target pool sizes. Planning runs
outside the request path and does not resize a deployment by itself.
"""
