###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SLA-based planner: decide how many prefill/decode replicas meet TTFT/ITL.

Two processes, run in order.

``python -m infera.planner.profile`` is the offline step: before the fleet
takes traffic, it sweeps one prefill replica and one decode replica and writes
``profile.json`` -- how TTFT and prefill throughput grow with prompt length,
and how ITL and decode throughput degrade as the KV cache fills.

``python -m infera.planner`` is the online step. It runs outside the request
path and, every adjustment interval:

  1. scrapes the server fleet's ``/metrics`` for the observed TTFT, ITL, ISL,
     OSL and request rate (:mod:`infera.planner.metrics_source`);
  2. compares them against what the profile predicted, to derive correction
     factors that absorb queueing and prefix-cache effects;
  3. assumes the next interval repeats the observed load and solves for the
     prefill/decode replica counts that meet the SLA
     (:mod:`infera.planner.core`);
  4. logs the decision.

Nothing is resized yet. ``SlaPlanner`` accepts an optional decision callback so
an actuator can be added later without changing the sizing code.
"""
