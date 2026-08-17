###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SLA-based planner: size the prefill/decode pools to meet TTFT/ITL targets.

The planner runs as its own process (``python -m infera.planner``), outside the
request path. Every adjustment interval it:

  1. scrapes the server fleet's ``/metrics`` for the observed TTFT, ITL, ISL,
     OSL and request rate (:mod:`infera.planner.metrics_source`);
  2. compares them against what pre-deployment profiling predicted, to derive
     correction factors that absorb queueing and prefix-cache effects;
  3. forecasts the next interval's load (:mod:`infera.planner.predictor`);
  4. solves for the prefill/decode replica counts that meet the SLA
     (:mod:`infera.planner.core`);
  5. hands the decision to a connector (:mod:`infera.planner.connectors`),
     which either patches an ``InferaDeployment`` or publishes it to etcd.

The scaling algorithm is adapted from NVIDIA Dynamo's SLA planner
(Apache-2.0); this is an independent implementation.
"""
