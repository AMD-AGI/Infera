#!/usr/bin/env bash
# Purpose: Diagnostic arm: C-perf unchanged except a 900 s AgentX profiling
#   phase, long enough for py-spy windows on the decode schedulers.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
AGENTX_DURATION=900
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.c-perf.sh"
