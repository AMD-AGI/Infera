#!/usr/bin/env bash
# Purpose: Diagnostic reference arm: B-perf unchanged except a 900 s AgentX
#   profiling phase, for py-spy windows comparable with C-diag.
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
AGENTX_DURATION=900
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.decrad.b-perf.sh"
