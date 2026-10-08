#!/usr/bin/env bash
# Diagnostic run: the v3 fix run with the RC ACK timeout raised to 24 (69 s) on
# both legs, to tell late ACKs from a NIC timer that fires regardless of it.
source "$(dirname "${BASH_SOURCE[0]}")/config.rca-fix.p8d8.sh"
PREFILL_EXTRA_ENV="MC_IB_TIMEOUT=24"
DECODE_EXTRA_ENV="MC_IB_TIMEOUT=24"
