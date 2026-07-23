#!/bin/bash
# dump last N non-server_args lines, CR-stripped, truncated. Args: <logfile> [nlines]
L="$1"; N="${2:-40}"
grep -av 'server_args=' "$L" | tr '\r' '\n' | cut -c1-200 | tail -"$N"
