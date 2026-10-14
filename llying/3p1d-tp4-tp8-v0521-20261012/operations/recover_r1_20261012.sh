#!/usr/bin/env bash
# r1 recovery (2026-10-12 ~18:45 UTC): launch.sh died ("Stale file handle" after an NFS
# script replace) once the three prefills were healthy, before the router; the manually
# started decode-0 lost ranks to a full 8g TMPDIR tmpfs. The prefills are kept. This
# restarts decode-0 (real acceptance, TMPDIR now 32g), starts the router, and resumes
# the pipeline at the known-answer gate. Run detached on crsuse2-m2m-137.
set -uo pipefail
R=/home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012
RUNS="$R/runtime/runs/r1"
since="$(date -u +%Y-%m-%dT%H:%M:%S)"
printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" recover "restart decode-0 (real acceptance, tmpfs 32g) + router" >> "$RUNS/status.tsv"
bash "$R/scripts/restart_decode.sh" ROUND=r1 SIM= "LAUNCH_DIR=$RUNS/launch" TAG=tmpfsfail \
    > "$RUNS/recover.log" 2>&1 || {
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" recover "FAIL rc=$? (see recover.log)" >> "$RUNS/status.tsv"
    exit 1
}
printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" recover ok >> "$RUNS/status.tsv"
ROUND=r1 POINTS="144 160 176" CENSUS_SINCE="$since" bash "$R/scripts/run_3p1d.sh" after-up
