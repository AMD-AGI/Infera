#!/usr/bin/env bash
# Run a command on the node of job 295840 (v3-240), from the login node: onnode.sh <command...>
# spur allows one exec-into-job session per node, so once runner.sh holds the session, use q.sh instead.
# The remote command always exits 0 (any non-zero step exit ends a spur job); its real status is printed as [rc=N].
set -uo pipefail
job=${JOB:-295840}
node=${NODE:-$(squeue -j "$job" -h -o "%N")}
[ -n "$node" ] || { echo "job $job is not running" >&2; exit 1; }
remote="export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER; cd /tmp; ( $* ); echo \"[rc=\$?]\"; exit 0"
srun --jobid="$job" --overlap -D /tmp -N1 -n1 -w "$node" --gres=gpu:8 --export=NONE /bin/bash -c "$remote" </dev/null 2>&1 \
    | tr -d '\r' | grep --line-buffered -v -e '^srun: warning: --pty runs' -e '^srun: warning: a --pty step' \
        -e '^spur: warning: raw mode unavailable' -e '^srun: dispatched'
