#!/usr/bin/env bash
# Login-node watcher: when 88-require-6.16 asked for a new holder (NEED_RESUBMIT) and the old holder
# has exited, submit a new one with the current exclude list. Exits once the queue is done (99-stop ran).
H=/apps/liyingli/holder-infx
EX=/apps/liyingli/jobs/exclude_nodes.txt
while true; do
    if [ -e "$H/done/99-stop.rc" ] && [ ! -e "$H/NEED_RESUBMIT" ]; then echo "$(date -u +%FT%TZ) queue done"; exit 0; fi
    if [ -e "$H/NEED_RESUBMIT" ] && ! squeue -h -u "$USER" -n llying-infx4629 | grep -q .; then
        rm -f "$H/NEED_RESUBMIT" "$H/STOP"
        echo "$(date -u +%FT%TZ) resubmit: $(sbatch --exclude="$(paste -sd, "$EX")" "$H/holder.sh")"
    fi
    sleep 60
done
