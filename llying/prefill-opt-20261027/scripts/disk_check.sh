#!/usr/bin/env bash
# Disk check on the prefill node (local) and the decode node (DECODE_SSH): /, the docker / data disk and the shared
# /home, plus docker's own accounting. Prints one block per node; exits 1 if / or /mnt/m2m_nobackup is >= LIMIT%
# (default 95) on either node. /home is shared NFS (already ~95% on 10-30): logged, never a stop condition.
# 10-30: crsuse2-m2m-140 started refusing ssh (publickey); a full disk is one suspected cause.
# Usage: disk_check.sh DECODE_SSH [LIMIT=95] [TAG]
set -uo pipefail
dec="$1" limit="${2:-95}" tag="${3:-}"
bad=0
for n in localhost "$dec"; do
    cmd='df -P / /mnt/m2m_nobackup /home 2>/dev/null | awk "NR>1 {printf \"%s %s%% %dG-free\n\", \$6, \$5+0, \$4/1048576}"; docker system df --format "{{.Type}} {{.Size}} reclaimable {{.Reclaimable}}" 2>/dev/null'
    if [[ "$n" == localhost ]]; then out="$(bash -c "$cmd")"; else out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$n" "$cmd" 2>&1)"; fi
    echo "== disk $(date -u +%T) $tag $(hostname -s 2>/dev/null)${n/localhost/} ==" | sed "s/ $(hostname -s)crsuse/ crsuse/"
    echo "$out"
    if ! grep -q "^/ " <<<"$out"; then echo "disk_check: cannot read $n"; bad=1; continue; fi
    while read -r mnt pct _; do
        p="${pct%\%}"
        [[ "$mnt" == / || "$mnt" == /mnt/m2m_nobackup ]] && ((p >= limit)) && { echo "disk_check: $n $mnt at $pct >= $limit%"; bad=1; }
    done < <(grep -E "^/(mnt/m2m_nobackup)? " <<<"$out")
done
exit $bad
