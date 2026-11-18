#!/usr/bin/env bash
# B2: P4D4 scoring with vLLM's TieringOffloadingSpec (CPU primary tier in /dev/shm + FS secondary tier on the local
# NVMe xfs volume) and the PP fix bind-mounted ($R/overlay/vllm/**); otherwise as b1_p4d4.sh (prefill GEM at P_UTIL
# unless P_GEM=0). The FS tier gets a fresh directory per run and it is deleted afterwards (AgentX replays the same
# prompts, so leftover chunks would inflate hits).
#   b2_p4d4.sh TAG [phases]   (phases as b1_p4d4.sh; default: score)
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
cd "$R" || exit 1
tag=${1:?tag}; shift
phases=${*:-score}
CPU=${CPU:-1250000000000}
KVFS=/mnt/m2m_nobackup/liyingli_vpd/kvfs-trackB/p4d4-$tag
V=/usr/local/lib/python3.12/dist-packages/vllm
mounts=""
for f in $(cd overlay/vllm && find . -name '*.py' | sed 's#^\./##'); do mounts+=" $R/overlay/vllm/$f:$V/$f:ro"; done
rm -rf "$KVFS"; mkdir -p "$KVFS"; chmod 777 "$KVFS"
mkdir -p runs/b2-$tag
( for i in $(seq 900); do echo "$(date -u +%T) $(df -B1 --output=used,avail "$KVFS" | tail -1) $(du -sb "$KVFS" 2>/dev/null | cut -f1)"
      [ -e runs/b2-$tag/kvfs.stop ] && break; sleep 30; done > runs/b2-$tag/kvfs-df.log ) &
tiers="[{\"type\":\"fs\",\"root_dir\":\"/kvfs\",\"n_read_threads\":${FS_READ_THREADS:-32},\"n_write_threads\":${FS_WRITE_THREADS:-16}}]"
OFF_EXTRA="{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":$CPU,\"eviction_policy\":\"lru\",\"secondary_tiers\":$tiers}" \
P_MOUNT_ADD="$mounts $KVFS:/kvfs" RUN_PREFIX=b2 CPU=$CPU bash scripts/p4d4_run.sh "$tag" $phases
touch runs/b2-$tag/kvfs.stop
{ du -sb "$KVFS"; find "$KVFS" -type f | wc -l; } > runs/b2-$tag/kvfs-du.txt 2>&1
# The engine container writes the FS tier as root.
docker run --rm -v "$(dirname "$KVFS"):/k" --entrypoint bash infera-vllm:v0340-0c5e5e89 -c "ionice -c3 rm -rf /k/$(basename "$KVFS")"
echo "$(date -u +%T) b2-$tag: FS tier $(head -1 runs/b2-$tag/kvfs-du.txt | cut -f1) bytes in $(tail -1 runs/b2-$tag/kvfs-du.txt) files, removed" | tee -a logs/summary.out
exit 0
