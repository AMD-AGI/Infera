#!/bin/bash
# Keep resubmitting an 8-GPU sleeper until one lands on a good node and STAYS
# running (survives a spur exec probe). idle nodes currently NODE_FAIL GPU
# launches → JobHoldMaxRequeue; a fresh submit gets priority 1000 and a new node
# draw, so we spam like the forge system does until one sticks.
set -u
HOLD=/home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/hold_node.sh
MAXTRIES="${1:-40}"
for attempt in $(seq 1 "$MAXTRIES"); do
  J=$(sbatch --parsable -A amd-primus -p amd-spur -N1 -G8 -t 08:00:00 "$HOLD" 2>/dev/null)
  [ -z "$J" ] && { sleep 2; continue; }
  # give it up to ~25s to either stick or get held
  verdict="pending"
  for i in $(seq 1 5); do
    R=$(scontrol show job "$J" 2>/dev/null | grep -oE "Reason=[A-Za-z]+" | head -1)
    ST=$(scontrol show job "$J" 2>/dev/null | grep -oE "JobState=[A-Z]+" | head -1)
    if [ "$R" = "Reason=JobHoldMaxRequeue" ]; then verdict="held"; break; fi
    if [ "$ST" = "JobState=RUNNING" ]; then verdict="running"; break; fi
    sleep 5
  done
  if [ "$verdict" = "running" ]; then
    sleep 6
    if spur exec "$J" true 2>/dev/null; then
      NODE=$(squeue -j "$J" -h -o "%N" 2>/dev/null)
      echo "STABLE job=$J node=$NODE (attempt $attempt)"
      echo "$J" > /home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round0_image/held_jobid.txt
      echo "$NODE" > /home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round0_image/held_node.txt
      exit 0
    fi
  fi
  echo "attempt $attempt: job=$J -> $verdict (dropping)"
  scancel "$J" 2>/dev/null
  sleep 1
done
echo "NO STABLE NODE after $MAXTRIES tries"
exit 1
