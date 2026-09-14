#!/bin/bash
# Find a node with a free GPU half, without taking a hold to find out.
#
#   ./nodeprobe.sh --auto              every idle/mix node sinfo lists
#   ./nodeprobe.sh crsuse2-m2m-057 …   named nodes
#   ./nodeprobe.sh --auto --need 4     how many free cards counts as a candidate
#
# ---------------------------------------------------------------------------
# The three facts that decide a node, and the fourth that vetoes it
#
# **No node has yet failed for the same reason twice**, which is the whole
# argument for a single-command verdict over a habit of checking one thing:
#
#   1. **which cards are actually free** — `rocm-smi --showmemuse`. `243` had
#      the right base and not one free card.
#   2. **can `Dockerfile.sglang` build against a base that is here** — m1's
#      anchor: a line following `background=request.background,` that reads
#      `require_reasoning=require_reasoning,` means yes, a bare `)` means no.
#      Present in `v0.5.17-rocm720`, absent in 0.5.12/0.5.14. `249` had free
#      cards and the wrong image family.
#   3. **is there disk, on BOTH filesystems** — `/mnt/m2m_nobackup` holds the
#      image store, `/` is where docker builds, and they are different numbers.
#      `186` had the right base, free-looking Slurm state, and **3.4 G on `/`**.
#
# The fourth is a veto rather than a measure:
#
#   4. **will `spur-authz` accept the mounts this flow makes.** There is an
#      authorization plugin on the daemon, and its refusal names neither
#      docker's usual vocabulary nor the variable at fault:
#      `denied [BH]: /home:/home -- mount your own directory instead`.
#      `-v $HOME:$HOME` and `-v /shared_nfs:/shared_nfs` both pass, so the rule
#      is about *whose* directory and not about depth. It caught m3's derived
#      mount. Probed with the mounts this flow actually makes rather than with a
#      canonical example: **a probe that tests a mount nobody makes cannot see
#      the refusal that stops the run.**
#
# Each rule has been observed to fire *both* ways before being trusted — the
# anchor returns `yes`, `no` and `unknown` on one node's four local images; the
# free-card path was exercised against a stub `rocm-smi` because no real node
# had ever made that branch execute. **A `NO` from a rule that has never fired
# the other way is not evidence.**
#
# ---------------------------------------------------------------------------
# Why this is not "read sinfo"
#
# **Slurm's view is not the truth here.** Co-tenants run containers through the
# host docker daemon, outside Slurm entirely, so `docker ps` inside the node
# shows the whole machine and Slurm shows nothing. Measured: `crsuse2-m2m-057`
# is `idle` to sinfo and has another tenant's `cell_e_e1_full` on all eight
# cards at 42 % VRAM. Every node frustration of 2026-09-04 was that difference.
#
# Why this does not take a hold
#
# The expensive version of this search is `sbatch -t 08:00:00`, look, `scancel`
# — a shared-resource action per candidate, one at a time, with a human
# remembering to release. It is not necessary: a **six-minute** job under
# `amd-burst-qos` lands on a named node in ~20 s, runs on the host as you, and
# ends by itself. Nothing to remember to release, and burst's bottom priority
# costs nothing at this size. `-w <node>` pins it, so the probe measures the
# node you asked about rather than one the scheduler preferred.
#
# `srun` cannot do this: it has no `--qos` flag, so it runs under the team QOS,
# which sat at `PENDING (QOSGrpNodeLimit)` — `amd-primus-qos` is capped at
# node=15 and the team is at the cap. That is why every probe here is `sbatch`.
#
# What it will not do
#
# * never pulls an image — an absent base is a fact about the node, and pulling
#   gigabytes to answer "is this node worth taking" costs more than the answer;
# * never stops, removes or inspects into another tenant's container — it reads
#   names so a reader can tell one tenant's five containers from five tenants;
# * never takes a hold, so there is nothing here to forget to `scancel`.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PAYLOAD="$HERE/probe_payload.sh"
STAMP="$(date +%Y%m%dT%H%M%S)"
#: **Results go to scratch, never beside the script.** These three files live in
#: the package now, and a sweep writes one directory per run — 57 rows, a job
#: script and a log. Defaulting that under `$HERE` would have the package
#: accumulate run output in `assets/lib/`, which is the tree every body is
#: staged from. Override with `NODEPROBE_RESULTS` (CONTRACT §5 principle 5:
#: temp activity lives in scratch, the repo receives deliverables).
RESULTS="${NODEPROBE_RESULTS:-$HOME/ws_handoff_refine_m2/nodeprobe/results}/$STAMP"
OUT="$RESULTS/rows.jsonl"
mkdir -p "$RESULTS"

#: A "free half" is the ask (M5 needs two arms; m1-m4 share one container).
NEED=4
#: `/mnt/m2m_nobackup` is dockerd's root. `crsuse2-m2m-186` was released over
#: 3.4 G there, so disk is a first-class reason and not a footnote.
NEED_DISK_GB=200
#: burst's `MaxSubmitPU` is 4 and **it counts jobs you already have**, including
#: a hold under a different QOS. Measured: with the team's 8 h hold running, the
#: fourth probe of a wave came back `QOSMaxSubmitJobPerUserLimit` and that node
#: went unmeasured. So the width is computed against what is outstanding rather
#: than fixed at the cap, and `--width` overrides it.
WIDTH=""
#: How long to wait for a wave before giving up on it. Under burst a probe that
#: is going to start starts in ~20 s; one still queued after this is the
#: scheduler saying the node is not available, which is an answer and not a
#: reason to keep waiting.
WAVE_WAIT_S=300

nodes=()
while [ $# -gt 0 ]; do
  case "$1" in
    --auto)  mapfile -t auto < <(sinfo -h -N -o "%N %t" | awk '$2=="idle"||$2=="mix"{print $1}' | sort -u)
             nodes+=("${auto[@]}"); shift ;;
    --need)  NEED="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    -*)      echo "unknown flag $1" >&2; exit 2 ;;
    *)       nodes+=("$1"); shift ;;
  esac
done
[ "${#nodes[@]}" -gt 0 ] || { echo "usage: $0 [--auto] [--need N] [node …]" >&2; exit 2; }

if [ -z "$WIDTH" ]; then
  mine="$(squeue -u "$USER" -h -o "%i" 2>/dev/null | grep -c .)"
  WIDTH=$(( 4 - mine )); [ "$WIDTH" -lt 1 ] && WIDTH=1
  echo "you have $mine job(s) outstanding, so the wave width is $WIDTH (burst MaxSubmitPU=4)"
fi

echo "probing ${#nodes[@]} node(s), $WIDTH at a time, need >= $NEED free cards and >= ${NEED_DISK_GB}G"
echo "results: $RESULTS"

job_script() {
  cat <<EOF
#!/bin/bash
#SBATCH --partition=amd-spur
#SBATCH --nodes=1
#SBATCH --time=00:06:00
PROBE_OUT="$OUT" bash "$PAYLOAD"
EOF
}
SCRIPT="$RESULTS/job.sh"; job_script > "$SCRIPT"

# Submit in waves. A node that never produces a row is reported as such rather
# than dropped — "we could not measure it" and "it is unsuitable" are different
# answers and only one of them is worth re-checking later.
submitted=()
i=0
while [ $i -lt ${#nodes[@]} ]; do
  wave=("${nodes[@]:$i:$WIDTH}")
  jobs=()
  for n in "${wave[@]}"; do
    j="$(sbatch --parsable -q amd-burst-qos -w "$n" "$SCRIPT" 2>"$RESULTS/submit.err")"
    if [ -n "$j" ]; then jobs+=("$j:$n"); submitted+=("$n"); else echo "  $n: submit refused ($(tail -1 "$RESULTS/submit.err"))"; fi
  done
  # Wait for the wave, with a ceiling: a probe that outlives its own walltime is
  # the scheduler telling us the node is not available, which is an answer.
  for _ in $(seq 1 $((WAVE_WAIT_S / 10))); do
    left=0
    for jn in "${jobs[@]}"; do
      squeue -j "${jn%%:*}" -h -o "%T" 2>/dev/null | grep -q . && left=$((left+1))
    done
    [ "$left" -eq 0 ] && break
    sleep 10
  done
  for jn in "${jobs[@]}"; do
    if squeue -j "${jn%%:*}" -h -o "%T" 2>/dev/null | grep -q .; then
      echo "  ${jn#*:}: still queued after ${WAVE_WAIT_S}s, cancelling"
      scancel "${jn%%:*}" 2>/dev/null
    fi
  done
  i=$((i + WIDTH))
  echo "  … $((i < ${#nodes[@]} ? i : ${#nodes[@]}))/${#nodes[@]}"
done

# **One renderer, called — not a second copy inlined here.** The verdict rules
# live in `report.py` so that a sweep can be re-read without being re-run, and a
# heredoc copy of them here would be a second answer to "what disqualifies a
# node" that drifts from the first the day one of them changes.
python3 "$HERE/report.py" "$OUT" --need "$NEED" --disk "$NEED_DISK_GB" \
  --asked "$(IFS=,; echo "${submitted[*]}")"
