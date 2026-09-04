#!/bin/bash
# Runs ON a candidate node, as the submitting user, on the host.
#
# Answers the only two questions that decide whether a node is worth holding,
# plus the one that cost us a node after we took it:
#
#   1. which GPUs are ACTUALLY free    -- rocm-smi, not Slurm
#   2. can Dockerfile.sglang build here -- m1's anchor line in a local base
#   3. is there disk                    -- 186 was released for 3.4 G
#
# **Slurm's view is not the truth here.** Co-tenants run containers through the
# host docker daemon, outside Slurm entirely, so a node Slurm calls `idle` can
# have another tenant on all eight cards. That difference is the whole reason
# this file exists; every fact below is read from the machine, never from the
# scheduler.
#
# Writes exactly one line of JSON to $PROBE_OUT so the driver never has to parse
# prose. Everything is bounded by a timeout: a probe that hangs is a probe that
# costs more than the hold it was meant to save.
set -u

OUT="${PROBE_OUT:?PROBE_OUT must name the results file}"
HOST="$(hostname)"
# **The node makes its own results directory.** `/home` is NFS and a directory
# created on the login node a second earlier is not reliably visible here yet;
# the first run of this file failed exactly that way, wrote nothing, and looked
# like a probe defect. Cheaper to create it than to diagnose it twice.
mkdir -p "$(dirname "$OUT")" 2>/dev/null

# `docker` here emits a config-file warning on every call because $HOME is not
# the one it expects. Harmless, and dropped rather than parsed around.
d() { timeout 60 docker "$@" 2>/dev/null; }

# --- 1. the cards ------------------------------------------------------------
# VRAM% per card. A card another tenant is serving on sits at 80-something; a
# genuinely free card is at 0. The bar is deliberately loose — 5 % — because a
# card can carry a few hundred MB of somebody's idle context without being in
# use, and because the decision this feeds ("is half the node free") does not
# get better from a tighter one.
FREE_BAR=5
mem="$(timeout 60 rocm-smi --showmemuse --csv 2>/dev/null | grep -E '^card[0-9]+,')"
free_cards=""; busy_cards=""; ncard=0
while IFS=, read -r card pct _rest; do
  [ -n "$card" ] || continue
  ncard=$((ncard + 1))
  n="${card#card}"
  if [ "${pct:-100}" -le "$FREE_BAR" ] 2>/dev/null; then
    free_cards="${free_cards}${free_cards:+,}${n}"
  else
    busy_cards="${busy_cards}${busy_cards:+,}${n}(${pct}%)"
  fi
done <<< "$mem"
nfree=0; [ -n "$free_cards" ] && nfree="$(awk -F, '{print NF}' <<< "$free_cards")"

# --- 2. the base image and m1's anchor ---------------------------------------
# **Local images only — this never pulls.** A probe that pulls a multi-gigabyte
# base to answer "is this node worth taking" costs more than the question, and
# on a node with no disk it is also the thing that fills it. An absent base is
# itself a fact about the node and is reported as `none`.
ANCHOR_FILE=/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py
ANCHOR_KEY='background=request.background,'
ANCHOR_WANT='require_reasoning'
imgs="$(d image ls --format '{{.Repository}}:{{.Tag}}' | grep -Ei 'sgl-dev|sglang' | grep -v '<none>' | head -4)"
anchor_rows=""
while read -r img; do
  [ -n "$img" ] || continue
  # **Both questions in one container start.** The anchor asks *can this build*;
  # `infera.engine.sglang` and `infera.server` importing asks *can this serve
  # right now*, and they are different answers with a four-minute build between
  # them — m1's correction: a base carrying the anchor does **not** mean a
  # servable image exists. Measured 2026-09-04 on `crsuse2-m2m-037`, where both
  # `infera/engine-sglang` images answered `servable` and the node needed no
  # build at all, exactly as `006` had not. Two greps, one `docker run`, because
  # the container start is the whole cost.
  got="$(timeout 240 docker run --rm --entrypoint /bin/bash "$img" -lc \
           "grep -A1 '$ANCHOR_KEY' '$ANCHOR_FILE' 2>/dev/null;
            python3 -c 'import infera.engine.sglang, infera.server' 2>/dev/null \
              && echo E2E_SERVABLE" 2>/dev/null)"
  # A following line naming require_reasoning means Dockerfile.sglang can build
  # against this base; a bare `)` means it cannot. Absent file / failed run is
  # neither, and is reported as such rather than folded into "no".
  if   [ -z "$got" ];                          then verdict="unknown"
  elif grep -q "$ANCHOR_WANT" <<< "$got";      then verdict="yes"
  else                                              verdict="no"
  fi
  # The two modules the kit's `start_worker.sh` and `start_router.sh` exec, so
  # this is the same reading the leader took by hand on 006 before spending it.
  servable=false; grep -q E2E_SERVABLE <<< "$got" && servable=true
  anchor_rows="${anchor_rows}${anchor_rows:+,}{\"image\":\"${img}\",\"anchor\":\"${verdict}\",\"servable\":${servable}}"
done <<< "$imgs"

# --- 3. disk, BOTH filesystems -----------------------------------------------
# `/mnt/m2m_nobackup` holds dockerd's root, so it decides whether an image can
# land. **`/` is a different number and it is the one that stopped a build**:
# `crsuse2-m2m-186` had the right base, free-looking Slurm state, and **3.4 G on
# `/`**, which is where docker builds. Reporting only the big number would have
# called that node fine. Two filesystems, two gates.
disk_gb="$(df -BG --output=avail /mnt/m2m_nobackup 2>/dev/null | tail -1 | tr -dc '0-9')"
root_gb="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"

# --- 3a. is the shared filesystem here, and are the weights on it ------------
# **The tier the first two do not cover, and it cost a hold to learn.** m5 took
# `crsuse2-m2m-037` on a SERVABLE verdict and released it again: the node has
# **no `/shared_nfs` at all**. A servable image cannot tell you the node is
# missing the filesystem the weights live on — the image is a fact about the
# node's docker, the mount is a fact about the node.
#
# It is also what `require_visible_on_node` needs (`assets/lib/remote.sh:170`):
# this package runs bodies on the node by absolute path and exchanges files
# through the zone, so **both hosts must mount the run root**. A node without
# the shared filesystem fails that three layers from its symptom.
#
# Two readings, because they fail separately: the mount can be present while
# the model path is not, which is the same distinction as image-present versus
# image-servable one directory over.
shared_mnt=false; mount 2>/dev/null | grep -q ' /shared_nfs ' && shared_mnt=true
model_ok=false
[ -n "${E2E_MODEL_PATH:-}" ] || E2E_MODEL_PATH=/shared_nfs/yihou/models/Qwen3.6-27B
[ -d "$E2E_MODEL_PATH" ] && model_ok=true

# --- 3b. can this node run OUR container -------------------------------------
# **There is an authorization plugin on the daemon**, and its refusal names
# neither docker's usual vocabulary nor the variable at fault:
#
#   docker run -v /home:/home … ->
#     authorization denied by plugin spur-authz: denied [BH]: /home:/home
#     -- mount your own directory instead
#
# Measured on 243: `-v /home/yihou:/home/yihou` and `-v /shared_nfs:/shared_nfs`
# both pass, so the rule is about **whose** directory and not about depth. It
# caught m3's derived mount before rung 3 could. Probed with the mounts this
# flow actually uses rather than with a canonical example, because a probe that
# tests a mount nobody makes cannot see the refusal that stops the run.
mounts="unknown"
if [ -n "${imgs:-}" ]; then
  probe_img="$(head -1 <<< "$imgs")"
  mout="$(timeout 120 docker run --rm \
            -v "$HOME:$HOME" -v /shared_nfs:/shared_nfs \
            --entrypoint /bin/true "$probe_img" 2>&1)"
  case "$mout" in
    "")                      mounts="ok" ;;
    *"authorization denied"*) mounts="denied" ;;
    *)                       mounts="unknown" ;;
  esac
fi
ncontainers="$(d ps -q | grep -c . )"
# Names **and process counts**, so a reader can tell one tenant's eight
# containers from eight tenants, and a serving container from an idle one.
#
# **`docker top`, never `docker exec`.** Both answer "is anything alive in
# there"; `exec` answers it by running a process inside somebody else's
# container, and `top` answers it from outside through the daemon. Same fact,
# strictly less access, and on the right side of the line that says we read
# other tenants' names and do nothing else with them.
#
# **This does NOT distinguish a corpse from live work, and must not be read as
# if it did.** Measured 2026-09-04 (RUN-PLAN, `41c8540`): job `109192` was
# cancelled while four of our containers were serving on 006, and fifteen
# minutes later all four were still `Up`, the engine still answered `/health`
# with 200, and all eight cards read 74–76 %. Containers talk to the **host**
# daemon, so they are not in the job's cgroup and nothing tears them down when
# the hold ends. A corpse in that sense is *genuinely running* — `State.Running`
# is true, `docker top` shows a full process table, and both are correct. What
# died is the *claim on the node*, not the process, and no reading available
# here can see that. Reported as facts a person can weigh, never as a verdict.
tenants="$(d ps --format '{{.Names}}' | head -8 \
           | while read -r c; do
               n="$(timeout 30 docker top "$c" 2>/dev/null | tail -n +2 | grep -c .)"
               printf '%s(%s);' "$c" "${n:-?}"
             done)"

printf '{"node":"%s","cards_total":%s,"cards_free":%s,"free":"%s","busy":"%s","disk_gb":%s,"root_gb":%s,"shared_nfs":%s,"model_path":%s,"mounts":"%s","containers":%s,"tenants":"%s","bases":[%s]}\n' \
  "$HOST" "${ncard:-0}" "${nfree:-0}" "${free_cards}" "${busy_cards}" \
  "${disk_gb:-0}" "${root_gb:-0}" "${shared_mnt}" "${model_ok}" \
  "${mounts}" "${ncontainers:-0}" "${tenants}" "${anchor_rows}" \
  >> "$OUT"
