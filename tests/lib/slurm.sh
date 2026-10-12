# shellcheck shell=bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# SLURM/Spur plumbing: scheduler settings, the account/QoS ladder, refusal
# classifiers, node-availability queries and the single-node tier dispatcher.

_default_partition() {
  command -v sinfo >/dev/null 2>&1 || return 0
  sinfo -h -o '%P' 2>/dev/null | sed -n 's/\*$//p' | head -1
}

init_slurm_config() {
  SLURM_PART="${INFERA_E2E_SLURM_PARTITION:-$(_default_partition)}"
  SLURM_PART="${SLURM_PART:-amd-spur}"
  SLURM_TIME="${INFERA_E2E_SLURM_TIME:-02:00:00}"
  # Ceiling on real scheduler submissions per tier, retries and rotations included.
  SLURM_MAX_ATTEMPTS="${INFERA_E2E_SLURM_MAX_ATTEMPTS:-5}"
  case "$SLURM_MAX_ATTEMPTS" in
    "" | *[!0-9]* | 0)
      echo "FATAL: INFERA_E2E_SLURM_MAX_ATTEMPTS must be a positive integer (got '$SLURM_MAX_ATTEMPTS')." >&2
      exit 2 ;;
  esac
  # Total wait for a per-user submit slot; kept inside the workflow's job timeouts.
  SLOT_WAIT="${INFERA_E2E_SLURM_SLOT_WAIT:-1200}"
  SLOT_POLL="${INFERA_E2E_SLURM_SLOT_POLL:-60}"
  _SLOT_WAITED=0
  HOLD_WAIT="${INFERA_E2E_HOLD_WAIT:-60}"

  # SRUN_EXTRA applies to new allocations; only STEP_SRUN_EXTRA may reach steps
  # inside an existing one.
  _SLURM_USER_SRUN_EXTRA="${INFERA_E2E_SRUN_EXTRA:-}"
  _SLURM_STEP_SRUN_EXTRA="${INFERA_E2E_STEP_SRUN_EXTRA:-}"
  _SLURM_ACCOUNT_QOS_PAIRS=()
  if [ -n "${INFERA_E2E_SLURM_ACCOUNT_QOS_PAIRS:-}" ]; then
    IFS=, read -ra _SLURM_ACCOUNT_QOS_PAIRS <<< "$INFERA_E2E_SLURM_ACCOUNT_QOS_PAIRS"
  elif [ "$SLURM_PART" = "amd-spur" ]; then
    _SLURM_ACCOUNT_QOS_PAIRS=(
      "amd-collectives:amd-collectives-qos"
      "amd-primus:amd-primus-qos"
      "amd-general:amd-general-qos"
      "amd-burst:amd-burst-qos"
      "amd-general:amd-burst-qos"
      "amd-primus:amd-burst-qos"
    )
  fi
  _CRED=0
  _SLURM_ACCOUNT=""
  _SLURM_QOS=""

  # On Spur `spur show` stands in for a missing scontrol alias.
  _SCTL=""
  local c
  for c in scontrol spur; do
    if command -v "$c" >/dev/null 2>&1; then _SCTL="$c"; break; fi
  done
}

_have_slurm() { command -v srun >/dev/null 2>&1; }
_have_sc() { [ -n "$_SCTL" ]; }
_sc() { [ -n "$_SCTL" ] || return 127; "$_SCTL" "$@"; }
_in_allocation() { [ -n "${SLURM_JOB_ID:-${SLURM_JOBID:-}}" ]; }

# The rung in use is shared by every submission in this run: once a rung has
# refused us, later dispatches and holds must not spend submissions on it again.
_set_slurm_account_qos() {
  local pair="${_SLURM_ACCOUNT_QOS_PAIRS[$1]}"
  _SLURM_ACCOUNT="${pair%%:*}"
  _SLURM_QOS="${pair#*:}"
  INFERA_E2E_SRUN_EXTRA="-A $_SLURM_ACCOUNT -q $_SLURM_QOS${_SLURM_USER_SRUN_EXTRA:+ $_SLURM_USER_SRUN_EXTRA}"
  export INFERA_E2E_SRUN_EXTRA
}
_use_cred() {
  [ "${#_SLURM_ACCOUNT_QOS_PAIRS[@]}" -gt 0 ] && _set_slurm_account_qos "$_CRED"
  return 0
}
_next_cred() {
  [ $((_CRED + 1)) -lt "${#_SLURM_ACCOUNT_QOS_PAIRS[@]}" ] || return 1
  _CRED=$((_CRED + 1))
  _set_slurm_account_qos "$_CRED"
}
_account_qos_label() {
  printf 'account=%s qos=%s' "${_SLURM_ACCOUNT:-default}" "${_SLURM_QOS:-default}"
}

# Refusals the next account/QoS rung can fix (credential, association, group limit).
_accounting_blocked() {
  case "$1" in
    QOS* | Qos* | qos* | Assoc* | Association* | Accounting* | InvalidAccount* | \
    *"Invalid account"* | *"Invalid qos"* | *"invalid account"* | *"invalid qos"* | \
    *"not associated with"* | *"no association"* | *"not permitted"* | \
    *"QOSGrpNodeLimit"* | \
    *"QOS"*"does not exist"* | *"Qos"*"does not exist"* | *"qos"*"does not exist"* | \
    *"violates accounting/QOS policy"*) return 0 ;;
    *) return 1 ;;
  esac
}
# The per-user submit ceiling: held by this fleet's own sibling legs, so wait for one
# to finish. No job is created, and no rung can help (the limit is per user per QoS).
_submit_slot_blocked() {
  case "$1" in
    *QOSMaxSubmitJobPerUser* | *QOSMaxJobsPerUser* | *AssocMaxSubmitJob* | \
    *"limit on submitted jobs per user"*) return 0 ;;
    *) return 1 ;;
  esac
}
# A wrong controller or auth plugin, e.g. a runner whose env predates a cluster change.
# Retrying cannot fix it.
_auth_refused() {
  case "$1" in
    *"does not have valid authentication credentials"*) return 0 ;;
    *) return 1 ;;
  esac
}
_report_auth_refused() {
  echo "[$1] FATAL: the scheduler rejected our credentials (SPUR_CONTROLLER_ADDR=${SPUR_CONTROLLER_ADDR:-unset})" >&2
  echo "[$1] fix: compare SPUR_* with /etc/environment; CI refreshes them via .github/scripts/refresh_spur_env.sh" >&2
}

# Stays on the refused rung: the better ones already turned this tier away.
_wait_for_slot() {
  local label="$1" mine
  if [ "$_SLOT_WAITED" -ge "$SLOT_WAIT" ]; then
    echo "[$label] no submit slot in this QoS after ${_SLOT_WAITED}s — giving up" >&2
    return 1
  fi
  mine=$(squeue -h -u "$(id -un)" -o '%i %j' 2>/dev/null | grep -c 'infera-' || true)
  _SLOT_WAITED=$((_SLOT_WAITED + SLOT_POLL))
  echo "[$label] QoS submit slots all taken (${mine} of this fleet's own jobs in the queue) — waiting ${SLOT_POLL}s for one (${_SLOT_WAITED}/${SLOT_WAIT}s)" >&2
  sleep "$SLOT_POLL"
}

# Nodes of reservation $1, one per line. Non-zero = the query failed; empty = no such
# reservation. List all of them: `show reservation NAME` exits 1 for both cases.
_reservation_nodes() {
  local out
  out=$(_sc show reservation 2>&1) || { printf '%s\n' "$out" >&2; return 1; }
  printf '%s\n' "$out" | awk -v r="ReservationName=$1" '
    BEGIN{RS="";FS="\n"}
    $1==r { for(i=1;i<=NF;i++) if($i ~ /Nodes=/){ n=$i; sub(/.*Nodes=/,"",n); sub(/[[:space:]].*/,"",n); print n; exit } }' \
    | tr ',' '\n' | sed '/^$/d'
}

# Free for an exclusive whole-node, all-GPU job: not DRAIN/DOWN, no CPUs, GPUs or
# memory allocated, not in someone else's reservation, no running GPU job listed.
_node_free() {
  local out state alloc gpus mem cpus resv gpu_jobs
  out=$(_sc show node "$1" 2>/dev/null) || return 1

  state=$(printf '%s\n' "$out" | grep -oE 'State=[A-Z+_]+' | head -1 | cut -d= -f2)
  case "$state" in
    *DRAIN* | *DOWN* | *FAIL* | *NOT_RESPONDING* | *INVAL*) return 1 ;;
  esac

  # Read gres/gpu from AllocTRES only: CfgTRES carries the node's total under the same key.
  alloc=$(printf '%s\n' "$out" | grep -oE 'AllocTRES=[^[:space:]]*' | head -1)
  gpus=$(printf '%s\n' "$alloc" | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
  mem=$(printf '%s\n' "$out" | grep -oE 'AllocMem=[0-9]+' | head -1 | cut -d= -f2)
  cpus=$(printf '%s\n' "$out" | grep -oE 'CPUAlloc=[0-9]+' | head -1 | cut -d= -f2)
  [ -z "$cpus" ] || [ "$cpus" -eq 0 ] 2>/dev/null || return 1
  if [ -n "$alloc" ] || [ -n "$mem" ]; then
    [ -z "$gpus" ] || [ "$gpus" -eq 0 ] 2>/dev/null || return 1
    [ -z "$mem" ] || [ "$mem" -eq 0 ] 2>/dev/null || return 1
  else
    [ -n "$cpus" ] && [ "$cpus" -eq 0 ] 2>/dev/null || return 1
  fi

  # Spur shows a node inside another team's reservation as IDLE; sbatch then refuses it.
  resv=$(printf '%s\n' "$out" | grep -oE 'ActiveReservation=[^[:space:]]+' | head -1 | cut -d= -f2)
  { [ -z "$resv" ] || [ "$resv" = "${INFERA_E2E_RESERVATION:-}" ]; } || return 1
  # %N is a compacted hostlist (node-[090,183]); an unreadable queue fails closed.
  gpu_jobs=$(squeue -h -t running -o '%b|%N' 2>/dev/null) || return 1
  awk -F'|' -v node="$1" '
    function has_node(list, node, prefix, id, n, parts, i, range) {
      if (list == node) return 1
      prefix = node; sub(/[0-9]+$/, "", prefix)
      if (index(list, prefix "[") != 1) return 0
      sub(/^[^[]*\[/, "", list); sub(/\].*$/, "", list)
      id = node; sub(/^.*-/, "", id)
      n = split(list, parts, ",")
      for (i = 1; i <= n; i++) {
        if (parts[i] == id) return 1
        if (split(parts[i], range, "-") == 2 &&
            id + 0 >= range[1] + 0 && id + 0 <= range[2] + 0) return 1
      }
      return 0
    }
    tolower($1) ~ /gpu:/ && has_node($2, node) { busy = 1 }
    END { exit busy ? 1 : 0 }' <<<"$gpu_jobs"
}

# Free nodes, one per line: the reservation's own, else the partition's idle/mixed
# ones (controllers lag while allocations drain; _node_free has the final say).
_candidate_nodes() {
  local n nodes=""
  if [ -n "${INFERA_E2E_RESERVATION:-}" ]; then
    nodes=$(_reservation_nodes "$INFERA_E2E_RESERVATION") || nodes=""
  fi
  [ -n "$nodes" ] ||
    nodes=$(sinfo -h -N -p "$SLURM_PART" -t idle,mixed -o '%n' 2>/dev/null | awk 'NF && !seen[$0]++')
  for n in $nodes; do
    _node_free "$n" && echo "$n"
  done
}

# Up to $1 free nodes not in the comma list $2.
_pick_idle_nodes() {
  local count="$1" excl=",${2:-}," n out=() all
  all="$(_candidate_nodes)"
  while read -r n; do
    [ -n "$n" ] || continue
    case "$excl" in *,"$n",*) continue ;; esac
    out+=("$n"); [ "${#out[@]}" -ge "$count" ] && break
  done <<< "$all"
  printf '%s\n' "${out[@]-}"
}

# Free nodes in reservation $1; -1 gone, -2 no scheduler CLI, -3 query failed.
_reservation_free() {
  local nodes n free=0
  _have_sc || { echo -2; return; }
  nodes=$(_reservation_nodes "$1") || { echo -3; return; }
  [ -n "$nodes" ] || { echo -1; return; }
  for n in $nodes; do
    _node_free "$n" && free=$((free + 1))
  done
  echo "$free"
}

# Our jobs currently borrowing open-partition nodes; non-zero if the queue cannot say.
_spill_inflight() {
  local out
  out=$(squeue -h -u "$(id -un)" -o '%j' 2>&1) || { printf '%s\n' "$out" >&2; return 1; }
  printf '%s\n' "$out" | grep -c -- 'spill' || true
}

_dirty_nodes_from() {
  sed -n "s/^${GPU_DIRTY_NODE_PREFIX}\([A-Za-z0-9._-]*\).*/\1/p" "$@" 2>/dev/null |
    awk 'NF && !seen[$0]++ { nodes = nodes sep $0; sep = "," } END { print nodes }'
}

# Killing the srun client does not stop a Spur job, so a cancel must scancel it:
# ids from the current dispatch's banner, else by this job's tag.
_CUR_DISPATCH_OUT=""
_cancel_dispatched() {
  local jids="" i suf csv left queue
  if [ -n "$_CUR_DISPATCH_OUT" ] && [ -f "$_CUR_DISPATCH_OUT" ]; then
    jids=$(grep -oE 'srun: job [0-9]+' "$_CUR_DISPATCH_OUT" 2>/dev/null \
      | grep -oE '[0-9]+' | sort -u | tr '\n' ' ')
  fi
  if [ -z "$jids" ] && [ -n "${INFERA_E2E_JOB_TAG:-}" ]; then
    suf="-${INFERA_E2E_JOB_TAG}"
    if ! queue=$(squeue -h -u "$(id -un)" -o '%i %j' 2>&1); then
      echo "[cleanup] squeue failed, leaving this run's jobs to the workflow's reclaim step: $queue" >&2
      return 1
    fi
    jids=$(printf '%s\n' "$queue" \
      | awk -v suf="$suf" '$2 ~ /^infera-ci-/ && substr($2, length($2)-length(suf)+1)==suf {print $1}' \
      | tr '\n' ' ')
  fi
  [ -n "$jids" ] || return 0
  echo "[cleanup] cancelling dispatched SLURM job(s): $jids" >&2
  csv=$(echo $jids | tr ' ' ',')
  for i in 1 2 3 4 5; do
    scancel $jids >/dev/null 2>&1 || true
    sleep 2
    # On Spur a gone job queries as exit 0 with no output; only that confirms it.
    left=$(squeue -h -j "$csv" -o '%i' 2>&1) && [ -z "$left" ] && return 0
  done
  echo "[cleanup] could not confirm the cancel of $jids: ${left:-no output}" >&2
}

# A waiting srun prints nothing, so report the queue state, and cancel + flag the
# waits only a resubmission can clear.  $1=srun-out $2=flag file $3=label
_watch_job() {
  local out="$1" hold="$2" label="$3" jid="" state reason waited=0
  local every="${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}" next="${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}"
  while sleep 5; do
    waited=$((waited + 5))
    [ -n "$jid" ] || jid=$(grep -oE 'job (allocation )?[0-9]+' "$out" 2>/dev/null \
      | grep -oE '[0-9]+' | head -1)
    [ -n "$jid" ] || continue
    state=$(squeue -h -j "$jid" -o '%T' 2>/dev/null)
    reason=$(squeue -h -j "$jid" -o '%r' 2>/dev/null)
    [ "$state" = "PENDING" ] || continue
    if _accounting_blocked "$reason"; then
      printf 'accounting:%s\n' "$reason" > "$hold"
      scancel "$jid" >/dev/null 2>&1
      return
    fi
    case "$reason" in
      JobHoldMaxRequeue* | JobLaunchFailure*)
        printf '%s\n' "${reason%% (*}" > "$hold"; scancel "$jid" >/dev/null 2>&1; return ;;
    esac
    if [ "$waited" -ge "$next" ]; then
      next=$((waited + every))
      echo "[$label] still QUEUED on SLURM after ${waited}s — job $jid, reason=${reason:-unknown}" >&2
    fi
  done
}

# Reservation while it has free nodes; when full, spill to the open partition up
# to INFERA_E2E_SPILL_MAX; when gone or unanswerable, the open partition.
_plan_placement() {
  local label="$1" rfree inflight smax="${INFERA_E2E_SPILL_MAX:-3}"
  local tag="${INFERA_E2E_JOB_TAG:+-$INFERA_E2E_JOB_TAG}"
  _PLACE_RESV=()
  _PLACE_JOB="infera-ci-${label}${tag}"
  _PLACE_MODE="open"
  [ -n "${INFERA_E2E_RESERVATION:-}" ] || return 0
  rfree=$(_reservation_free "$INFERA_E2E_RESERVATION")
  case "$rfree" in
    -1)
      echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' does not exist — falling back to open partition '$SLURM_PART'" >&2
      _PLACE_MODE="resv-gone->open" ;;
    -2 | -3)
      echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' could not be queried — falling back to open partition '$SLURM_PART'" >&2
      _PLACE_MODE="resv-unknown->open" ;;
    0)
      if inflight=$(_spill_inflight) && [ "$smax" -gt 0 ] && [ "$inflight" -lt "$smax" ]; then
        # The spill marker precedes the job-tag suffix so ci.yml's reclaim still matches.
        _PLACE_JOB="infera-ci-${label}-spill${tag}"
        _PLACE_MODE="spill($((inflight + 1))/$smax)"
      else
        _PLACE_RESV=(--reservation="$INFERA_E2E_RESERVATION"); _PLACE_MODE="resv-wait"
      fi ;;
    *)
      _PLACE_RESV=(--reservation="$INFERA_E2E_RESERVATION"); _PLACE_MODE="resv" ;;
  esac
}

# One srun of this script on an 8-GPU node, streamed live; returns srun's status.
#   $1=label $2=client-out $3=shared log or "" $4=flag file, then the remote args.
_srun_once() {
  local label="$1" out="$2" logf="$3" flag="$4"; shift 4
  local remote=(bash "$SCRIPT" "$@") excl=() owner="${INFERA_E2E_EXCLUSIVE:-}" tailpid srunpid watchpid rc
  # Shared mode: the remote leg writes its own output straight to the NFS log.
  [ -n "$logf" ] && remote=(bash -c 'lf="$1"; shift; exec >"$lf" 2>&1; exec bash "$@"' _ "$logf" "$SCRIPT" "$@")
  if ! _in_allocation; then excl=(--exclusive); owner=1; fi
  : > "$out"
  [ -n "$logf" ] && : > "$logf"
  rm -f "$flag"
  stdbuf -oL tail -n +1 -F "${logf:-$out}" 2>/dev/null &
  tailpid=$!
  # Backgrounded + wait: a foreground srun would defer the INT/TERM trap that scancels it.
  INFERA_E2E_LOCAL=1 INFERA_E2E_EXCLUSIVE="$owner" \
    srun -N1 -p "$SLURM_PART" --gres=gpu:8 "${excl[@]}" -t "$SLURM_TIME" \
      -J "$_PLACE_JOB" "${_PLACE_ARGS[@]}" ${INFERA_E2E_SRUN_EXTRA:-} \
      "${remote[@]}" > "$out" 2>&1 &
  srunpid=$!
  _watch_job "$out" "$flag" "$label" &
  watchpid=$!
  wait "$srunpid"; rc=$?
  kill "$watchpid" 2>/dev/null; wait "$watchpid" 2>/dev/null
  sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null
  return "$rc"
}

# 0 if a failed dispatch is worth another submission. Updates _dispatch_slurm's
# exclude / pin / attempt / fixed_node locals through bash's dynamic scoping.
_dispatch_retry() {
  local label="$1" out="$2" logf="$3" flag="$4" why client dirty ran blocked
  local logs=("$out")
  [ -n "$logf" ] && logs+=("$logf")
  client=$(cat "$out" 2>/dev/null)
  if [ -f "$flag" ]; then
    why=$(cat "$flag" 2>/dev/null)
    if [[ "$why" = accounting:* ]] && [ "${#_SLURM_ACCOUNT_QOS_PAIRS[@]}" -gt 0 ]; then
      blocked=$(_account_qos_label)
      if _next_cred; then
        echo "[$label] ${why#accounting:} blocks $blocked — trying $(_account_qos_label)" >&2
        return 0
      fi
      echo "[$label] all ${#_SLURM_ACCOUNT_QOS_PAIRS[@]} SLURM account/QoS pairs are blocked (${why#accounting:}) — giving up" >&2
      return 1
    fi
    # JobLaunchFailure names no node, so pin our own pick; a repeat then names one to exclude.
    if [[ "$why" = JobLaunchFailure* ]] && ! _in_allocation; then
      [ -n "$pin" ] && exclude="${exclude:+$exclude,}$pin"
      pin="$(_pick_idle_nodes 1 "$exclude" | head -1)"
      echo "[$label] $why — scheduler names no node; pinning ${pin:-nothing free}${exclude:+, excluding $exclude}" >&2
      sleep 5; return 0
    fi
    echo "[$label] job ${why:-held} — cancelled, retrying within the $SLURM_MAX_ATTEMPTS-submission limit in 5s" >&2
    sleep 5; return 0
  fi
  # No job was created, so the attempt is given back; only the wait is bounded.
  if _submit_slot_blocked "$client"; then
    attempt=$((attempt - 1))
    _wait_for_slot "$label"
    return $?
  fi
  if _auth_refused "$client"; then
    _report_auth_refused "$label"
    return 1
  fi
  if [ "${#_SLURM_ACCOUNT_QOS_PAIRS[@]}" -gt 0 ] && _accounting_blocked "$client"; then
    if _next_cred; then
      echo "[$label] SLURM rejected the account/QoS pair — trying $(_account_qos_label)" >&2
      return 0
    fi
    echo "[$label] SLURM rejected all ${#_SLURM_ACCOUNT_QOS_PAIRS[@]} account/QoS pairs — giving up" >&2
    return 1
  fi
  dirty="$(_dirty_nodes_from "${logs[@]}")"
  if [ -n "$dirty" ]; then
    exclude="${exclude:+$exclude,}$dirty"
    if [ "$fixed_node" -eq 1 ]; then
      echo "[$label] node $dirty has dirty GPU state, but this one-node allocation cannot reselect" >&2
      return 1
    fi
    echo "[$label] node(s) $dirty still have GPU owners after exclusive cleanup — excluding, retrying elsewhere" >&2
    return 0
  fi
  # Spur's banner names the node; stock SLURM only via the remote leg's marker line.
  ran="$(sed -n \
    -e 's/.*srun: job [0-9][0-9]* running on \([A-Za-z0-9._-]*\).*/\1/p' \
    -e 's/^INFERA_E2E_SLURM_NODE=\([A-Za-z0-9._-]*\)$/\1/p' \
    "${logs[@]}" 2>/dev/null | tail -1)"
  if grep -qiE 'node failure|Cannot connect to the Docker daemon' "${logs[@]}" 2>/dev/null; then
    [ -n "$ran" ] && exclude="${exclude:+$exclude,}$ran"
    if [ "$fixed_node" -eq 1 ]; then
      echo "[$label] node ${ran:-?} is unusable, but this one-node allocation cannot reselect" >&2
      return 1
    fi
    echo "[$label] node ${ran:-?} unusable — excluding, retrying elsewhere" >&2
    return 0
  fi
  if grep -qiE 'not the Raft leader|service is currently unavailable|job submission failed' "$out" 2>/dev/null; then
    echo "[$label] transient controller error — retry in 15s" >&2
    sleep 15; return 0
  fi
  return 1
}

# Drops aged-out per-run folders (and pre-folder flat logs) under the shared log root.
_prune_shared_logs() {
  local ttl="${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" root
  root="$(dirname "$SHARED_LOG_DIR")"
  find "$root" -mindepth 1 -maxdepth 1 -type d -mmin "+$ttl" -exec rm -rf {} + 2>/dev/null
  find "$root" -mindepth 1 -maxdepth 1 -type f -name '*.log' -mmin "+$ttl" -delete 2>/dev/null
  return 0
}

# Runs this script with the given args on one SLURM GPU node, retrying what a new
# placement can fix, within SLURM_MAX_ATTEMPTS real submissions.  $1=label
_dispatch_slurm() {
  local label="$1"; shift
  if ! _have_slurm; then
    _skip_or_fail "$label" \
      "no SLURM: srun is not on PATH, so this tier cannot be dispatched to a GPU node" \
      "expose the SLURM client on this host, or run where docker + >=8 AMD GPUs are present"
    return $?
  fi
  _in_allocation &&
    echo "[$label] inside allocation ${SLURM_JOB_ID:-$SLURM_JOBID} — dispatching as a step in it"
  local out="$SCRATCH/.dispatch-$label.out" flag="$SCRATCH/.hold-$label" logf=""
  local prc=1 attempt=0 retryable=0 exclude="" pin="" fixed_node=0
  [ -n "$SHARED_LOG_DIR" ] && logf="$SHARED_LOG_DIR/dispatch-${label}-$$.log"
  _CUR_DISPATCH_OUT="$out"
  if _in_allocation && [ "${SLURM_JOB_NUM_NODES:-1}" -le 1 ] 2>/dev/null; then fixed_node=1; fi
  _use_cred
  while [ "$attempt" -lt "$SLURM_MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    case ",$exclude," in *,"$pin",*) pin="" ;; esac
    _plan_placement "$label"
    _PLACE_ARGS=(${exclude:+-x "$exclude"} ${pin:+-w "$pin"} "${_PLACE_RESV[@]}")
    echo "[$label] dispatch $attempt/$SLURM_MAX_ATTEMPTS to '$SLURM_PART' mode=$_PLACE_MODE $(_account_qos_label)${exclude:+ exclude=$exclude} (remote: $*)"
    echo "[$label] submitted to SLURM — it queues until a node frees up and prints nothing" \
         "until it starts; queue status follows every ${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}s."
    echo "[$label] streaming remote output below (live via ${logf:-$out}):"
    _srun_once "$label" "$out" "$logf" "$flag" "$@"; prc=$?
    [ "$prc" -eq 0 ] && break
    if [ -n "$logf" ] && [ -s "$out" ]; then
      echo "[$label] srun exited $prc — its client output was:" >&2
      tail -n 40 "$out" | sed "s/^/[$label]   /" >&2
    fi
    if _dispatch_retry "$label" "$out" "$logf" "$flag"; then retryable=1; else retryable=0; break; fi
  done
  if [ "$prc" -ne 0 ] && [ "$retryable" -eq 1 ]; then
    echo "[$label] SLURM submission limit reached ($SLURM_MAX_ATTEMPTS attempts) — giving up" >&2
  fi
  # Carry the remote leg's failure lines into this run's summary.
  awk '/FAILED TEST SUMMARY/{p=1; next} p && /^=+$/{p=0} p && /^  \[/{sub(/^  /, ""); print}' \
    "$out" ${logf:+"$logf"} 2>/dev/null >> "$SCRATCH/failures.txt"
  [ -n "$logf" ] && _prune_shared_logs
  return "$prc"
}
