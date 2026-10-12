# shellcheck shell=bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# PD-disaggregated tier: a pytest orchestrator on this host drives prefill and
# decode on two nodes held for the whole run (see tests/e2e/harness/launcher.py).

_DISAG_NODES=""
_HOLDER_JID=""
_INHERITED_JID=""
_HOLD_SUBMISSIONS=0

# A killed run skips pytest's teardown, so remove this job's PD containers here.
_wipe_disag_nodes() {
  local n cmd extra=() resv=() wrapped
  for n in ${_DISAG_NODES//,/ }; do
    echo "[cleanup] removing this job's PD containers on $n" >&2
    if [ -n "$_HOLDER_JID" ]; then
      cmd=(srun --overlap --nodes=1 --ntasks=1 --nodelist "$n" --jobid "$_HOLDER_JID")
    else
      cmd=(srun -N1 -n1 -p "$SLURM_PART" -w "$n")
      [ -n "${INFERA_E2E_RESERVATION:-}" ] && resv=("--reservation=$INFERA_E2E_RESERVATION")
      read -ra extra <<< "${INFERA_E2E_SRUN_EXTRA:-}"
    fi
    # Named so ci.yml's `infera-ci-`+run-id reclaim finds it if this is interrupted too.
    cmd+=("${resv[@]}" "${extra[@]}" \
      -J "infera-ci-wipe-${INFERA_E2E_JOB_TAG:-local}" -t 00:05:00 \
      bash -lc 'docker rm -f $(docker ps -aq --filter "label=infera.e2e.job_tag=$1") 2>/dev/null || true' \
      _ "$CTR_TAG")
    # Spur steps inside an allocation need a pseudo-TTY.
    if [ -n "$_HOLDER_JID" ] && [ -n "${SPUR_CONTROLLER_ADDR:-}" ] && command -v script >/dev/null 2>&1; then
      printf -v wrapped '%q ' "${cmd[@]}"
      script -qefc "$wrapped" /dev/null >/dev/null 2>&1 || true
    else
      "${cmd[@]}" >/dev/null 2>&1 || true
    fi
  done
  _DISAG_NODES=""
}

# An allocation we were handed (_inherited_pair) belongs to its owner; never cancel it.
_release_hold() {
  [ -n "$_HOLDER_JID" ] || return 0
  if [ "$_HOLDER_JID" = "$_INHERITED_JID" ]; then
    echo "[e2e disagg] leaving inherited allocation $_HOLDER_JID to its owner" >&2
  else
    echo "[e2e disagg] releasing held nodes (scancel $_HOLDER_JID)" >&2
    scancel "$_HOLDER_JID" >/dev/null 2>&1 || true
  fi
  _HOLDER_JID=""
}

# Waits for two free nodes rather than giving up: the pool is shared with the other
# legs. $1=exclude list.
_wait_for_pair() {
  local excl="$1" waited=0 every=30 limit="${INFERA_E2E_WAIT_NODES_TIMEOUT:-6400}" nodes
  while :; do
    nodes="$(_pick_idle_nodes 2 "$excl")"
    [ "$(printf '%s\n' "$nodes" | sed '/^$/d' | wc -l)" -ge 2 ] && { printf '%s\n' "$nodes"; return 0; }
    [ "$waited" -ge "$limit" ] && return 1
    [ $((waited % 120)) -eq 0 ] &&
      echo "[e2e disagg] fewer than 2 free nodes — waiting (${waited}s/${limit}s)" >&2
    sleep "$every"; waited=$((waited + every))
  done
}

# A running pair holder other than $1, on the same nodes and submitted earlier.
_rival_holder() {
  local self="$1" mine
  mine=$(squeue -h -j "$self" -o '%N' 2>/dev/null)
  [ -n "$mine" ] || return 0
  squeue -h -t running -o '%i %j %N' 2>/dev/null | awk -v self="$self" -v mine="$mine" '
    $2 ~ /^infera-ci-hold-/ && $3 == mine && $1 + 0 < self + 0 { print $1; exit }'
}

# Waits up to HOLD_WAIT for hold job $1 to start.
# 0 = held, 1 = lost the pair to a rival, 2 = not started, 3 = account/QoS-blocked
_await_hold() {
  local jid="$1" pair="$2" try="$3" waited=0 info st="" rs="" other
  while [ "$waited" -lt "$HOLD_WAIT" ]; do
    info=$(_sc show job "$jid" 2>/dev/null)
    st=$(printf '%s\n' "$info" | grep -oE 'JobState=[A-Z_]+' | cut -d= -f2)
    rs=$(printf '%s\n' "$info" | grep -oE 'Reason=[A-Za-z]+' | cut -d= -f2)
    if [ "$st" = RUNNING ]; then
      # Two legs can submit for one pair at the same instant; the lower job id keeps it.
      other=$(_rival_holder "$jid")
      [ -z "$other" ] && { _HOLDER_JID="$jid"; return 0; }
      scancel "$jid" >/dev/null 2>&1
      echo "[e2e disagg] holder $jid started on $pair but $other holds it too — yielding" >&2
      return 1
    fi
    _accounting_blocked "$rs" && break
    case "$st" in NODE_FAIL | FAILED | CANCELLED) break ;; esac
    sleep 5; waited=$((waited + 5))
  done
  scancel "$jid" >/dev/null 2>&1
  echo "[e2e disagg] hold attempt $try on $pair not started (${st:-?}/${rs:-?})" >&2
  _accounting_blocked "$rs" && return 3
  return 2
}

# Holds both nodes' GPUs for the run, so none is handed out between per-step sruns.
# 0 held, 1 lost race, 2 never placed, 3 ceiling hit, 4 no submit slot, 5 auth refused
_hold_pair() {
  local pair="$1" script="$SCRATCH/hold.sh" out jid i retries=3 rotate acct=()
  # A real script, not --wrap: on Spur --wrap always NODE_FAILs at -N2.
  printf '#!/bin/bash\nsleep %s\n' "${INFERA_E2E_HOLD_SLEEP:-10800}" > "$script"
  [ "${#_SLURM_ACCOUNT_QOS_PAIRS[@]}" -gt 0 ] && retries=1
  while :; do
    _use_cred
    acct=()
    [ -n "$_SLURM_ACCOUNT" ] && acct=(-A "$_SLURM_ACCOUNT" -q "$_SLURM_QOS")
    rotate=0
    for ((i = 1; i <= retries; i++)); do
      [ "$_HOLD_SUBMISSIONS" -ge "$SLURM_MAX_ATTEMPTS" ] && return 3
      _HOLD_SUBMISSIONS=$((_HOLD_SUBMISSIONS + 1))
      echo "[e2e disagg] hold submission $_HOLD_SUBMISSIONS/$SLURM_MAX_ATTEMPTS on $pair ($(_account_qos_label), credential attempt $i/$retries)"
      if ! out=$(sbatch --parsable --exclusive -N2 -n2 -w "$pair" --gres=gpu:8 -p "$SLURM_PART" \
          -t "$SLURM_TIME" -J "infera-ci-hold-${INFERA_E2E_JOB_TAG:-local}" -o /dev/null -e /dev/null \
          ${INFERA_E2E_RESERVATION:+--reservation="$INFERA_E2E_RESERVATION"} \
          "${acct[@]}" "$script" 2>&1); then
        echo "[e2e disagg] hold submission rejected: $out" >&2
        # A per-user ceiling refusal creates no job: hand the submission back and wait.
        if _submit_slot_blocked "$out"; then
          _HOLD_SUBMISSIONS=$((_HOLD_SUBMISSIONS - 1))
          _wait_for_slot "e2e disagg" || return 4
          i=$((i - 1))
          continue
        fi
        _auth_refused "$out" && { _report_auth_refused "e2e disagg"; return 5; }
        _accounting_blocked "$out" && { rotate=1; break; }
        continue
      fi
      # Submit filters may print banner lines too; take the last bare id ("id;cluster" if federated).
      jid=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\?$/\1/p' | tail -1)
      if [ -z "$jid" ]; then
        echo "[e2e disagg] hold submitted but no job id in: $out" >&2
        continue
      fi
      _await_hold "$jid" "$pair" "$i"
      case $? in
        0) return 0 ;;
        1) return 1 ;;
        3) rotate=1; break ;;
      esac
    done
    # Only a credential refusal is fixable by the next rung; busy nodes mean re-pick.
    { [ "$rotate" = 1 ] && _next_cred; } || break
    echo "[e2e disagg] trying the next SLURM account/QoS pair" >&2
  done
  [ "$_HOLD_SUBMISSIONS" -ge "$SLURM_MAX_ATTEMPTS" ] && return 3
  return 2
}

# "node1 node2" of a running allocation we were handed (salloc/sbatch/outer CI job),
# else empty. Holding that pair ourselves would queue behind it forever.
_inherited_pair() {
  local jid="${SLURM_JOB_ID:-}" state nodelist expanded
  [ -n "$jid" ] || return 0
  state=$(squeue -h -j "$jid" -o '%T' 2>/dev/null | tr -d '[:space:]')
  [ "$state" = RUNNING ] || return 0
  nodelist=$(squeue -h -j "$jid" -o '%N' 2>/dev/null | tr -d '[:space:]')
  [ -n "$nodelist" ] || return 0
  # Expanding a compacted list needs `scontrol show hostnames`; without it, decline.
  expanded=$(scontrol show hostnames "$nodelist" 2>/dev/null | awk 'NF')
  [ "$(printf '%s\n' "$expanded" | grep -c .)" -ge 2 ] || return 0
  printf '%s\n' "$expanded" | head -2 | paste -sd' ' -
}

# An expired or unanswerable reservation would fail every step's srun; drop it.
_drop_dead_reservation() {
  local label="$1" nodes
  [ -n "${INFERA_E2E_RESERVATION:-}" ] || return 0
  if ! nodes=$(_reservation_nodes "$INFERA_E2E_RESERVATION"); then
    echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' could not be queried — falling back to open partition '$SLURM_PART'" >&2
    unset INFERA_E2E_RESERVATION
  elif [ -z "$nodes" ]; then
    echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' does not exist — falling back to open partition '$SLURM_PART'" >&2
    unset INFERA_E2E_RESERVATION
  fi
}

# Every remote op must be a step in the holder's allocation: a fresh job would queue
# behind our own holder forever. The launcher attaches via SLURM_JOB_ID.
_disagg_pytest() {
  local e="$1" nodes="$2" owner="$3" out="$4"
  env SLURM_JOB_ID="$_HOLDER_JID" \
    INFERA_E2E_EXCLUSIVE="$owner" \
    INFERA_E2E_STEP_SRUN_EXTRA="$_SLURM_STEP_SRUN_EXTRA" \
    INFERA_E2E_NODES="$nodes" INFERA_E2E_SLURM_PARTITION="$SLURM_PART" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s \
      ${INFERA_E2E_K:+-k "$INFERA_E2E_K"} \
      "$REPO/tests/e2e/pd_disag/$e" 2>&1 | tee "$out"
  return "${PIPESTATUS[0]}"
}

# One engine's suite: pick and hold a pair, run it, retry with a fresh pair on node faults.
#   $1=engine $2=inherited pair or ""
_disagg_engine() {
  local e="$1" inherited="$2" out="$SCRATCH/.e2e-disag.out"
  local attempt=0 max=3 ok=0 exclude="" races=0 owner=1 n1 n2 nodes hold_rc prc dirty
  local max_races="${INFERA_E2E_HOLD_RACE_MAX:-$SLURM_MAX_ATTEMPTS}"
  # A handed-over pair is fixed (one attempt); only its owner may attest exclusivity.
  if [ -n "$inherited" ]; then
    max=1
    owner="${INFERA_E2E_EXCLUSIVE:-}"
  fi
  echo "----- e2e disagg — tests/e2e/pd_disag/$e -----"
  : > "$out"
  _HOLD_SUBMISSIONS=0
  while [ "$attempt" -lt "$max" ]; do
    attempt=$((attempt + 1))
    # An explicit INFERA_E2E_NODES pin wins the first try, even over an inherited allocation.
    if [ -n "${INFERA_E2E_NODES:-}" ] && [ "$attempt" -eq 1 ]; then
      n1="${INFERA_E2E_NODES%%,*}"; n2="${INFERA_E2E_NODES##*,}"
    elif [ -n "$inherited" ]; then
      n1="${inherited%% *}"; n2="${inherited##* }"
    else
      nodes="$(_wait_for_pair "$exclude")"
      n1="$(printf '%s\n' "$nodes" | sed -n 1p)"
      n2="$(printf '%s\n' "$nodes" | sed -n 2p)"
    fi
    if [ -z "$n1" ] || [ -z "$n2" ] || [ "$n1" = "$n2" ]; then
      echo "[e2e disagg] WARNING: no 2 free nodes in '$SLURM_PART' within ${INFERA_E2E_WAIT_NODES_TIMEOUT:-6400}s — skipping $e" >&2
      break
    fi

    if [ -n "$inherited" ]; then
      _HOLDER_JID="$SLURM_JOB_ID"; _INHERITED_JID="$SLURM_JOB_ID"; hold_rc=0
    else
      _hold_pair "$n1,$n2"; hold_rc=$?
    fi
    case "$hold_rc" in
      0) ;;
      3) echo "[e2e disagg] SLURM hold submission limit reached ($SLURM_MAX_ATTEMPTS attempts) — giving up on $e" >&2; break ;;
      4) echo "[e2e disagg] no SLURM submit slot for a node hold within ${SLOT_WAIT}s — giving up on $e" >&2; break ;;
      5) break ;;
      *)
        races=$((races + 1))
        # A lost race is no node fault; a hold never placed would re-pick the same pair.
        [ "$hold_rc" -eq 2 ] && exclude="${exclude:+$exclude,}$n1,$n2"
        if [ "$races" -ge "$max_races" ]; then
          if [ "$hold_rc" -eq 2 ]; then
            echo "[e2e disagg] SLURM never placed a node hold in $races attempts — giving up on $e" >&2
          else
            echo "[e2e disagg] lost the node-hold race $races times — giving up on $e" >&2
          fi
          break
        fi
        echo "[e2e disagg] could not hold $n1,$n2 (attempt $races/$max_races) — re-picking in 30s" >&2
        attempt=$((attempt - 1)); sleep 30; continue ;;
    esac
    races=0

    echo "[e2e disagg] $e attempt $attempt/$max on nodes: $n1 (prefill), $n2 (decode)"
    _DISAG_NODES="$n1,$n2"
    _disagg_pytest "$e" "$n1,$n2" "$owner" "$out"; prc=$?
    _DISAG_NODES=""
    _release_hold
    [ "$prc" -eq 0 ] && { ok=1; break; }
    dirty="$(_dirty_nodes_from "$out")"
    if [ -n "$dirty" ]; then
      exclude="${exclude:+$exclude,}$dirty"
      echo "[e2e disagg] $e found dirty GPU state on $dirty — excluding and retrying with a fresh pair" >&2
      continue
    fi
    if grep -qiE 'node failure|Cannot connect to the Docker daemon|could not resolve a routable IP|docker build .* failed' "$out"; then
      exclude="${exclude:+$exclude,}$n1,$n2"
      echo "[e2e disagg] $e hit a bad node ($n1/$n2) — excluding, retrying with a fresh pair" >&2
      continue
    fi
    # Back-to-back reuse can leave Mooncake ports in TIME_WAIT.
    if grep -qiE 'Address already in use|bind:|exited before becoming active' "$out"; then
      echo "[e2e disagg] $e transient bind/port collision — retry in 45s (ports draining)" >&2
      sleep 45; continue
    fi
    break
  done
  grep -aE '^(FAILED|ERROR) ' "$out" 2>/dev/null | sed "s|^|[e2e disagg $e] |" >> "$SCRATCH/failures.txt"
  [ "$ok" -eq 1 ]
}

run_e2e_disagg() {
  local engines=("$@") e rc=0 deps_err inherited
  echo "===== e2e PD-disaggregated (cross-node, 2 nodes): ${engines[*]} ====="
  if ! _have_slurm; then
    _skip_or_fail "e2e disagg" \
      "no SLURM: srun is not on PATH, so the PD-disaggregated tests cannot run" \
      "expose the SLURM client on this host"
    return $?
  fi
  # Name the interpreter actually consulted and quote its ImportError.
  if ! deps_err=$(python3 -c "import pytest, pytest_asyncio, httpx" 2>&1); then
    _skip_or_fail "e2e disagg" \
      "the disagg orchestrator runs pytest on THIS host, and $(command -v python3 || echo 'python3 (not on PATH)') cannot import its deps: ${deps_err##*$'\n'}" \
      "pip install pytest pytest-asyncio httpx"
    return $?
  fi
  if [ -n "$SHARED_LOG_DIR" ]; then
    exec > >(stdbuf -oL tee -a "$SHARED_LOG_DIR/dispatch-disag-$$.log") 2>&1
  fi
  _drop_dead_reservation "e2e disagg"
  inherited="$(_inherited_pair)"
  [ -n "$inherited" ] && echo "[e2e disagg] reusing allocation $SLURM_JOB_ID on nodes: $inherited"
  for e in "${engines[@]}"; do
    _disagg_engine "$e" "$inherited" || rc=1
  done
  return "$rc"
}
