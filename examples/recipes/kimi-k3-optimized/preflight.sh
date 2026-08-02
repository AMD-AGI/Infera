#!/bin/sh
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Pre-flight for a kimi-k3-optimized deployment.
#
#   ./preflight.sh <NODE> --find                  -- where are the weights on that node?
#   ./preflight.sh <NODE> <MODEL_DIR> [--dspark]  -- is this deployment going to work?
#
# Checks, from the reader's position (kubectl only — no shell on the node), the
# things that otherwise surface 12-14 minutes into a deploy and blame something
# else:
#
#   * the namespace and the InferaDeployment CRD exist
#   * the node exists and has 8 free amd.com/gpu
#   * MODEL_DIR exists ON THAT NODE and actually holds the weights
#   * MODEL_DIR is local storage, not a network mount
#
# The two cases that make this worth running:
#
#   Empty directory. Copying a working node's path onto a node that never had
#   the weights leaves a path that EXISTS, so hostPath type: Directory passes,
#   the pod starts, and the engine crashloops minutes later with
#       ValueError: '/models/Kimi-K3': not a local path and HF resolution failed:
#         Repo id must be in the form 'repo_name' or 'namespace/repo_name'
#   which reads as a bad model name.
#
#   Network mount wearing a local-looking name. The shard count is right, the
#   directory is right, and weight load goes from ~8 minutes to ~95 -- past the
#   ready timeout, so the worker restarts mid-load forever and presents as a
#   crash loop rather than as slow storage.
#
# Exit 0 = clear to deploy. Exit 1 = at least one check failed.
set -u

# --- discovery mode ----------------------------------------------------------
# `preflight.sh <NODE> --find` answers the question the checks below assume you
# have already answered: where are the weights ON THAT NODE. You generally cannot
# log into it, and reading the path off an existing deployment does not work for
# PD -- PD needs all the GPUs, so the deployment you would have read it from has
# to be deleted first.
if [ "${2:-}" = "--find" ]; then
    NODE="${1:?usage: preflight.sh <NODE> --find}"
    POD=preflight-find-$$
    kubectl -n infera delete pod "$POD" --ignore-not-found >/dev/null 2>&1
    cat <<EOF | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: {name: $POD, namespace: infera}
spec:
  nodeSelector: {kubernetes.io/hostname: $NODE}
  restartPolicy: Never
  tolerations: [{operator: Exists}]
  containers:
  - name: p
    image: busybox
    command: ["sh","-c","find /host/mnt -maxdepth 5 -name 'Kimi-K3' -type d 2>/dev/null | while read d; do echo \"\$(ls \$d/*.safetensors 2>/dev/null | wc -l) \$(dirname \$d)\"; done"]
    volumeMounts: [{name: h, mountPath: /host, readOnly: true}]
  volumes:
  - {name: h, hostPath: {path: /, type: Directory}}
EOF
    i=0
    while [ $i -lt 60 ]; do
        ph=$(kubectl -n infera get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
        case "$ph" in Succeeded|Failed) break;; esac
        i=$((i+1)); sleep 3
    done
    out=$(kubectl -n infera logs "$POD" 2>/dev/null)
    kubectl -n infera delete pod "$POD" --ignore-not-found >/dev/null 2>&1
    if [ -z "$out" ]; then
        echo "no Kimi-K3 directory found on $NODE under /mnt"
        exit 1
    fi
    echo "Kimi-K3 directories on $NODE:"
    echo "$out" | sed 's|/host||' | while read n dir; do
        printf '  %-4s shards  %s\n' "$n" "$dir"
    done
    echo
    # Deliberately NOT reporting the filesystem here. Through a hostPath mount of
    # / the node's submounts are not propagated, so df inside this pod reports the
    # root filesystem for every path and would mislabel a local array as network
    # storage -- steering you away from the right answer. The full check below
    # mounts the candidate directly, where df is correct.
    echo "This lists candidates only. Several of these are usually network mounts,"
    echo "which look identical here and cost ~95 min of weight load. Confirm each:"
    echo "  $0 $NODE <one of the above>"
    exit 0
fi

NODE="${1:?usage: preflight.sh <NODE> <MODEL_DIR> [--dspark]   |   preflight.sh <NODE> --find}"
MODEL_DIR="${2:?usage: preflight.sh <NODE> <MODEL_DIR> [--dspark]}"
WANT_DRAFT="${3:-}"
NS=infera
FAIL=0
say() { printf '%s %s\n' "$1" "$2"; }
ok()   { say "  OK  " "$1"; }
bad()  { say "  FAIL" "$1"; FAIL=1; }

# --- cluster-level -----------------------------------------------------------
kubectl get ns "$NS" >/dev/null 2>&1 \
  && ok "namespace $NS exists" \
  || bad "namespace $NS missing — kubectl create namespace $NS"

kubectl get crd inferadeployments.infera.amd.com >/dev/null 2>&1 \
  && ok "InferaDeployment CRD installed" \
  || bad "CRD missing — install the operator (see section 2)"

# --- node --------------------------------------------------------------------
alloc=$(kubectl get node "$NODE" -o jsonpath='{.status.allocatable.amd\.com/gpu}' 2>/dev/null)
if [ -z "$alloc" ]; then
  bad "node $NODE not found, or advertises no amd.com/gpu"
else
  # ALL namespaces, and not only Running: a pod in another namespace holds the GPU
  # just as hard, and one still ContainerCreating has already had it assigned.
  # Scoping this to $NS reported "8 of 8 free" on a node with nothing free.
  used=$(kubectl get pods --all-namespaces --field-selector spec.nodeName="$NODE" \
          -o jsonpath='{range .items[?(@.status.phase!="Succeeded")]}{range .spec.containers[*]}{.resources.requests.amd\.com/gpu}{"\n"}{end}{end}' 2>/dev/null \
        | awk '$1 ~ /^[0-9]+$/ {s+=$1} END{print s+0}')
  free=$((alloc - used))
  [ "$free" -ge 8 ] \
    && ok "node $NODE: $free of $alloc GPUs free" \
    || bad "node $NODE: only $free of $alloc GPUs free, need 8"
fi

# --- the model directory, ON THAT NODE ---------------------------------------
# A pod is the only way to see the node's filesystem from here. It mounts the
# PARENT and looks inside, so nothing is created: an earlier version used
# hostPath type DirectoryOrCreate on MODEL_DIR itself, which meant checking a
# typo'd path CREATED it -- and a later `hostPath type: Directory` deploy then
# succeeded on that empty directory, turning a fast mount failure into the
# 12-minute crashloop this script exists to prevent. A check must not make the
# thing it checks more broken.
POD=preflight-$$
kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1
cat <<EOF | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: {name: $POD, namespace: $NS}
spec:
  nodeSelector: {kubernetes.io/hostname: $NODE}
  restartPolicy: Never
  tolerations: [{operator: Exists}]
  containers:
  - name: p
    image: busybox
    command: ["sh","-c","B=/m$MODEL_DIR; if [ ! -d \$B ]; then echo NOPATH; exit 0; fi; for d in Kimi-K3 Kimi-K3-DSpark; do if [ -d \$B/\$d ]; then echo \"HAVE \$d \$(ls \$B/\$d/*.safetensors 2>/dev/null | wc -l) \$(du -sm \$B/\$d 2>/dev/null | awk '{print \$1}')\"; else echo \"MISS \$d\"; fi; done; echo \"FSTYPE \$(df -PT \$B | tail -1 | awk '{print \$2}')\"; echo \"SOURCE \$(df -P \$B | tail -1 | awk '{print \$1}')\""]
    volumeMounts: [{name: m, mountPath: /m, readOnly: true}]
  volumes:
  - {name: m, hostPath: {path: /, type: Directory}}
EOF
i=0
while [ $i -lt 40 ]; do
  ph=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  case "$ph" in Succeeded|Failed) break;; esac
  i=$((i+1)); sleep 3
done
out=$(kubectl -n "$NS" logs "$POD" 2>/dev/null)
kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1

if [ "$out" = "NOPATH" ]; then
  bad "$MODEL_DIR does not exist on $NODE"
elif [ -z "$out" ]; then
  bad "could not inspect $MODEL_DIR on $NODE (probe pod produced no output)"
else
  n=$(echo "$out" | awk '/^HAVE Kimi-K3 /{print $3}')
  mb=$(echo "$out" | awk '/^HAVE Kimi-K3 /{print $4}')
  if [ -z "$n" ]; then
    bad "$MODEL_DIR/Kimi-K3 not present on $NODE — this is the case that mounts fine and crashloops later"
  elif [ "$n" -lt 90 ]; then
    bad "$MODEL_DIR/Kimi-K3 on $NODE has only $n safetensors shards (expected 96) — incomplete copy"
  elif [ "${mb:-0}" -lt 100000 ]; then
    # Counting files is not enough: a failed or in-flight download leaves the
    # right number of zero-byte or partial shards. Kimi-K3 is ~1.4 TB.
    bad "$MODEL_DIR/Kimi-K3 on $NODE has $n shards but only ${mb} MB — incomplete or still downloading (expect ~1400000 MB)"
  else
    ok "$MODEL_DIR/Kimi-K3 on $NODE: $n shards, ${mb} MB"
  fi

  if [ "$WANT_DRAFT" = "--dspark" ]; then
    # $d is a COUNT, not a presence flag. An earlier version tested [ -n "$d" ],
    # which is true for the string "0" -- so an empty draft directory passed, in
    # exactly the case this script exists to catch.
    d=$(echo "$out" | awk '/^HAVE Kimi-K3-DSpark /{print $3}')
    if [ -z "$d" ]; then
      bad "$MODEL_DIR/Kimi-K3-DSpark missing on $NODE — required for the speculative combinations"
    elif [ "$d" -lt 1 ]; then
      bad "$MODEL_DIR/Kimi-K3-DSpark on $NODE exists but holds no safetensors — an interrupted download leaves exactly this"
    else
      ok "$MODEL_DIR/Kimi-K3-DSpark on $NODE: $d shard(s)"
    fi
  fi

  fs=$(echo "$out" | awk '/^FSTYPE/{print $2}')
  src=$(echo "$out" | awk '/^SOURCE/{print $2}')
  case "$fs" in
    nfs*|cifs|fuse*|9p)
      bad "$MODEL_DIR on $NODE is $fs ($src) — a network mount. Weight load goes from ~8 min to ~95 min and exceeds the ready timeout" ;;
    "") bad "could not determine the filesystem type of $MODEL_DIR on $NODE" ;;
    *)  ok "$MODEL_DIR on $NODE is local ($fs, $src)" ;;
  esac
fi

echo
[ "$FAIL" = 0 ] && echo "pre-flight: PASS" || echo "pre-flight: FAIL — fix the failures above before deploying. Each of them otherwise surfaces 12-14 minutes in, as something else."
exit "$FAIL"
