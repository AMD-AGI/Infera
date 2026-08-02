#!/bin/sh
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Pre-flight for a kimi-k3-optimized deployment.
#
#   ./preflight.sh <NODE> <MODEL_DIR> [--dspark]
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
NODE="${1:?usage: preflight.sh <NODE> <MODEL_DIR> [--dspark]}"
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
  used=$(kubectl -n "$NS" get pods --field-selector spec.nodeName="$NODE",status.phase=Running \
          -o jsonpath='{range .items[*].spec.containers[*]}{.resources.requests.amd\.com/gpu}{"\n"}{end}' 2>/dev/null \
        | awk '{s+=$1} END{print s+0}')
  free=$((alloc - used))
  [ "$free" -ge 8 ] \
    && ok "node $NODE: $free of $alloc GPUs free" \
    || bad "node $NODE: only $free of $alloc GPUs free, need 8"
fi

# --- the model directory, ON THAT NODE ---------------------------------------
# A pod is the only way to see the node's filesystem from here. hostPath type
# DirectoryOrCreate rather than Directory: a missing path must be reported by
# this script, not turned into a pod that fails to schedule.
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
    command: ["sh","-c","for d in Kimi-K3 Kimi-K3-DSpark; do if [ -d /m/\$d ]; then echo \"HAVE \$d \$(ls /m/\$d/*.safetensors 2>/dev/null | wc -l)\"; else echo \"MISS \$d\"; fi; done; echo \"FSTYPE \$(df -PT /m | tail -1 | awk '{print \$2}')\"; echo \"SOURCE \$(df -P /m | tail -1 | awk '{print \$1}')\""]
    volumeMounts: [{name: m, mountPath: /m}]
  volumes:
  - {name: m, hostPath: {path: $MODEL_DIR, type: DirectoryOrCreate}}
EOF
i=0
while [ $i -lt 40 ]; do
  ph=$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null)
  case "$ph" in Succeeded|Failed) break;; esac
  i=$((i+1)); sleep 3
done
out=$(kubectl -n "$NS" logs "$POD" 2>/dev/null)
kubectl -n "$NS" delete pod "$POD" --ignore-not-found >/dev/null 2>&1

if [ -z "$out" ]; then
  bad "could not inspect $MODEL_DIR on $NODE (probe pod produced no output)"
else
  n=$(echo "$out" | awk '/^HAVE Kimi-K3 /{print $3}')
  if [ -z "$n" ]; then
    bad "$MODEL_DIR/Kimi-K3 not present on $NODE — this is the case that mounts fine and crashloops later"
  elif [ "$n" -lt 90 ]; then
    bad "$MODEL_DIR/Kimi-K3 on $NODE has only $n safetensors shards (expected 96) — incomplete copy"
  else
    ok "$MODEL_DIR/Kimi-K3 on $NODE: $n shards"
  fi

  if [ "$WANT_DRAFT" = "--dspark" ]; then
    d=$(echo "$out" | awk '/^HAVE Kimi-K3-DSpark /{print $3}')
    [ -n "$d" ] \
      && ok "$MODEL_DIR/Kimi-K3-DSpark on $NODE present" \
      || bad "$MODEL_DIR/Kimi-K3-DSpark missing on $NODE — required for the speculative combinations"
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
[ "$FAIL" = 0 ] && echo "pre-flight: PASS" || echo "pre-flight: FAIL — fix the above before deploying; the failures below show up 12-14 minutes in, as something else"
exit "$FAIL"
