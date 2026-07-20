#!/usr/bin/env bash
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# gaie-smoke.sh — end-to-end smoke test for the GAIE (Gateway API Inference
# Extension) path of an Infera InferaDeployment.
#
# It applies a InferaDeployment with spec.gaie.enabled=true, waits for the
# operator to report .status.gaie.ready=true (EPP ready + InferencePool ready +
# HTTPRoute Accepted), then drives an inference request through the Inference
# Gateway and checks for a well-formed OpenAI response.
#
# Readiness is gated on the operator's own GAIE status roll-up, so this also
# exercises the status-backfill controller path.
#
# Prerequisites (NOT installed here):
#   1. infera-operator running (deploy/scripts/deploy-k8s.sh) with the GAIE-aware CRD.
#   2. Gateway API + GAIE CRDs and a Gateway (default name: inference-gateway)
#      installed in $GATEWAY_NAMESPACE (see dynamo-amd
#      docs/kubernetes/inference-gateway.md for the kGateway setup).
#   3. AMD GPUs (amd.com/gpu) for the worker pods, and a pullable engine image.
#
# Usage:
#   deploy/scripts/gaie-smoke.sh                      # apply sample, wait, curl, keep
#   IDEP_FILE=my-idep.yaml deploy/scripts/gaie-smoke.sh # use your own CR
#   CLEANUP=true deploy/scripts/gaie-smoke.sh         # delete the IDEP at the end
#   SKIP_APPLY=true deploy/scripts/gaie-smoke.sh      # only verify an existing IDEP
#   deploy/scripts/gaie-smoke.sh --help
#
set -euo pipefail

# ---- defaults (override via env) -------------------------------------------
NAMESPACE="${NAMESPACE:-default}"
IDEP_NAME="${IDEP_NAME:-demo-gaie}"
GATEWAY_NAME="${GATEWAY_NAME:-inference-gateway}"
GATEWAY_NAMESPACE="${GATEWAY_NAMESPACE:-$NAMESPACE}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"

READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # seconds to wait for status.gaie.ready
POLL_INTERVAL="${POLL_INTERVAL:-10}"

SKIP_APPLY="${SKIP_APPLY:-false}"
SKIP_CURL="${SKIP_CURL:-false}"
CLEANUP="${CLEANUP:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives in deploy/scripts/; the operator samples are under deploy/.
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IDEP_FILE="${IDEP_FILE:-$DEPLOY_DIR/operator/config/samples/infera_v1alpha1_inferadeployment_gaie.yaml}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

log()  { printf '\033[1;34m[gaie-smoke]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gaie-smoke]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[gaie-smoke]\033[0m %s\n' "$*" >&2; }

command -v kubectl >/dev/null || { fail "kubectl not found in PATH"; exit 1; }

jsonpath() { kubectl -n "$NAMESPACE" get inferadeployment "$IDEP_NAME" -o jsonpath="$1" 2>/dev/null || true; }

dump_diagnostics() {
  warn "---- diagnostics ----"
  kubectl -n "$NAMESPACE" get inferadeployment "$IDEP_NAME" -o yaml 2>/dev/null \
    | sed -n '/^status:/,$p' | head -60 || true
  warn "pods:"
  kubectl -n "$NAMESPACE" get pods -l "infera.amd.com/deployment=$IDEP_NAME" -o wide 2>/dev/null || true
  warn "inferencepool / httproute:"
  kubectl -n "$NAMESPACE" get inferencepool,httproute -l "infera.amd.com/deployment=$IDEP_NAME" 2>/dev/null || true
}

# ---- 0. preflight -----------------------------------------------------------
log "Preflight: CRDs + Gateway"
for crd in inferadeployments.infera.amd.com inferencepools.inference.networking.k8s.io httproutes.gateway.networking.k8s.io; do
  kubectl get crd "$crd" >/dev/null 2>&1 || { fail "missing CRD: $crd (install operator + Gateway API + GAIE)"; exit 1; }
done
if ! kubectl -n "$GATEWAY_NAMESPACE" get gateway "$GATEWAY_NAME" >/dev/null 2>&1; then
  fail "Gateway '$GATEWAY_NAME' not found in namespace '$GATEWAY_NAMESPACE'"
  exit 1
fi

# ---- 1. apply the InferaDeployment ----------------------------------------
if [[ "$SKIP_APPLY" != "true" ]]; then
  [[ -f "$IDEP_FILE" ]] || { fail "IDEP file not found: $IDEP_FILE"; exit 1; }
  log "Applying InferaDeployment from $IDEP_FILE into $NAMESPACE"
  kubectl -n "$NAMESPACE" apply -f "$IDEP_FILE"
else
  warn "SKIP_APPLY=true — verifying existing IDEP '$IDEP_NAME'"
fi

# ---- 2. wait for status.gaie.ready ------------------------------------------
log "Waiting up to ${READY_TIMEOUT}s for .status.gaie.ready=true (operator roll-up)"
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready="false"
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  ready="$(jsonpath '{.status.gaie.ready}')"
  epp="$(jsonpath '{.status.gaie.eppReadyReplicas}')"
  pool="$(jsonpath '{.status.gaie.inferencePoolReady}')"
  route="$(jsonpath '{.status.gaie.httpRouteAccepted}')"
  log "  gaie: ready=${ready:-?} eppReady=${epp:-0} pool=${pool:-?} routeAccepted=${route:-?}"
  [[ "$ready" == "true" ]] && break
  sleep "$POLL_INTERVAL"
done
if [[ "$ready" != "true" ]]; then
  fail "GAIE resources not ready within ${READY_TIMEOUT}s"
  dump_diagnostics
  exit 1
fi
log "GAIE is ready."

# ---- 3. drive an inference request through the gateway ----------------------
if [[ "$SKIP_CURL" == "true" ]]; then
  warn "SKIP_CURL=true — skipping inference request"
  log "Smoke OK (readiness only)."
  exit 0
fi

log "Port-forwarding the gateway (svc/$GATEWAY_NAME -n $GATEWAY_NAMESPACE :8000->80)"
kubectl -n "$GATEWAY_NAMESPACE" port-forward "svc/$GATEWAY_NAME" 8000:80 >/dev/null 2>&1 &
pf_pid=$!
cleanup_pf() { kill "$pf_pid" >/dev/null 2>&1 || true; }
trap cleanup_pf EXIT
sleep 3

GATEWAY_URL="http://localhost:8000"
log "GET $GATEWAY_URL/v1/models"
if ! curl -sf "$GATEWAY_URL/v1/models" | grep -q '"data"'; then
  fail "/v1/models did not return a model list"
  dump_diagnostics
  exit 1
fi

log "POST $GATEWAY_URL/v1/chat/completions (model=$MODEL_NAME)"
resp="$(curl -sf "$GATEWAY_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one word.\"}],\"max_tokens\":8,\"stream\":false}")" || {
    fail "chat/completions request failed"; dump_diagnostics; exit 1; }

if echo "$resp" | grep -q '"choices"'; then
  log "Inference response OK:"
  echo "$resp" | head -c 600; echo
else
  fail "unexpected response: $resp"
  exit 1
fi

log "Smoke OK: GAIE end-to-end (gateway -> EPP -> worker frontend -> engine)."

# ---- 4. optional cleanup ----------------------------------------------------
if [[ "$CLEANUP" == "true" ]]; then
  log "CLEANUP=true — deleting InferaDeployment '$IDEP_NAME'"
  kubectl -n "$NAMESPACE" delete inferadeployment "$IDEP_NAME" --wait=false || true
fi
