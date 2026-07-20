#!/usr/bin/env bash
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# deploy-k8s.sh — one-shot Helm installer for the Infera Kubernetes dependencies.
#
# Installs only the platform / infrastructure Infera relies on (no inference
# workload — deploy those separately with an InferaDeployment CR). Each step is
# an idempotent `helm upgrade --install` (or `kubectl apply` for CRDs):
#
#   1. LeaderWorkerSet (LWS) controller  — required for multi-node TP workers
#      (numberOfNodes > 1). Upstream OCI chart.
#   2. NATS (JetStream)                  — shared broker for the KV-event plane
#      and the per-instance request transport. Upstream chart. (Optional: each
#      InferaDeployment can instead provision its own NATS via
#      spec.nats.deploy=true; skip this with INSTALL_NATS=false.)
#   3. infera-operator                  — reconciles InferaDeployment CRs.
#      Local chart (deploy/operator/helm/infera-operator), ships the CRD + RBAC.
#   4. Inference Gateway (kgateway, optional, off) — the GAIE gateway stack:
#      Gateway API CRDs + GAIE inference-extension CRDs + kgateway controller
#      (control + data plane) + (optional) an `inference-gateway` Gateway.
#      Enable with INSTALL_GATEWAY=true. Charts/images default to upstream
#      (cr.kgateway.dev); override KGATEWAY_* to use a Harbor / docker.io mirror.
#
# Everything is configurable via env vars (see DEFAULTS below) and each step can
# be toggled, so you can re-run safely or install only what you need. To run an
# actual inference service, apply an InferaDeployment afterwards (see
# examples/k8s-deployments/).
#
# Prerequisites this script does NOT install (set them up before applying a
# InferaDeployment):
#   1. AMD GPU device plugin — the cluster must expose GPUs as the extended
#      resource `amd.com/gpu` (AMD GPU Operator / device plugin). Without it
#      worker pods stay Pending. This is the real hard prerequisite; note etcd
#      is NOT required (the default discoveryBackend is kubernetes, zero etcd).
#   2. Engine image — the `infera-sglang|vllm` image referenced by your IDEP
#      must be pullable by the nodes (from your registry / a mirror).
#   3. Model data — the model path must be reachable inside the pod: mount it via
#      `extraPodSpec` (hostPath/PVC), bake it into the image, or let the engine
#      download it from HuggingFace (HF_HOME / token).
#   4. imagePullSecret — for a private registry, create the pull secret in the
#      namespace where you apply the InferaDeployment (it is per-namespace).
#
# Usage:
#   deploy/scripts/deploy-k8s.sh                 # install infra (LWS + NATS + operator)
#   INSTALL_NATS=false deploy/scripts/deploy-k8s.sh
#   INFERA_NAMESPACE=infera OPERATOR_IMAGE_TAG=0.1.3 deploy/scripts/deploy-k8s.sh
#   # NATS JetStream PVC on a cluster without a default StorageClass:
#   NATS_STORAGE_CLASS=local-path NATS_STORAGE_SIZE=10Gi deploy/scripts/deploy-k8s.sh
#   # infra + the inference gateway (kgateway) from upstream:
#   INSTALL_GATEWAY=true GATEWAY_NS=infera deploy/scripts/deploy-k8s.sh
#   # everything from a mirror (replace docker.io/inferaimage with your registry):
#   INSTALL_GATEWAY=true \
#     KGATEWAY_CRDS_CHART=oci://registry-1.docker.io/inferaimage/kgateway-crds \
#     KGATEWAY_CHART=oci://registry-1.docker.io/inferaimage/kgateway \
#     KGATEWAY_IMAGE_REGISTRY=docker.io/inferaimage deploy/scripts/deploy-k8s.sh
#   deploy/scripts/deploy-k8s.sh --help
#
set -euo pipefail

# ---- defaults (override via env) -------------------------------------------
INFERA_NAMESPACE="${INFERA_NAMESPACE:-infera-system}"   # operator + shared NATS
LWS_NAMESPACE="${LWS_NAMESPACE:-lws-system}"

INSTALL_LWS="${INSTALL_LWS:-true}"
INSTALL_NATS="${INSTALL_NATS:-true}"
INSTALL_OPERATOR="${INSTALL_OPERATOR:-true}"
INSTALL_GATEWAY="${INSTALL_GATEWAY:-false}"       # GAIE gateway stack (kgateway)

# Inference Gateway (kgateway) — only used when INSTALL_GATEWAY=true.
KGATEWAY_NAMESPACE="${KGATEWAY_NAMESPACE:-kgateway-system}"
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.4.1}"   # k8s Gateway API CRDs
GAIE_CRD_VERSION="${GAIE_CRD_VERSION:-v1.5.0}"         # GAIE inference-extension CRDs (tested)
KGATEWAY_VERSION="${KGATEWAY_VERSION:-v2.1.1}"
# Charts default to the upstream OCI registry; override to a mirror, e.g.
# oci://registry-1.docker.io/inferaimage/kgateway[-crds].
KGATEWAY_CRDS_CHART="${KGATEWAY_CRDS_CHART:-oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds}"
KGATEWAY_CHART="${KGATEWAY_CHART:-oci://cr.kgateway.dev/kgateway-dev/charts/kgateway}"
# Empty => chart default registry (cr.kgateway.dev/kgateway-dev). Set to a
# mirror prefix (e.g. docker.io/inferaimage) to pull the controller AND the
# data-plane (envoy-wrapper/sds) images from there — kgateway derives the
# data-plane registry from image.registry.
KGATEWAY_IMAGE_REGISTRY="${KGATEWAY_IMAGE_REGISTRY:-}"
# CRD manifests (plain kubectl apply, not helm). Override with a local file
# path (file://… or a downloaded copy) for air-gapped clusters.
GATEWAY_API_MANIFEST="${GATEWAY_API_MANIFEST:-https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml}"
GAIE_CRD_MANIFEST="${GAIE_CRD_MANIFEST:-https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$GAIE_CRD_VERSION/manifests.yaml}"
# Optionally create an `inference-gateway` Gateway (gatewayClassName=kgateway).
CREATE_GATEWAY="${CREATE_GATEWAY:-true}"
GATEWAY_NAME="${GATEWAY_NAME:-inference-gateway}"
GATEWAY_NS="${GATEWAY_NS:-infera}"
GATEWAY_MANIFEST="${GATEWAY_MANIFEST:-https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/$GAIE_CRD_VERSION/config/manifests/gateway/kgateway/gateway.yaml}"

# LWS upstream OCI Helm chart.
LWS_CHART="${LWS_CHART:-oci://registry.k8s.io/lws/charts/lws}"
LWS_VERSION="${LWS_VERSION:-0.9.0}"                # tested
LWS_RELEASE="${LWS_RELEASE:-lws}"

# NATS upstream Helm chart.
NATS_REPO_NAME="${NATS_REPO_NAME:-nats}"
NATS_REPO_URL="${NATS_REPO_URL:-https://nats-io.github.io/k8s/helm/charts/}"
NATS_CHART="${NATS_CHART:-nats/nats}"
NATS_VERSION="${NATS_VERSION:-}"                 # empty = latest in the repo
NATS_RELEASE="${NATS_RELEASE:-infera-nats}"
# JetStream PVC knobs (applied via --set; the chart config is fully inline, no
# local values file).
NATS_STORAGE_SIZE="${NATS_STORAGE_SIZE:-10Gi}"
# Empty = cluster default StorageClass; set to e.g. local-path on clusters
# without a default SC (otherwise the JetStream PVC stays Pending).
NATS_STORAGE_CLASS="${NATS_STORAGE_CLASS:-}"

# infera-operator (local chart relative to this script).
OPERATOR_RELEASE="${OPERATOR_RELEASE:-infera-operator}"
OPERATOR_IMAGE_REPO="${OPERATOR_IMAGE_REPO:-}"   # empty = chart default
OPERATOR_IMAGE_TAG="${OPERATOR_IMAGE_TAG:-}"     # empty = chart default
HELM_TIMEOUT="${HELM_TIMEOUT:-5m}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives in deploy/scripts/; charts and manifests are under deploy/.
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPERATOR_CHART="${OPERATOR_CHART:-$DEPLOY_DIR/operator/helm/infera-operator}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,58p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

log()  { printf '\033[1;34m[deploy-k8s]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy-k8s]\033[0m %s\n' "$*"; }

command -v helm    >/dev/null || { echo "helm not found in PATH" >&2; exit 1; }
command -v kubectl >/dev/null || { echo "kubectl not found in PATH" >&2; exit 1; }

# ---- 1. LeaderWorkerSet controller -----------------------------------------
if [[ "$INSTALL_LWS" == "true" ]]; then
  log "Installing LeaderWorkerSet controller ($LWS_CHART:$LWS_VERSION) into $LWS_NAMESPACE"
  helm upgrade --install "$LWS_RELEASE" "$LWS_CHART" \
    --version "$LWS_VERSION" \
    --namespace "$LWS_NAMESPACE" --create-namespace \
    --wait --timeout "$HELM_TIMEOUT"
else
  warn "Skipping LWS (INSTALL_LWS=$INSTALL_LWS)"
fi

# ---- 2. NATS (JetStream) ----------------------------------------------------
if [[ "$INSTALL_NATS" == "true" ]]; then
  log "Adding NATS Helm repo ($NATS_REPO_URL) and installing JetStream NATS into $INFERA_NAMESPACE"
  helm repo add "$NATS_REPO_NAME" "$NATS_REPO_URL" >/dev/null 2>&1 || true
  helm repo update "$NATS_REPO_NAME" >/dev/null
  version_flag=()
  [[ -n "$NATS_VERSION" ]] && version_flag=(--version "$NATS_VERSION")
  # Infera needs JetStream on (the upstream chart ships it OFF) with a durable
  # file store; single replica by default. All set inline — no local values file.
  # storageClass is passed only when set (empty -> cluster default SC).
  nats_flags=(
    --set "config.jetstream.enabled=true"
    --set "config.jetstream.fileStore.enabled=true"
    --set "config.jetstream.fileStore.pvc.size=$NATS_STORAGE_SIZE"
    --set "config.cluster.enabled=false"
    --set "config.cluster.replicas=1"
  )
  [[ -n "$NATS_STORAGE_CLASS" ]] && \
    nats_flags+=(--set "config.jetstream.fileStore.pvc.storageClassName=$NATS_STORAGE_CLASS")
  helm upgrade --install "$NATS_RELEASE" "$NATS_CHART" \
    "${version_flag[@]}" \
    --namespace "$INFERA_NAMESPACE" --create-namespace \
    "${nats_flags[@]}" \
    --wait --timeout "$HELM_TIMEOUT"
  log "Shared NATS available at: nats://${NATS_RELEASE}.${INFERA_NAMESPACE}.svc:4222"
  log "  -> set spec.nats.deploy=false + NATS_SERVER on InferaDeployments to use it."
else
  warn "Skipping NATS (INSTALL_NATS=$INSTALL_NATS) — InferaDeployments must set spec.nats.deploy=true or point NATS_SERVER at an existing broker"
fi

# ---- 3. infera-operator ----------------------------------------------------
if [[ "$INSTALL_OPERATOR" == "true" ]]; then
  log "Installing infera-operator (chart: $OPERATOR_CHART) into $INFERA_NAMESPACE"
  image_flags=()
  [[ -n "$OPERATOR_IMAGE_REPO" ]] && image_flags+=(--set "image.repository=$OPERATOR_IMAGE_REPO")
  [[ -n "$OPERATOR_IMAGE_TAG" ]]  && image_flags+=(--set "image.tag=$OPERATOR_IMAGE_TAG")
  helm upgrade --install "$OPERATOR_RELEASE" "$OPERATOR_CHART" \
    --namespace "$INFERA_NAMESPACE" --create-namespace \
    "${image_flags[@]}" \
    --wait --timeout "$HELM_TIMEOUT"
else
  warn "Skipping operator (INSTALL_OPERATOR=$INSTALL_OPERATOR)"
fi

# ---- 4. Inference Gateway (kgateway) ----------------------------------------
if [[ "$INSTALL_GATEWAY" == "true" ]]; then
  log "Installing Gateway API CRDs ($GATEWAY_API_VERSION)"
  kubectl apply -f "$GATEWAY_API_MANIFEST"
  log "Installing GAIE inference-extension CRDs ($GAIE_CRD_VERSION)"
  kubectl apply -f "$GAIE_CRD_MANIFEST"

  log "Installing kgateway CRDs ($KGATEWAY_CRDS_CHART:$KGATEWAY_VERSION) into $KGATEWAY_NAMESPACE"
  helm upgrade --install kgateway-crds "$KGATEWAY_CRDS_CHART" \
    --version "$KGATEWAY_VERSION" \
    --namespace "$KGATEWAY_NAMESPACE" --create-namespace \
    --wait --timeout "$HELM_TIMEOUT"

  log "Installing kgateway controller ($KGATEWAY_CHART:$KGATEWAY_VERSION) into $KGATEWAY_NAMESPACE"
  kgw_flags=(--set inferenceExtension.enabled=true)
  [[ -n "$KGATEWAY_IMAGE_REGISTRY" ]] && kgw_flags+=(--set "image.registry=$KGATEWAY_IMAGE_REGISTRY")
  helm upgrade --install kgateway "$KGATEWAY_CHART" \
    --version "$KGATEWAY_VERSION" \
    --namespace "$KGATEWAY_NAMESPACE" --create-namespace \
    "${kgw_flags[@]}" \
    --wait --timeout "$HELM_TIMEOUT"

  if [[ "$CREATE_GATEWAY" == "true" ]]; then
    log "Creating Gateway '$GATEWAY_NAME' in $GATEWAY_NS (gatewayClassName=kgateway)"
    kubectl get namespace "$GATEWAY_NS" >/dev/null 2>&1 || kubectl create namespace "$GATEWAY_NS"
    # The upstream sample names the Gateway 'inference-gateway' with class
    # 'agentgateway'; apply it then rename to our $GATEWAY_NAME / kgateway class.
    kubectl apply -n "$GATEWAY_NS" -f "$GATEWAY_MANIFEST"
    kubectl patch gateway inference-gateway -n "$GATEWAY_NS" --type=json \
      -p='[{"op":"replace","path":"/spec/gatewayClassName","value":"kgateway"}]' 2>/dev/null || true
    log "Gateway status: kubectl -n $GATEWAY_NS get gateway $GATEWAY_NAME"
  else
    warn "CREATE_GATEWAY=false — skipping Gateway resource (create one before applying a GAIE-enabled InferaDeployment)"
  fi
else
  warn "Skipping inference gateway (INSTALL_GATEWAY=$INSTALL_GATEWAY) — set INSTALL_GATEWAY=true to install kgateway + GAIE CRDs"
fi

# ---- summary ----------------------------------------------------------------
log "Done. Releases:"
helm list -n "$INFERA_NAMESPACE" 2>/dev/null || true
[[ "$INSTALL_LWS" == "true" ]] && helm list -n "$LWS_NAMESPACE" 2>/dev/null || true
[[ "$INSTALL_GATEWAY" == "true" ]] && helm list -n "$KGATEWAY_NAMESPACE" 2>/dev/null || true
log "Verify the operator + CRD:"
log "  kubectl get crd inferadeployments.infera.amd.com"
log "  kubectl -n $INFERA_NAMESPACE get deploy,pod"
if [[ "$INSTALL_GATEWAY" == "true" ]]; then
  log "Verify the gateway:"
  log "  kubectl get crd inferencepools.inference.networking.k8s.io httproutes.gateway.networking.k8s.io"
  log "  kubectl -n $KGATEWAY_NAMESPACE get deploy,pod"
  [[ "$CREATE_GATEWAY" == "true" ]] && log "  kubectl -n $GATEWAY_NS get gateway $GATEWAY_NAME"
fi
