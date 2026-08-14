#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build/push/ship one Infera engine image on a node that has docker.
# ship = login+build+push
#   build|push|ship <sglang|vllm|atom|kvd|server|overlay>
#
# `overlay` builds the base-agnostic payload (deploy/overlay/). It consumes the
# vllm and sglang images, so it must run AFTER them — release.yml gates it, and
# it publishes to two repos from one build (see the `refs` array below).
set -euo pipefail

cmd="${1:-}"
engine="${2:-}"

case "$engine" in
  sglang)  dockerfile="deploy/docker/Dockerfile.sglang" ;;
  vllm)    dockerfile="deploy/docker/Dockerfile.vllm" ;;
  atom)    dockerfile="deploy/docker/Dockerfile.atom" ;;
  kvd)     dockerfile="deploy/docker/Dockerfile.kvd" ;;
  server)  dockerfile="deploy/docker/Dockerfile.server" ;;
  overlay) dockerfile="deploy/overlay/Dockerfile.payload" ;;
  *) echo "usage: $0 <build|push|ship> <sglang|vllm|atom|kvd|server|overlay>" >&2; exit 2 ;;
esac

# Tag precedence: ID (release/nightly) > PR > local. IMAGE = target repo.
IMAGE="${IMAGE:-docker.io/inferaimage/infera}"
if   [ -n "${ID:-}" ]; then tag="${engine}-${ID}"
elif [ -n "${PR:-}" ]; then tag="${engine}-pr${PR}"
else                        tag="${engine}-local"
fi
ref="${IMAGE}:${tag}"

# The overlay ships to two places from one build, and the two have different
# lifetimes -- this is a transition, not a permanent design.
#
#   ${IMAGE}:overlay-<id>   The point of this change. Engine images publish to
#                           the private staging repo and are promoted to the
#                           public docker.io/rocm/infera after review; the
#                           overlay never entered that pipeline, so it had no
#                           reviewed public home. Same tag shape as the engines
#                           (<component>-<id>), so promotion is the same
#                           mechanical retag rather than a special case.
#
#   ${IMAGE}-overlay:<id>   Temporary. A public repo under the staging
#                           namespace, currently the only published overlay and
#                           what every recipe in this tree pulls. It goes away
#                           once a promoted rocm/infera:overlay-<id> exists and
#                           the manifests point at it.
#
# Deriving the second name from IMAGE rather than hard-coding it means
# overriding IMAGE moves both together, instead of silently publishing half a
# release to the wrong place.
#
# One build, two tags -- never two builds. The overlay harvests compiled pieces
# out of the engine images, so a rebuild is both expensive (~80 min) and not
# guaranteed to reproduce the same bits; tagging one image twice makes the two
# refs provably the same artefact, which matters when one of them is the thing
# that gets reviewed and the other is what users are already running.
refs=("$ref")
if [ "$engine" = overlay ]; then
  overlay_id="${ID:-${PR:+pr$PR}}"
  overlay_id="${overlay_id:-local}"
  refs+=("${IMAGE}-overlay:${overlay_id}")
fi

cd "$(dirname "$0")/../.."

# ---- overlay build args ------------------------------------------------------
# The overlay is not self-contained: it harvests compiled pieces (Mooncake,
# hipFile, the Rust router) out of the engine images, and builds its Python trees
# inside the STOCK vendor bases so they can be pruned down to what those bases
# lack. So it needs four refs, and getting any of them wrong is silent:
#
#   NATIVE_IMAGE            <- the vLLM engine image from THIS run, so a release
#       never ships an overlay carrying a previous release's native code. Only
#       the vLLM (CPython 3.12) tree is harvested: the SGLang (3.10) family's
#       Mooncake is COMPILED in Dockerfile.payload's `mooncake310` stage, so
#       that the HIP-transport gate is guaranteed present rather than inherited
#       from whatever the engine image happened to ship. There is deliberately
#       no SGLANG_NATIVE_IMAGE here.
#   VLLM_BASE_IMAGE / SGLANG_BASE_IMAGE <- read out of the engine Dockerfiles
#       rather than repeated here. The prune keeps exactly what the base lacks,
#       so a base ref that drifts from the one the engine image was built on
#       produces a payload that is wrong in both directions: missing packages the
#       real base does not have, and shadowing ones it does.
build_args=()
if [ "$engine" = overlay ]; then
  base_arg() {  # default value of an ARG in an engine Dockerfile
    sed -n "s/^ARG $2=//p" "$1" | head -1
  }
  vllm_base="$(base_arg deploy/docker/Dockerfile.vllm VLLM_BASE_IMAGE)"
  sglang_base="$(base_arg deploy/docker/Dockerfile.sglang SGLANG_BASE_IMAGE)"
  [ -n "$vllm_base" ]   || { echo "cannot read VLLM_BASE_IMAGE from Dockerfile.vllm" >&2; exit 1; }
  [ -n "$sglang_base" ] || { echo "cannot read SGLANG_BASE_IMAGE from Dockerfile.sglang" >&2; exit 1; }

  # Same tag scheme as above, so this run's engines are what gets harvested.
  if   [ -n "${ID:-}" ]; then esuf="${ID}"
  elif [ -n "${PR:-}" ]; then esuf="pr${PR}"
  else                        esuf="local"
  fi
  native_vllm="${NATIVE_IMAGE:-${IMAGE}:vllm-${esuf}}"

  build_args=(
    --build-arg "VLLM_BASE_IMAGE=${vllm_base}"
    --build-arg "SGLANG_BASE_IMAGE=${sglang_base}"
    --build-arg "NATIVE_IMAGE=${native_vllm}"
  )
  echo "overlay: harvest  vllm=${native_vllm}"
  echo "overlay: compile  sglang mooncake in-image (mooncake310 stage)"
  echo "overlay: deps on  vllm=${vllm_base}"
  echo "overlay: deps on  sglang=${sglang_base}"
fi

case "$cmd" in
  build)
    # BUILD_TARGET stops at one stage instead of building the whole file. CI uses
    # it to verify the overlay compiles on a PR: the final image harvests from
    # ${IMAGE}:vllm-<id>, which only exists after a release run has pushed it, but
    # every stage up to native310 needs nothing but the public vendor bases.
    # Tagging a partial build would publish a half image under a real name, so a
    # targeted build is left untagged.
    tag_args=()
    if [ -n "${BUILD_TARGET:-}" ]; then
      echo "build: --target ${BUILD_TARGET} (verify only, not tagged)"
      tag_args=(--target "$BUILD_TARGET")
    else
      for r in "${refs[@]}"; do tag_args+=(-t "$r"); done
    fi
    # --network=host: RUN steps need DNS, and these nodes resolve via 127.0.0.1
    # which a default bridge build netns can't reach.
    docker build --network=host ${build_args[@]+"${build_args[@]}"} \
                 "${tag_args[@]}" -f "$dockerfile" .
    ;;

  push)
    for r in "${refs[@]}"; do
      echo "pushing $r"
      docker push "$r"
    done
    ;;

  ship)
    # Login in a throwaway DOCKER_CONFIG (wiped on exit) so the token
    # is never written to the shared docker config or left on the node.
    DOCKER_CONFIG="$(mktemp -d)"; export DOCKER_CONFIG
    trap 'exit 143' INT TERM
    trap 'docker logout >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG"' EXIT
    if [ -n "${INFERAIMAGE_DOCKERHUB_TOKEN:-}" ]; then
      printf '%s' "$INFERAIMAGE_DOCKERHUB_TOKEN" | docker login -u inferaimage --password-stdin
      unset INFERAIMAGE_DOCKERHUB_TOKEN
    else
      docker login -u inferaimage --password-stdin
    fi
    bash "$0" build "$engine"
    bash "$0" push "$engine"
    ;;

  *)
    echo "usage: $0 <build|push|ship> <sglang|vllm|atom|kvd|server|overlay>" >&2; exit 2 ;;
esac
