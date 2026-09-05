#!/bin/sh
# The runtime contract deploy_kit.layout.yaml requires. Every identifier bound on
# a shared host is a parameter with a default.
: "${E2E_KIT_RUN_TAG:=stub-$(date +%Y%m%d-%H%M%S)-$$}"
: "${E2E_KIT_PORT_BASE:=8160}"
: "${E2E_KIT_WORK_ROOT:=/mnt/m2m_nobackup/yihou/e2e_flow/stub/${E2E_KIT_RUN_TAG}}"
: "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
: "${E2E_KIT_ENGINE_EXTRA_ENV:=}"
: "${KIT_MODEL:=Qwen/Qwen3.6-27B}"
: "${KIT_CTX:=32768}"
ROUTER_PORT=$((E2E_KIT_PORT_BASE + 0))
ENGINE_PORT=$((E2E_KIT_PORT_BASE + 1))
KIT_DIR="$(cd "$(dirname "$0")" && pwd)"
