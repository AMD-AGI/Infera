#!/usr/bin/env bash
# collect_env.sh — snapshot the hardware + software environment for a reproduction kit.
#
# Run this on EACH machine that took part in the experiment, and paste the output
# into the packup's environment.md (one section per host). It is read-only and
# safe: it inspects state, changes nothing, and prints NO secret values.
#
# Usage:
#   ./collect_env.sh                       # inspect the host
#   ./collect_env.sh <image_tag>           # also resolve a docker image's digest
#   ssh chi2811 'bash -s' < collect_env.sh # run remotely, capture output locally
#
# Everything is best-effort: missing tools are reported, never fatal. GPU probing
# tries ROCm (rocm-smi) first, then NVIDIA (nvidia-smi).

set -u
IMAGE_TAG="${1:-}"

hr(){ printf '\n===== %s =====\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1; }
try(){ if have "$1"; then shift; "$@" 2>&1; else echo "($1 not present)"; fi; }

hr "HOST / TIME"
echo "hostname : $(hostname 2>/dev/null)"
echo "date     : $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "uptime   : $(uptime 2>/dev/null)"
echo "kernel   : $(uname -a 2>/dev/null)"
if [ -r /etc/os-release ]; then . /etc/os-release; echo "os       : ${PRETTY_NAME:-unknown}"; fi

hr "CPU / MEM"
if have lscpu; then lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\))' ; else echo "(lscpu not present)"; fi
if have free; then free -h; fi

hr "GPU"
if have rocm-smi; then
  echo "-- rocm-smi --"
  rocm-smi --showproductname --showdriverversion 2>&1 | grep -viE 'warn|===' | sed '/^$/d'
  echo "-- count --"
  rocm-smi -i 2>/dev/null | grep -ciE 'GPU\[' || echo "?"
  echo "-- mem/util snapshot --"
  rocm-smi --showmeminfo vram --showuse 2>&1 | grep -viE 'warn|===' | sed '/^$/d' | head -40
elif have nvidia-smi; then
  echo "-- nvidia-smi --"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
else
  echo "(no rocm-smi or nvidia-smi)"
fi

hr "RDMA / FABRIC"
echo "-- ibv_devices --"
try ibv_devices ibv_devices
echo "-- rail link state (ibstat / ibv_devinfo) --"
if have ibstat; then ibstat 2>&1 | grep -E 'CA |State|Rate|Physical|Link layer' | head -60
elif have ibv_devinfo; then ibv_devinfo 2>&1 | grep -E 'hca_id|state|link_layer|active_width|active_speed' | head -60
else echo "(no ibstat/ibv_devinfo)"; fi
echo "-- rdma link --"
try rdma rdma link show

hr "DOCKER"
if have docker; then
  echo "-- version --"; docker version --format '{{.Server.Version}}' 2>&1
  echo "-- running containers --"; docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1 | head -20
  if [ -n "$IMAGE_TAG" ]; then
    echo "-- image digest for $IMAGE_TAG --"
    docker image inspect "$IMAGE_TAG" --format '{{.Id}}{{"\n"}}{{range .RepoDigests}}{{.}}{{"\n"}}{{end}}' 2>&1
  else
    echo "(no image tag passed; re-run as ./collect_env.sh <image_tag> to resolve base digest)"
  fi
else
  echo "(docker not present on this host)"
fi

hr "GIT (run from inside the repo the experiment used)"
if have git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "branch : $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "commit : $(git rev-parse HEAD 2>/dev/null)"
  echo "descr  : $(git describe --always --dirty --tags 2>/dev/null)"
  echo "-- dirty? (uncommitted files matter for repro) --"
  git status --porcelain 2>/dev/null | head -40
else
  echo "(not inside a git work tree — cd into the repo and re-run, or record branch+SHA by hand)"
fi

hr "DONE"
echo "Paste this block into environment.md under a heading for host $(hostname 2>/dev/null)."
