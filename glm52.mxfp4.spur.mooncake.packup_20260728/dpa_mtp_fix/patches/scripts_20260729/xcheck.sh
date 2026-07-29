#!/bin/bash
# xcheck.sh -- differential ("对拍") harness for the GLM-5.2 DPA+MTP+PD debug.
#
# Two deployments are kept alive side by side:
#
#   GRAPH   job 9005 (prefill) / 9006 (decode)   -- CUDA graphs ON, the failing config
#   EAGER   job 10922 (prefill) / 10923 (decode) -- --disable-cuda-graph, the known-good baseline
#
# The eager pair is the CONTROL. Any behaviour that reproduces on BOTH is not a
# graph problem and the diagnosis must move elsewhere. Any behaviour that
# appears only on GRAPH is a genuine graph-path divergence.
#
# Usage:
#   ./xcheck.sh probe [n]     send n (default 4) short requests to BOTH, compare
#   ./xcheck.sh long          send one 512-token request to BOTH
#   ./xcheck.sh health        health of all endpoints
#   ./xcheck.sh stacks graph  py-spy every DP rank on the GRAPH decode leg
#   ./xcheck.sh stacks eager  py-spy every DP rank on the EAGER decode leg
#
# Design note: this exists because a single passing run proves nothing about a
# race. Every claim in this investigation must be reproduced on both sides
# before it is written down as a finding.

set -u

# -- endpoints ---------------------------------------------------------------
G_PJOB=9005;  G_DJOB=9006
G_PIP=10.245.157.105; G_DIP=10.245.146.21; G_ROUTER=8031

E_PJOB=10922; E_DJOB=10923
E_PIP=10.245.158.92;  E_DIP=10.245.149.64;  E_ROUTER=8041

export DOCKER_CONFIG=/tmp/dockercfg

# Run a command inside the dbg2 container on a given spur job.
inj() {  # inj <job> <cmd>
  spur exec "$1" bash -c "docker exec dbg2 bash -c '$2'" 2>&1 | grep -v libtinfow
}

PROMPT='The capital of France is'

req() {  # req <job> <ip> <port> <ntok> <tag>
  local job=$1 ip=$2 port=$3 ntok=$4 tag=$5
  local body="{\\\"text\\\":\\\"$PROMPT\\\",\\\"sampling_params\\\":{\\\"max_new_tokens\\\":$ntok,\\\"temperature\\\":0}}"
  # One line per remote command; see the note in `health`.
  inj "$job" "curl -s -m 200 -X POST http://$ip:$port/generate -H \"Content-Type: application/json\" -d \"$body\" -o /tmp/xc_$tag.json -w \"$tag http=%{http_code} t=%{time_total}\n\""
  inj "$job" "python3 -c \"import json;d=json.load(open(\\\"/tmp/xc_$tag.json\\\"));m=d.get(\\\"meta_info\\\",{});print(\\\"   text=\\\",repr(d.get(\\\"text\\\",\\\"\\\"))[:70]);print(\\\"   tok=\\\",m.get(\\\"completion_tokens\\\"),\\\"accept_len=\\\",m.get(\\\"spec_accept_length\\\"),\\\"dp_rank=\\\",m.get(\\\"dp_rank\\\"))\" 2>&1 | tail -3"
}

case "${1:-help}" in

health)
  # NB: keep each remote command on ONE line -- backslash continuations do not
  # survive the spur -> docker -> bash nesting and silently truncate the URL.
  echo "== GRAPH (9005/9006, cuda graph ON) =="
  inj $G_PJOB "curl -s -m 8 -o /dev/null -w \"prefill=%{http_code} \" http://$G_PIP:30000/health; curl -s -m 8 -o /dev/null -w \"decode=%{http_code} \" http://$G_DIP:30000/health; curl -s -m 8 -o /dev/null -w \"router=%{http_code}\n\" http://$G_PIP:$G_ROUTER/health"
  echo "== EAGER (10922/10923, --disable-cuda-graph) =="
  inj $E_PJOB "curl -s -m 8 -o /dev/null -w \"prefill=%{http_code} \" http://$E_PIP:30000/health; curl -s -m 8 -o /dev/null -w \"decode=%{http_code} \" http://$E_DIP:30000/health; curl -s -m 8 -o /dev/null -w \"router=%{http_code}\n\" http://$E_PIP:$E_ROUTER/health"
  ;;

probe)
  N=${2:-4}
  echo "== EAGER control: $N x 24 tokens =="
  for i in $(seq 1 "$N"); do req $E_PJOB $E_PIP $E_ROUTER 24 "e$i"; done
  echo
  echo "== GRAPH under test: $N x 24 tokens =="
  for i in $(seq 1 "$N"); do req $G_PJOB $G_PIP $G_ROUTER 24 "g$i"; done
  ;;

long)
  echo "== EAGER control: 512 tokens =="
  req $E_PJOB $E_PIP $E_ROUTER 512 elong
  echo
  echo "== GRAPH under test: 512 tokens =="
  req $G_PJOB $G_PIP $G_ROUTER 512 glong
  ;;

stacks)
  case "${2:-graph}" in
    graph) JOB=$G_DJOB ;;
    eager) JOB=$E_DJOB ;;
    *) echo "stacks graph|eager"; exit 1 ;;
  esac
  inj "$JOB" 'for p in $(ps aux | grep "[s]glang::scheduler" | awk "{print \$2}"); do \
      printf "== %s : " $p; \
      timeout 60 py-spy dump --pid $p 2>&1 | sed -n "4,12p" | tr -s " " | tr "\n" "|"; echo; \
    done'
  ;;

*)
  sed -n '2,30p' "$0"
  ;;
esac
