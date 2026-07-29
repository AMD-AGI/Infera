#!/bin/bash
# Repeat the conc=128 / 512-token stress N times against the GRAPH-side router
# and report, per round: HTTP code histogram, degenerate-output count, and the
# decode leg's health afterwards.
#
# Rationale: this bug class is racy -- a single passing run proves nothing. One
# round already showed 128/128 HTTP 200 but 2/128 degenerate outputs, and the
# same prompt run solo came back coherent, so the degeneration is
# concurrency-induced rather than a greedy-decoding artifact. Repeating tells us
# whether that rate is stable, drifting, or a one-off.
set -u

ROUNDS="${1:-4}"
CONC="${2:-128}"
NTOK="${3:-512}"

PJOB=9005; PIP=10.245.157.105; ROUTER=8031
DJOB=9006; DIP=10.245.146.21
export DOCKER_CONFIG=/tmp/dockercfg

for r in $(seq 1 "$ROUNDS"); do
  echo "===== round $r/$ROUNDS  (conc=$CONC ntok=$NTOK) ====="

  spur exec $PJOB bash -c "docker exec dbg2 bash -c 'rm -f /tmp/sr_*.json; for i in \$(seq 1 $CONC); do curl -s -m 400 -X POST http://$PIP:$ROUTER/generate -H \"Content-Type: application/json\" -d \"{\\\"text\\\":\\\"Explain quantum computing in detail, part \$i.\\\",\\\"sampling_params\\\":{\\\"max_new_tokens\\\":$NTOK,\\\"temperature\\\":0}}\" -o /tmp/sr_\$i.json -w \"%{http_code} \" & done; wait; echo' 2>&1 | tr ' ' '\n' | grep -E '^[0-9]{3}$' | sort | uniq -c | sed 's/^/   http /'" 2>&1 | grep -v libtinfow

  spur exec $PJOB bash -c "docker exec dbg2 bash -c 'python3 /home/yihou/glm52_fix/qcheck2.py'" 2>&1 | grep -v libtinfow | sed 's/^/   /'

  spur exec $DJOB bash -c "docker exec dbg2 bash -c 'printf \"   sched=%s exc=%s kvterr=%s maxrun=%s\n\" \$(ps aux|grep -c \"[s]glang::scheduler\") \$(strings /tmp/pd_decode_30000.log | grep -ac \"Scheduler hit an exception\") \$(strings /tmp/pd_decode_30000.log | grep -ac KVTransferError) \"\$(strings /tmp/pd_decode_30000.log | grep -oaE \"#running-req: [0-9]+\" | sort -t: -k2 -n | tail -1 | grep -oE \"[0-9]+\$\")\"'" 2>&1 | grep -v libtinfow
  echo
done
