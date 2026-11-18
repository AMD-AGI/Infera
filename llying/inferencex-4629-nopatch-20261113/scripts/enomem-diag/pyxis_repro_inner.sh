#!/usr/bin/env bash
# Inside a pyxis container like the workers': limits, identity, then the decode-layout registration on GPUs 0-3.
D=/apps/liyingli/holder-infx/diag; O=$D/out/$(hostname -s)-pyxis-$(date -u +%m%dT%H%M); mkdir -p $O
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
echo "id: $(id) | memlock soft=$(ulimit -Sl) hard=$(ulimit -Hl)"; grep -E "CapEff|CapBnd" /proc/self/status; cat /proc/self/uid_map
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
run() {  # NAME
    for g in 0 1 2 3; do
        python3 $D/kfd_pin_repro.py --gpu $g --alloc kfd --reg peermem --buffers 84 --nics $NICS --chunk-gib 1 --threads 16 \
            --fork-init 1 --filler-gib 115 --ipc-export 1 --relaxed 1 --headroom 0 --tag $1 > $O/$1-gpu$g.jsonl 2>&1 &
    done; wait
    for g in 0 1 2 3; do
        grep -E '"step": "(registered_buffers|error)"' $O/$1-gpu$g.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print('$1 gpu$g', d['step'], 'mrs', d.get('mrs'), 'failed', d.get('failed'), 'errnos', d.get('errnos'), 'first', (d.get('first_failed') or [])[:3], d.get('what', ''))" 2>/dev/null || tail -3 $O/$1-gpu$g.jsonl
    done
}
run Q-default
ulimit -l unlimited 2>/dev/null && echo "raised memlock to $(ulimit -l)" && run Q-unlimited
