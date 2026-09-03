# Output quality of the TP4 control — 980 generations, MTP off

Scored 2026-09-03 from the per-request `generated_texts` in
`logs/tp4_control_jsonl.tar.gz`. **No GPU** — a file read. Metric is
`loopcheck.py`'s: a request is *looping* if some 10-gram repeats ≥ 5×.

This is the **largest MTP-off degeneracy dataset in the campaign** and it had
never been scored. It was nearly dropped from this packup as "not cited by any
claim".

## Result

```
980 generations, looping = some 10-gram repeats >= 5x

  ALL              n= 980  looping= 268 (27.3%)  tokens-in-looping=49.9%  worst=x2861

  osl 0-320        n= 490  looping=   0 ( 0.0%)  tokens-in-looping= 0.0%  worst=x4
  osl 1000-3300    n= 490  looping= 268 (54.7%)  tokens-in-looping=54.7%  worst=x2861

worst: p90_c24#186  osl=3300  max_repeat=x2861  unique_word_ratio=0.058
       p90_c24#159  osl=3300  max_repeat=x1712  unique_word_ratio=0.157
       p90_c8#8     osl=3300  max_repeat=x1283  unique_word_ratio=0.171
```

All 8 source files carry `accept_length: null` — **MTP was off on every one.**

## Three things this establishes

**1. The p50 arm is clean, measured, at n=490 — not assumed.**
`0/490 looping, worst x4` at osl 320. The alignment headline is carried mainly by
p50, and its output quality is now a measurement rather than a hope.

**2. The p90 arm is contaminated at 54.7 %, so p90 throughput is an upper bound.**
A repetition loop is trivially predictable. The numbers remain a valid
measurement *of the engine generating repetitive text*; they are not a quality
result. **Direction is knowable, magnitude is not, and is deliberately not
estimated.**

**3. It corroborates the PD packup's arm A at 49× the sample size, and closes out
the ROCm-argmax hypothesis.** Arm A measured **60 % on n=10** under the same
condition (MTP off, osl 3300). This says **54.7 % on n=490**. And
`repetition_measured.md` measured **54 %** at p90 conc 8 with **MTP ON**.

| condition | osl 3300 looping | n |
|---|---:|---:|
| MTP **on** (earlier MIX arms) | 54 % | 80 |
| MTP **off** (PD arm A) | 60 % | 10 |
| **MTP off (this arm)** | **54.7 %** | **490** |

**Turning MTP off does not change the looping rate.** The ROCm silent-greedy
fallback in EAGLE verify (`eagle_utils.py:726`) is therefore **not the cause** —
it cannot be, since the rate is the same when that path never executes.

## What remains, INFERRED and untested

The prompts are themselves repetitive by construction:
`benchmark/datasets/random.py:130-134` reaches a long ISL by **repeating one
ShareGPT conversation ~50× and truncating**. A model continuing a highly
repetitive prompt with repetitive output may not be a defect at all.

**The test:** one run at osl 3300 with a *naturally* long prompt rather than a
repetition-padded one. If clean, the whole phenomenon is an artifact of how we
synthesise prompts and no upstream ROCm PR is on the critical path.

## Reproduce

```bash
tar xzf logs/tp4_control_jsonl.tar.gz -C /tmp
cd /tmp/jsonl/fixlen && python3 -c "
import json,glob
out=open('/tmp/all.jsonl','w')
for f in sorted(glob.glob('*.jsonl')):
    d=json.load(open(f)); tag=f.replace('big_','').replace('.jsonl','')
    for i,(t,o) in enumerate(zip(d['generated_texts'], d['output_lens'])):
        out.write(json.dumps({'request_id':f'{tag}#{i}','text':t,'completion_tokens':o})+'\n')
"
python3 ../glm53.big.mxfp4.pd.packup_20260903/scripts/loopcheck.py /tmp/all.jsonl
```

**Convert first.** Fed a raw `--output-details` file, `loopcheck.py` scores an
empty string and reports `looping = 0 (0.0%)` — a clean bill of health from a
parser that never saw the text.
