# NOTE — needle-at-depth: RESOLVED (was open)

Status: **closed 2026-07-31**. Needle is **5/5** at `temperature=0.0`, ctx=262144,
prefill kvd ON. Goal item 1 is a clean pass. The history below is kept because the
mechanism matters and the wrong explanation was nearly reported.

## Final result

    PART 1 - short factual, chat template, temp=0        4/4
    PART 2 - needle at depth, ~120,000-token prompt      5/5

    depth=  5%  OK   9.22s  prompt_tok=120047  cached=120000  want=6159362  got 6159362
    depth= 25%  OK  11.52s  prompt_tok=120045  cached=120000  want=3331179  got 3331179
    depth= 50%  OK  10.55s  prompt_tok=120046  cached=120000  want=5271814  got 5271814
    depth= 75%  OK   8.26s  prompt_tok=120046  cached=120000  want=8251068  got 8251068
    depth= 95%  OK   7.09s  prompt_tok=120047  cached=120000  want=5385227  got 5385227

## History and what actually explains it

| run | legs | short | needle | failing depth |
|---|---|---|---|---|
| A | kvd OFF both, ctx 131072 | 4/4 | 4/5 | 25 % |
| B | kvd ON prefill, ctx 131072 | 4/4 | 3/5 | 5 %, 95 % |
| C | kvd ON prefill, ctx 262144, **L3 warm** | 4/4 | **5/5** | none |

The failing depth moved between A and B, so the score was never attributable to kvd.
Ruled out along the way, each by measurement rather than argument:

* **Not the GPU fault.** Run A had `faults=0` and still failed; kvd was off, so the
  hicache write-back path was never entered. Independent defects.
* **Not output truncation.** `needle_diag.py` re-ran the failures at `max_tokens` 256
  and 1024. Depth 5 % produced no 7-digit run at all at either budget.
* **Not a missing chat template.** Both legs log the detected template and the probe
  posts to `/v1/chat/completions`.
* **Not sampling.** This was the leading hypothesis — `correctness.py` forces
  `temperature=0.0` while `generation_config.json` recommends `1.0 / top_p 0.95`, and
  greedy decoding on a reasoning model is a classic degenerate-repetition trigger.
  Run C **refutes it**: same forced `temperature=0.0`, 5/5. The A/B in
  `needle_sampling.py` was never needed and was not run.

What actually changed in run C is the **KV cache state**. Every depth reports
`cached=120000` of `prompt_tok=120047`, i.e. the shared filler prefix was served from
the warm L3 store rather than re-prefilled. The failing runs re-prefilled the same
prefix through the multi-chunk path.

The NOTE's own second-order observation had already pointed here: depth 95 % returned
the exact needle in the isolated diagnostic while failing inside the graded suite at
identical prompt and temperature, the only difference being prefix-cache state. That
was the real signal, and the sampling hypothesis distracted from it.

**Left open deliberately:** *why* a re-prefill of the identical prefix degrades
retrieval at some depths is not established. It is no longer on the path to the
deliverable, and asserting a mechanism from three runs would repeat a mistake this
repo has already paid for. Do not report a cause.

## Reporting rule for the deliverable

Goal item 1: short factual **4/4**, needle **5/5** at ~120K tokens across five
depths, under `temperature=0.0`, ctx=262144, prefill kvd ON. State that the pass was
obtained with a warm L3 prefix cache (`cached=120000`), because the same suite scored
3/5 and 4/5 on cold-prefill runs and that dependence is real, measured, and
unexplained.
