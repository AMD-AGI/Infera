# Exemplar — GLM-5.3-Flash, mix mode, gfx950, TP8

Derived from a run of **this task** whose kit passed the shape check. Sanitised
per `README.md` beside this file. Where it differs from the Qwen exemplar, the
difference is the point: **the same closure, a model four times the size, TP8
instead of TP1, and a kit of the same shape.** Nothing in the task changed
between them except `--var`.

## What is the same, and that is the finding

Same four documents, same two mandatory directories, same two independent
readings of the deployment mode, same refusal to publish a filesystem path as
the model id, same teardown discipline. A standardised deploy step is one where
a reader cannot tell from the kit's *shape* which model it was.

## What legitimately differs

| | Qwen exemplar | this one |
|---|---|---|
| TP | 1, one card of eight | **8, all cards** |
| weights | ~51 GB | **~328 GB** |
| first `/health` | ~200–275 s | **~910 s** when read from a network filesystem |
| model-specific evidence | reasoning-budget probe | **AITER mHC lines per TP rank**, decode CUDA-graph line, both memory pools |

The last row is the one to imitate rather than copy: **a kit should carry the
evidence that this engine's fast path is live for this model.** For this family
that is the per-rank AITER lines; for another it will be something else. A kit
with no such evidence cannot tell a working deployment from a silently slow one.

## The `Expected output` section — abridged

> The run reproduced **if and only if step 3 prints `VERIFY: all checks passed`
> and exits 0.** That is the criterion. Everything below says what that looks
> like so a partial result can be read.

Twelve named assertions, each printing `PASS <name> -- <the value it read>`:

```
PASS  router_health                        http_status=200
PASS  router_lists_one_active_worker       1 active of 1 registered
PASS  router_reports_mode                  disagg_mode=['mixed']
PASS  worker_reports_mode                  disagg=DisaggMode.MIXED
PASS  served_name_is_published_name        ids=['<org>/<Model>']
PASS  served_name_is_not_a_filesystem_path ids=['<org>/<Model>']
PASS  completion_capital_through_router    finish_reason=stop
PASS  completion_arithmetic_through_router content='391'
PASS  aiter_mhc_pre_post_per_rank          8 lines, need >= 8 (one per TP rank)
PASS  aiter_mhc_fused_boundary_per_rank    8 lines, need >= 8
PASS  no_faults_in_worker_log              0 fault line(s)
PASS  decode_cuda_graphs_live              cuda graph: True
```

**Each assertion prints the value it read, not just its verdict.** That is what
makes a failing run diagnosable from the transcript alone, and it is why the
two per-rank counts are `>= 8` rather than `> 0`: a single line proves one rank
took the fast path and says nothing about the other seven.

## Traps this deployment recorded

1. **A per-rank count read with `tail -3` is not a count.** The fast-path lines
   appear once per TP rank, interleaved; counting them properly turned "the
   lines are there" into "8 and 8, ranks 0–7".
2. **The weight-completeness check must not use exact equality.** On-disk bytes
   exceed the index's `total_size` by the safetensors header of each shard
   (~172 KB × 62 here). A kit asserting `==` reports complete weights as
   corrupt.
3. **A GPU-reset gate that kills foreign PIDs cannot run on a shared node.** The
   PID visible in one namespace holding 0 VRAM may be another tenant's and
   invisible in yours; a `kill -9` then no-ops and the wait-for-zero loop aborts
   a bring-up on a machine that was already idle. Gate on *VRAM returning to
   baseline*, and kill only what you can attribute to your own containers.
4. **One throughput reading is not a measurement.** A single probe read 5.6×
   low while the engine's own log showed the full rate with an empty queue. If
   a kit reports a number at all, it must cross-check it against the engine's
   own counter — or not report it.

## Numbers, marked as one site's

TP8 across 8 × 288 GiB; ~328 GB of FP8 weights; first `/health` 910 s from a
network filesystem at ~440 MB/s effective; image build 4 min 44 s. The task
makes **no** throughput, latency or accuracy claim, and neither does this file.
