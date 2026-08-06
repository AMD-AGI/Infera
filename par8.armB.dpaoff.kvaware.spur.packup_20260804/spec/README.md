# spec/

| file | what |
|---|---|
| `par8.yaml` | **the workload as run.** md5 `968b1543155839135dc9eaf6dd142626` |
| `glm52_crxx_caseA.fix.yaml` | the Case A parent par8 derives from, for provenance |

`par8.yaml` is byte-identical to
`../../par8.glm52.dpaoff.packup_20260803/spec/par8.yaml` **except one line** — the
tokenizer path, retargeted from vultr's `/mnt/vast/xiaobo/models/...` to this
cluster's `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. Verify:

```bash
diff <(grep -vE '^\s*#|^\s*$' ../../par8.glm52.dpaoff.packup_20260803/spec/par8.yaml) \
     <(grep -vE '^\s*#|^\s*$' par8.yaml)
# -> exactly one hunk, the tokenizer line
```

par8 itself differs from Case A in exactly three numbers — `initial_sessions`
32→8, `max_sessions` 128→32, `max_inflight` 48→24. The whole request profile
(percentile triples, 0.89 cache construction, `new_session_rate` 0.10, seed 1337,
the 400/3600 window) is Case A verbatim.
