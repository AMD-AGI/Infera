# patches/ — YIHOU-PCOMPLETE-ATTNDP (used only by pc64)

- **What:** one-line change in `sglang/srt/arg_groups/pd_disaggregation_hook.py` (`_validate_prefill_complete`): use the attention-DP size
  after `handle_deprecated_dp_attention` rewrote `--dp-size 4 --enable-dp-attention` into `attn_dp_size=4, dp_size=1`.
  Files: `pd_disaggregation_hook.py.orig` (image), `pd_disaggregation_hook.yihou-attndp.py` (mounted), `0001-pcomplete-attndp-validation.diff`.
- **Why:** without it the stock v0.5.21 image rejects `--disaggregation-decode-allocation-policy prefill_complete` for TP4/DP4-attention
  (ValueError at startup, round pc48 on 10-13).
- **How:** bind-mounted read-only on both legs. pc48b (10-13) used `{PREFILL,DECODE}_MOUNTS_ADD` (tripped audit v3's role-only rule);
  pc64 uses `config.yihou.pcomplete2.sh` → `YIHOU_MOUNTS=mounts.yihou.v2-regbase-attndp.tsv` (regbase rows + hook row) — audit PASS today.
- **Context / origin:** full write-up and upstream-PR candidate in
  `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-prefill-complete.packup_20261014/patches/`. The path was abandoned by the user on 10-14.
