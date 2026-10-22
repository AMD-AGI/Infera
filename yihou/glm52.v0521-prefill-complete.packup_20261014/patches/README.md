# Patches

## 0001 — `prefill_complete` validation vs attention DP (YIHOU-PCOMPLETE-ATTNDP)

| file | sha256 |
|---|---|
| `pd_disaggregation_hook.py.orig` (= image file `srt/arg_groups/pd_disaggregation_hook.py`, image 9d755dd6) | `ad3a2f7aa22f45167242f05e00cfc2b04d220cf78fbe3a0c58ada2b3ddd935da` |
| `pd_disaggregation_hook.yihou-attndp.py` (mounted file) | `432766438680cc428a426953dc00d20aec2fccd3f5c0c360cc3e043e55c721ed` |
| `0001-pcomplete-attndp-validation.diff` | `diff -u orig patched` (one changed line + 3 comment lines) |

- **What:** in `_validate_prefill_complete`, compute the attention-DP width from `cfg.attn_dp_size` when it is > 1, falling back
  to the old `cfg.dp_size if cfg.enable_dp_attention else 1`.
- **Why:** `handle_deprecated_dp_attention` (`parallel_hook.py:36-57`) runs earlier in the resolution pipeline and turns
  `--dp-size 4 --enable-dp-attention` into `attn_dp_size=4, dp_size=1, enable_dp_attention=False`. The old expression then yields 1,
  `tp_size (4) != 1`, and startup fails with "does not yet support attention tensor parallelism greater than one" even though
  attention TP is 1 (TP4 / attention-DP4). Every TP4/DP4-attention PD deployment that uses the deprecated spelling hits this.
- **How applied:** runtime bind mount (no image rebuild), read-only, on both legs:
  `scripts/harness/mounts.yihou.pcomplete-attndp.tsv` → `PREFILL_MOUNTS_ADD` and `DECODE_MOUNTS_ADD` in `env.pc48b-274.yihou.sh`.
  Source of the mount on NFS: `/shared_nfs/yihou_glm52/v0521cap2_20261013/mounts/pcomplete/`.
  Decode needs it too: the validation runs on both roles.
- **Symptom cured:** round pc48 (prefill exit code 1 at 10:03:30) → pc48b came up, server-info shows the policy on both legs.
- **Not covered:** runtime checks in `common/conn.py:2057-2072` (decode marks a request Failed if the peer has attention TP ≠ 1)
  read the resolved fields at runtime; no request failed in pc48b (3 / 1 error-dropped requests in A2 / A1, same category
  `InvalidInferenceResultError` as the control), so they did not trip [observed, not code-traced for this case].
- **Alternative without a patch** [untested]: launch with the new spelling `--attn-dp-size 4` (no `--dp-size/--enable-dp-attention`);
  then `handle_deprecated_dp_attention` returns early but the validator's old expression still sees `enable_dp_attention=False` → 1,
  so the new spelling is expected to fail the same way [inferred from the code; not run].
