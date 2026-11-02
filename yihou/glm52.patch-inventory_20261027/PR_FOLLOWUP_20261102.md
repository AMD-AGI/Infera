# GLM-5.2 upstream follow-up, 2026-11-02

The original eleven owned core PRs now comprise **one merged, one closed as
superseded, four ready for review, and five drafts**. Infera #164 remains an
additional comparison-only draft. Ready for review does not mean approved or
merge-ready. See [the live status table](PR_STATUS.md), [Chinese export](PR_SUMMARY_zh-CN.md),
[PR-to-pack-up map](PR_PACKUP_MAP.md), and [timestamped GitHub snapshot](repro/PR_SNAPSHOT.json).

## Work completed

| PR | Result of this follow-up | Remaining action |
|---|---|---|
| Infera [#218](https://github.com/AMD-AGI/Infera/pull/218) | Verified merge on October 30, approved by xiaobochen-amd; merge commit `d8babfb09aee9ca741766f08e21362e5dbf52eda`. | Retired from the active queue. Its historical image backport remains optional build tooling. |
| SGLang [#46828](https://github.com/sgl-project/sglang/pull/46828) | Closed after verifying merged #47180 now prepares replicated Q from per-tensor FP8 GLM weights. Original head and historical T10 evidence preserved. | No replacement option justified without new isolated evidence against the changed baseline. |
| SGLang [#46824](https://github.com/sgl-project/sglang/pull/46824) | Rebased to main `acd6f5570`; signed head `1bcfc5214`. Added a test that calls the actual merged draft-layout predicate. 43 CPU tests / 61 subtests and scoped lint passed. | Draft: resolve or exclude historical DP2×DCP2 graph garbling; map missing transport/PP coverage. #44322 is no longer a dependency hold. |
| SGLang [#46826](https://github.com/sgl-project/sglang/pull/46826) | Rebased to main `acd6f5570`; signed head `21f8e688f`. Added implicit Triton and a2a Q-rep default tests. 12 CPU tests / 13 subtests and scoped lint passed. | Draft: #46824 must land before enabling this configuration; combined-stack graph/coverage gap remains. The old backend-default blocker is resolved. |
| SGLang [#33059](https://github.com/sgl-project/sglang/pull/33059) | Rebased to main `acd6f5570`; signed head `ea88d74bf`. Production delta unchanged; 12 CPU tests and scoped lint passed. Includes #46861's shared Qwen/Mooncake KV-allocation repair. | Draft: HaiShaw's DeepSeek-V3.2 AMD DP/stage-c accuracy matrix remains distinct from the repaired Qwen baseline failure. |
| Mooncake [#5005](https://github.com/kvcache-ai/Mooncake/pull/5005) | Rebased to main `72e267f4`; signed head `8313cccb`. Includes wheel fix #5030; source packaging regression and scoped lint passed. HIP implementation/test source unchanged. | Draft: human submitter line-by-line review and the new inherited CUDA build failure covered by open #5144, then CI/review. No new serving-speedup claim. |
| SGLang [#46820](https://github.com/sgl-project/sglang/pull/46820) | Ready; unchanged `82176e7`. Followed up with the existing four real-ZMQ regression results and a maintainer CI/review request. | Maintainer `run-ci` and review. |
| SGLang [#45484](https://github.com/sgl-project/sglang/pull/45484) | Ready; unchanged `43dec26`. Followed up with existing CPU and MI355X two-onload-cycle evidence. | Maintainer `run-ci` and review. |
| SGLang [#46822](https://github.com/sgl-project/sglang/pull/46822) | Ready; unchanged `11262e1`. Followed up with existing CPU controls and historical R18 equivalence. | Maintainer `run-ci` and review. |
| SGLang [#46830](https://github.com/sgl-project/sglang/pull/46830) | Ready; unchanged `8bd9fe8`. Followed up with existing CPU controls, consumer audit and historical deadlock evidence. | Maintainer `run-ci` and review. |
| SGLang [#46819](https://github.com/sgl-project/sglang/pull/46819) | Draft; unchanged `aed9bc6`. Corrected stale body text saying the GPU regression had not run; retained the actual MI355X and isolated cache results. | Narrow to caching or qualify pinned-index lifetime, stream-priority interference and PP producer ordering. #44724/#39768/#43782 remain open coordination items, not established mandatory dependencies. |

The four rebases were pushed only to the corresponding yihou PR branches, using
leases pinned to their observed old heads. No other author's branch or PR was
modified. Rebased production deltas are unchanged; only #46824 and #46826 gain
new tests. No GPU node or experiment container was used in this follow-up.

## Changes in external dependencies

| Patch family | Verified state | Consequence |
|---|---|---|
| DCP backend [#44322](https://github.com/sgl-project/sglang/pull/44322) | Merged October 28. | Its canonical draft-sharding policy and ROCm Triton defaults are in the rebased D10/D11 branches. Closed #42618 is history, not an active dependency. |
| Q replication [#47180](https://github.com/sgl-project/sglang/pull/47180) | Merged November 1 as `b50d4394945d3dcacea05ee4afd334a9b225cbdc`. | Removes D22's original need to retain BF16 merely to activate Q replication. This is functional supersession, not a verbatim merge of our precision option. |
| Non-CUDA/NIXL synchronization [#43299](https://github.com/sgl-project/sglang/pull/43299) | Merged October 29. | Retire this dependency. It does not replace the distinct PP producer event fix #45524. |
| PP producer event [#45524](https://github.com/sgl-project/sglang/pull/45524) | Open, ready; updated `14c8591329`. | Author removed the unnecessary cached-prefix hunk and reports the #47074 trigger improves from 1/24 to 24/24 with the remaining fix. These are the author's results, not a new run by us. [Author's update](https://github.com/sgl-project/sglang/pull/45524#issuecomment-6485757646). |
| Qwen/Mooncake allocation [#46861](https://github.com/sgl-project/sglang/pull/46861) | Merged October 28. | The historical Qwen PD 0–1% GSM8K failures match its upstream fused-KV regression diagnosis. #33059 now includes the repair; new-head CI/model accuracy is still separate. |
| Mooncake wheel staging [#5030](https://github.com/kvcache-ai/Mooncake/pull/5030) | Merged October 28 as `3fbd075442abd408e1291637439a0e327642b9e3`. | Fixes the missing `_structured_ranges` module introduced in base #3954. Included in #5005's new base; no duplicate packaging PR. |
| Mooncake CUDA guard build [#5144](https://github.com/kvcache-ai/Mooncake/pull/5144) | Open, ready. | Fixes the generated-header include path of `on_cuda_transport_guard_test`. The new #5005 CI fails there before tests. Target/test/header are unchanged from main; [our new CI evidence](https://github.com/kvcache-ai/Mooncake/pull/5144#issuecomment-6569064104) was contributed to the existing fix. |
| Transfer queue sharding [#42163](https://github.com/sgl-project/sglang/pull/42163) | Still open draft, unchanged `294113a01f`. | Continue following YAMY1234's existing PR; our isolated transfer evidence is already posted. |
| Active idle backend [#45156](https://github.com/sgl-project/sglang/pull/45156) | Open, ready; rebased `c02b88d488`. | Our duplicate #45483 stays closed. |
| `prefill_complete` [#45337](https://github.com/sgl-project/sglang/pull/45337) | Open, ready; rebased `60fe4e6a88`. | Existing evidence remains on the author's PR; no duplicate. |
| HIP IPC relocation [#4340](https://github.com/kvcache-ai/Mooncake/pull/4340) | Open, ready; unchanged `9b4b049d9c`. | SGLang #46861 is a workaround; the root Mooncake fix and a release carrying it still matter. |
| HIP async submission [#4953](https://github.com/kvcache-ai/Mooncake/pull/4953) | Open, ready; `35c488e9`. | Still calls the stream pool API; no competing priority implementation was found in its current diff. Combined runtime behavior is not newly validated. |

SGLang issues [#46829](https://github.com/sgl-project/sglang/issues/46829) and
[#46831](https://github.com/sgl-project/sglang/issues/46831) remain open with no
new replies. S13's attribution hold, K01/A01's readiness limits, the explicit
excluded experiments, and retired bundles retain their previous dispositions.

## Validation and evidence limits

Executed CPU commands after the relevant rebase:

```bash
# SGLang #46824, signed head 1bcfc5214
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/disaggregation/test_dsa_pd_dcp.py \
  test/registered/unit/disaggregation/test_dcp_pack.py \
  test/registered/unit/disaggregation/test_mooncake_transfer_batching.py

# SGLang #46826, signed head 21f8e688f
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/test_dsa_pd_dcp_config.py \
  test/registered/unit/test_dsa_dcp_backend_defaults.py

# SGLang #33059, signed head ea88d74bf
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py

# Mooncake #5005, signed head 8313cccb
python -m pytest -q \
  python/tests/packaging/test_project.py::test_structured_object_storage_has_one_authoritative_source
```

Each changed PR passed applicable PR-scoped pre-commit hooks. The Mooncake test
checks source packaging; it is not an installed-wheel or HIP runtime test.
Old and new upstream CI must be distinguished: most SGLang red entries are
initial draft/permission gate failures with downstream tests skipped. The four
ready PRs still lack `run-ci`. Title/lint success is not functional-test success.
The rebased #33059 has `run-ci`, but its functional jobs are blocked by the draft gate; no new AMD accuracy result was produced. The exact current CI snapshot is timestamped; reruns are not claimed complete
until GitHub reports completion.

The new Mooncake failure is not the old wheel failure: [CUDA CTest build job
125092088899](https://github.com/kvcache-ai/Mooncake/actions/runs/41387680143/job/125092088899)
cannot find `mooncake_te_build_config.h` while compiling the unchanged
`on_cuda_transport_guard_test`. Open #5144 adds the missing generated-header
include directory; the TENT cuda-on job fails at the same target. #5144 itself has 27 successful checks in the final snapshot. Its author owns that repair; no duplicate PR or unrelated
fix was bundled into #5005.

D22 retirement additionally used thirteen CPU probes of the unchanged current
weight-conversion helper and quantizer. An initial assertion incorrectly assumed
that a one-element FP32 vector scale preserves BF16; PyTorch promotes that case
to FP32. The corrected audit explicitly distinguishes that shape and verifies
that GLM's actual quantizer returns a zero-dimensional scale whose absorb result
is BF16. The public retirement comment was corrected. None of these probes
establishes distributed gather, GPU kernels or full-model numerical accuracy.

Mooncake's outstanding human review is an explicit repository requirement:
[AGENTS.md](https://github.com/kvcache-ai/Mooncake/blob/72e267f4/AGENTS.md)
requires that “the human submitter has reviewed every changed line and can
defend the change end-to-end.” An agent review cannot certify that on yihou's
behalf. The PR remains draft; no additional blanket serving benchmark gate is
imposed for reviewing the opt-in itself.

All historical serving-throughput caveats remain in force: simulated MTP
acceptance, known PP-overlap/custom-all-reduce correctness concerns, and no
attribution of combined-stack speed to an individual patch. The frozen llr37 source artifacts remain unchanged. All 22 checksum entries pass; only the live PR snapshot checksum was refreshed. Original untracked-file hashes also match the preflight inventory.
