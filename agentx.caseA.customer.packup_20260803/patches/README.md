# Patches — N/A for this run

**No patch was applied to anything.** This is a deliberate property of the run,
not an omission, so it is recorded here rather than left as an empty folder.

| target | patched? | why not |
|---|---|---|
| the customer's `replay_caseA.sh` | **no** | operator decision: zero modification, so the result stays credible to the customer. `md5sum` = `7cde1afc627c7e4868eac0fd13741baa`, verified in `../logs/run_caseA.log` and reproducible against `../spec/replay_caseA.sh`. |
| the customer's corpus / synthesizer | **no** | the frozen tarball from PR #173 was used as shipped; `verify_caseA.py` re-run on two machines, 13/13 PASS both times |
| aiperf (SemiAnalysis fork) | **no** | installed from the branch as published |
| the sglang engine / infera router | **no** | the deployment was already running and was **not** restarted or reconfigured. It is the par8 posture, byte-identical — see `../environment.md` and `../env/node_snapshots.txt`. |

## The two adaptations that could have been patches, and were not

Both are documented as defects in `../analysis/customer_method_review.md` and
were worked around **outside** the customer's code:

1. **The container cannot see `/mnt/vast`.** `replay_caseA.sh` mounts only
   `$HERE`, `/models`, `/shared_nfs`. Rather than add a mount to their script,
   the tokenizer files (20 MB, no weights) were **copied into `$HERE/tokenizer`**
   and `TOK` pointed there. Data staging, not a code change.
   *(REPRODUCE.md step 3.)*

2. **`OUT` outside `$HERE` sends results into the container namespace**, where
   they die on exit — and the script then prints `FAILED` for a run that
   succeeded. Rather than patch the script's mount list, `scripts/rescue_artifacts.sh`
   polls the running container and `docker cp`s the artifacts out every 20 s.
   *(REPRODUCE.md step 5; symptom and evidence in `../notes.md` Trap 2.)*

   A reproducer who simply sets `OUT=$HERE/results` avoids this entirely and does
   not need the rescue loop. We chose the loop so the upstream defect would be
   demonstrated rather than silently sidestepped.

## Everything else is environment

The full set of values that differ from the customer script's defaults —
`URL`, `SERVED`, `TOK`, `CONCS`, `DUR`, `IMG`, `OUT` — is tabulated with a reason
per row in `../environment.md` § "The measurement configuration", and applied by
`../scripts/run_caseA.sh`, which sets env and then calls their script unmodified.
