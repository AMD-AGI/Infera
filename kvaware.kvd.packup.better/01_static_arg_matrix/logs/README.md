# Logs — none, by design

This experiment starts no engine, loads no model, and touches no GPU. It runs
sglang's `ServerArgs.from_cli_args` in a throwaway container and exits, so there
is no server log to keep.

**The full observable output IS the result file:**
`../results/arg_matrix_verdicts.txt` — the 8 verdict lines plus the direct
`disable_radix_cache` probe.

Re-running `../scripts/run.sh` writes a fresh
`../results/arg_matrix_verdicts.observed.txt` for diffing against it.

## If you want more detail than the verdict lines

`argcheck.py` prints one summary line per OK case (`hier=`, `backend=`, `ratio=`,
`page=`, `disable_radix=`). To see sglang's own reasoning, drop the grep the
runner applies:

```bash
J "docker exec $CTR python3 /tmp/argcheck.py 2>&1"     # unfiltered
```

That surfaces sglang's WARNING/INFO lines during arg resolution — useful when a
verdict changes on a different sglang version, since it shows which normalisation
pass moved the value.
