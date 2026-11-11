# CPU configuration checks for #49732

Continuation of [#46918](https://github.com/sgl-project/sglang/pull/46918), original
implementation by yu-depend. Current production configuration AST is identical.
The checker compiles the actual production function and actual registered
TestAbsorbBmmConfig class, loading the real stdlib-only environ module. It
isolates GPU-heavy imports and substitutes integer cdiv/device discovery only;
it is not a full registered-module/import integration test.

```bash
python -B check_config.py --checkout /path/to/sglang-pr49732
python -B check_config.py --checkout /path/to/sglang-pr49732 \
  --negative-base eef6d87111ca94083f2840821de02d037463c174
```

On candidate da22f562: 9 tests / 259 subtests pass. The unpatched-main negative
control has 133 failures and one error, as expected. All repository pre-commit
hooks pass. Full registered-module pytest was attempted and collection lacked
pybase64 in the local CPU environment. CI must run the full registered test.
No new GPU or serving benchmark was run. Historical MI355X evidence is pinned
in [the original validation](../b8_46918/RESULTS.md), at original commit 9763ef56;
it does not qualify the new main/image combination.
