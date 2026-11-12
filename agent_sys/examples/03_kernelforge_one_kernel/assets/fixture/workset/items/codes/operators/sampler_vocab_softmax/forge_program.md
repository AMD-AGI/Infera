<!-- EXAMPLE: generated fixture for examples/03_kernelforge_one_kernel. -->
# Infera workset campaign: sampler_vocab_softmax

Optimize `srt/layers/sampler.py` with the `triton` backend inside the isolated engine
workspace. The operator's Definition, Workloads and invocation specification
remain authoritative. Do not change the protected driver, correctness/performance
harness, cases, baseline or acceptance gates. Run every declared case. Forge's
graph-search score is not an independent workset or serving-speedup claim.

## Required external knowledge consultation

Before editing in each round, you MUST explicitly read the resolved external
AKLR KB entry and relevant maintained pages with your file-reading tools. Locate
the actual KB root from the external skill-directory paths supplied in your
prompt (`--skill-dir`, also `AKLR_KB_ROOT` when available). Read that root's
`SKILL.md`, the supplied language entry, and follow their routes for this
operator, regime and GPU to the relevant pages. Inlined entry text and path
delivery alone do not prove that you consulted those pages.

Record the actual page/section references and the decision they informed in the
round's notes: `KB: matched <page> <section>`, the question, what changed in your
plan, and the source/test evidence used to check it. If lookup finds no relevant
answer, record `KB: no-match <precise question>` and which entry/routes you
actually inspected. Do not invent consultation or infer it from a timing gain.
Keep the KB external: reference its pages without copying their contents into
the kernel, workset, notes or handoff. If the selected KB cannot be read, report
that missing dependency before making kernel edits.
