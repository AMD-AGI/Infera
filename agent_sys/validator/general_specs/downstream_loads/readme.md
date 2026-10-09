# downstream_loads

"Can the next task actually use it?" is `strong` whenever the downstream loader
exists, because the loader *is* the criterion and it is stated in advance: the
artefact either parses or it does not.

Read `inputs.json`, run the declared loader over each handoff, write
`verdict.json`.
