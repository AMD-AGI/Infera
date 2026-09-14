# check_analyze_packup_shape — completeness, strong

The packup carries every file the deliverable layout mandates, and each has
substance.

## What it checks

1. The `reproducible` items — `result`, `env`, `command`, `code` — are present.
2. `README.md`, `REPRODUCE.md`, `environment.md` and `notes.md` all exist.
3. `results/` and `logs/` exist and are non-empty.
4. Every mandated file clears `min_bytes`.
5. None contains a placeholder phrase.

## Why rules 4 and 5 exist

A file that exists and says nothing passes a presence check and fails a reader.
`min_bytes` is set low enough that a genuinely terse section passes and high
enough that a stub does not; the phrase list catches the stubs that are long
enough to clear the byte floor.
