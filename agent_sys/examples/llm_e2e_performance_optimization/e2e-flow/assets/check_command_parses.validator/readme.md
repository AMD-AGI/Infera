# `check_command_parses`

Program validator (mission G4.1). One rule: a `reproducible` handoff's
`items/command` (or `items/script`) must parse under the shell that will run it.

`agent.gate` requires the item to be **executable**. Nothing required it to
**parse** — and 11 of the 14 sealed `items/command` scripts under
`cheat_for_mock/` do not, all from one cause: an apostrophe inside a
`${VAR:?word}` message opens a single-quoted string that runs to end of file.

Found by m2 sweeping for a class they had just fixed in their own generators.
See `check.py` for the full account.
