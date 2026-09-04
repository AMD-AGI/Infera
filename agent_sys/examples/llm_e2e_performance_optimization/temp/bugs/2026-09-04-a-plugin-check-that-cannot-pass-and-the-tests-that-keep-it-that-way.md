# A plugin check that cannot pass, and the two green tests that keep it that way

**Found:** 2026-09-04 by `researcher-core` during the PR-155 round; every byte
below re-read by the leader from the captured artefacts, not from the report.
**Severity:** `check` returns "missing" for every declared plugin on every run,
so `install` re-runs `claude plugin install` every time. Nothing is corrupted —
this is wasted work and a meaningless signal, not damage.
**Status:** reported, **not fixed**. The fix lands in `env_mgr/installers/`,
which `env_mgr` criterion 22 pins byte-identical; the owner holds that decision.

## 1. The defect

`env_mgr/installers/claude.py:22-30`. `_present_names` takes `line.split()[0]`
from each line of `claude plugin list` (`_installed()`, `claude.py:16` — note it
is `plugin list`, **not** `marketplace list`).

Real output, `cat -A`:

```
Installed plugins:$
$
  M-bM-^]M-/ code-review@claude-code-plugins$
    Version: 1.0.0$
    Scope: user$
    Status: M-bM-^\M-^T enabled$
```

`e2 9d af` is **U+276F `❯`**, preceded by two spaces. So the first whitespace
token of a plugin line is the glyph:

| input | `_present_names` returns | caller needs |
|---|---|---|
| 3 plugins installed | `{'Installed', 'Scope:', 'Status:', 'Version:', '❯'}` | `{'code-review', 'commit-commands', 'hookify'}` |
| none installed | `{'No'}` — the first word of *"No plugins installed."* | `set()` |

`check` compares against `spec.split("@")[0]` (`claude.py:36`), so no declared
plugin can ever match. The empty case yielding a junk name is harmless only for
as long as nobody ships a plugin called `No`.

## 2. The root cause is the tests, and that is the part worth reading

`tests/env_mgr/test_installers.py:151` feeds
`"superpowers 1.0\ncode-review 2.1\n"` — space-separated, no bullet, no
`@marketplace`. **The CLI does not produce that format and never did.** Measured:

```
SYNTHETIC input:  shipped -> ['code-review', 'superpowers']   PASS
                  anchored fix -> []                           FAIL
```

So the defect shipped *because* it had two green tests, and both encode a
remembered third-party format rather than a measured one. This is the repo's own
standing rule — *do not encode an unmeasured third-party schema* — seen from the
other side: the same mistake made in a **test** manufactures a passing check
instead of a failing one, and is correspondingly harder to notice.

**A one-line fix is therefore not the whole fix.** Both tests must be rewritten
against captured bytes. The second one's property — that `super` must not match
`superpowers` — is real and must survive the rewrite; it only needs a fixture in
the format the CLI actually emits.

## 3. A second wrongness, which the fix would expose rather than cause

A **disabled** plugin still appears in `plugin list`, identically bulleted,
differing only in one glyph:

```
  ❯ hookify@claude-code-plugins$
    Version: 0.1.0$
    Scope: user$
    Status: M-bM-^\M-^X disabled$     (U+2718)
```

Once `_present_names` parses correctly, `check` reports `hookify` **present** for
a plugin that will not load. *Installed* and *available to the session* are two
properties and the function would conflate them.

**Deliberately not fixed, and the reason is a prior question:** telling them
apart means parsing the `Status:` line, which is a second dependency on an
unmeasured format. Before writing that, someone has to decide whether `check` is
supposed to mean "installed" or "enabled". That is a design question, not a patch.

## 4. What is still open about the fix itself

Three candidates were run against every real input. **All three return exactly
the right set on all of them**, so the measurements do not pick a winner:

| candidate | rule |
|---|---|
| **anchored** | `^\s*\S\s+([^\s@]+)@\S+$` — bullet, then `name@marketplace`, to end of line |
| glyph-free | any whitespace token containing `@` |
| no-colon | a token with `@`, on a line containing no `": "` |

The input that would separate them — a metadata line carrying an `@`, e.g.
`Source: git@github.com:…` — **appeared in no capture and was not constructed**.
Recorded as open rather than resolved.

The anchored form is nevertheless the recommendation, on a **failure-mode**
argument rather than a measurement, and the distinction is the point:

- if the bullet is ever restyled, the anchored form yields a false **absent** →
  a redundant re-install, which is exactly today's behaviour and is safe;
- the glyph-free form yields a false **present** → the install is skipped and the
  plugin is silently missing.

An asymmetric failure mode is a legitimate reason to choose; it is not evidence
that the other two are wrong, and it should not be written up as though it were.

## 5. Provenance

Stated because a probe is evidence only about the build it ran on.

| | |
|---|---|
| binary | `/home/yihou/.local/bin/claude`, **2.1.246** — resolved through `PATH`, which is what `installers/base.py:31` gets (`shell=True`, inherited env). No SDK bundle involved |
| config | `CLAUDE_CONFIG_DIR` into `/tmp/yihou/agentsys_pr155_20260904/probe_claude_plugin_list/config` throughout |
| host isolation | `~/.claude/plugins` md5 `1214bce39983fb0398ebaec87ade0989` before and after seven installs — **unchanged**. (`~/.claude.json` was *not* used as the isolation test: live Claude Code sessions rewrite it continuously, so its checksum proves nothing on this host) |
| capture form | pipes, non-TTY — the only form `base.py` ever produces. Under a pty the output differs materially (ANSI column escapes `\e[3G`/`\e[5G` replace the indentation, CR line endings) and **all four functions fail on it**; that capture is kept and labelled `DECOY — do not build on it` |

Artefacts, with an index marking which inputs are real, which is
non-discriminating and which is the decoy:
`/tmp/yihou/agentsys_pr155_20260904/probe_claude_plugin_list/README.md`.

## 6. Why it is not fixed here

`env_mgr/installers/` is one of eight paths `tests/env_mgr/test_cli_subcommands.py
::test_the_shipped_modules_are_byte_identical` pins.

Worth separating from the defect itself: criterion 22's own wording is *"its 65
tests keep passing untouched"*, and this fix would fail **none** of those 65 — they
never covered `_present_names`, which is why it shipped. The test asserts
something stricter than the criterion it cites, so a reader trusting the citation
would wrongly conclude the machinery's own tests forbid the repair. That gap is
with the owner as decision (A) and is not worked around here.
