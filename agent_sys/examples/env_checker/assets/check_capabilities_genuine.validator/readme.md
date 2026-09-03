# check_capabilities_genuine — trustworthiness, strong

Every token in the report is the token that capability actually produces, for
**this run's nonce**. Three of the seven are re-derived by running the
capability here, in this body; the other four are recomputed from the salt in
the installed artefact.

This file spends most of its length on the difference between those two, because
the honest claim is narrower than the validator's name and a reader who quotes
its PASS deserves to know which half they are quoting.

## The scheme, in one paragraph

A token is `sha256(f"{salt}:{label}:{nonce}")[:12]`, prefixed
`ENVCHK-<LABEL>-`. The **nonce** is per-run and supplied at launch; the **salt**
is a 32-hex constant that exists in exactly one place — the artefact of the
capability it belongs to. There is no table of salts anywhere, including in
`../lib/envchk.py`, and that absence is load-bearing: a single file listing all
seven would let one read produce all seven tokens. This body obtains each salt
the same way the agent had to, out of the artefact.

## The three treatments

| capability | treatment | what runs |
|---|---|---|
| `mcp_external` (L2) | **replay** | the component's server is started with `python3`, spoken to over JSON-RPC — `initialize`, `notifications/initialized`, `tools/call` — and its answer is the token |
| `mcp_stdio` (L3) | **replay** | the same, against the bundled server |
| `tooldef` (L3) | **replay** | the module is imported and `TOOLS[0].call()` is invoked |
| `hook` (L3) | salt, **plus the payload** | the salt is read from the hook script; separately, the hook's own output file must carry a `payload` with a `session_id` and `hook_event_name: SessionStart` |
| `skill` (L3) | salt | the salt is read from `SKILL.md` |
| `plugin` (L3) | salt | the salt is read from the plugin's `SKILL.md` |
| `serena` (L1) | salt | the salt is read from `serena_probe.py` |

**The replays go through the declaration, not around it.** `TOOLS[0].call()`
rather than the `echo_token` function beside it, because `TOOLS` is what
`env_mgr` reads and what `claude_sdk._adapt_tool` adapts; calling the function
directly would check something the declaration happens to point at today. The
MCP replays speak the protocol rather than importing `report()`, for the same
reason: a server that answers `initialize` and a module with a function in it
are different facts.

## What a PASS means, per capability

**For the three replayed ones**: the string the agent reported is the string the
capability produces. There is no salt-reading path to that string that does not
also go through the artefact, and the artefact is a program the agent would have
had to run. This is the strongest of the three claims and it is still not a
claim about honesty — see below.

**For the hook**: the salt was installed *and* the harness invoked the script at
`SessionStart`. That second half is the part no file read can produce:
`session_id` and `hook_event_name` arrive on the hook's stdin from Claude Code,
so an agent that ran the script by hand gets an empty `payload` and this body
says so by name.

**For the skill, the plugin and serena**: the artefact carrying that salt was
installed where that capability puts it, in this zone, for this run. Nothing
more.

## What it cannot catch

Written out rather than left implicit, because a check that reads as complete is
worse than one whose edges are known.

- **A file read instead of a capability use.** Four of the seven artefacts are
  files an agent with `Read` can open — two `SKILL.md`s, `serena_probe.py`, and
  the hook script. An agent that opened them and reported the tokens passes this
  validator. It is an in-band limit and it does not have a fix inside one
  trusting environment: the agent and the artefacts are in the same zone by
  construction, because putting them there is the thing being measured.

  What the tokens *do* buy, in full, and it is the failure this package was
  built for: **an agent cannot report seven tokens if the seven capabilities
  were not installed into its zone.** A run where `env_mgr` silently delivered
  nothing produces no salts and therefore no tokens, however confident the
  narrative.

- **Freshness.** `pid` and `at` are checked for shape and not for value. This
  body does not know when the session started — it has no run window — so a
  freshness rule here would be a number with no basis, and enforcing a guess is
  worse than declining to.

- **A forged hook payload.** An agent that found its own session id and wrote
  the file itself is not caught. Nothing here cross-checks the id against a
  transcript, because the transcript is the agent's own and reading it would be
  taking the same word by a longer route.

- **Whether serena was *used* rather than installed.** `serena`'s replay is a
  salt read. `proof.raw` is additionally checked against the **shape** Serena
  1.28.1 returns — a `find_symbol` array whose hit for `envchk_serena_token`
  carries `name_path`, `kind`, `relative_path` naming `serena_probe.py`, a
  `body_location` with integer line numbers, and a `body` containing the salt.

  That schema was **measured on this host on 2026-09-03**, by starting the
  binary probe D installed and calling the tool, rather than remembered. It is a
  **forgery-cost increase and not a closure**, and the difference matters: it
  moves the bar from *read one file* to *read the file **and** know serena's
  response schema*, and **a model that has seen that schema can still fabricate
  it**. Anyone quoting this row should quote it as a cost, not as evidence
  serena ran.

  It also has a cost of its own, which is the reason it was nearly not written:
  it is a check against a **third-party** output format, so a serena release
  that changes the schema turns an honest run red. That is the one place in this
  body where a PASS can be lost without the report being wrong, and the failure
  message says so in as many words — *"if serena's response schema has changed,
  this is a validator update and not a capability failure"*.

  **The salt is checked, not the token.** Measured: `find_symbol` with
  `include_body=true` returns the symbol's body and nothing above it, so a
  module-level salt would be invisible in the response and every honest run
  would fail. `serena_probe.py` keeps the salt as a local inside the function
  for exactly that reason.

- **The L2 registry, when it cannot be found.** `agent_sys/components/` is a
  repository path and a task package is staged into a zone, so
  `../../components` does not exist beside it. This body takes
  `$AGENT_SYS_COMPONENTS_ROOT` first and searches upwards second, and when
  neither answers it reports the L2 capability as unverifiable **by name** —
  which is a fault, not a shrug. `env_mgr` exporting that variable is what
  closes it.

Every one of these is a **false negative**: this validator does not report a
report that has the problem. There is no configuration under which it reports a
report that does not — every rule compares a string the agent wrote against a
string this body derived, and a match cannot be produced by accident.

## serena's one exemption

`capabilities.serena` may read `"status": "unavailable"`. It is the only
capability that may, it is named in the spec's `may_be_unavailable` arg rather
than hard-coded here, and it is **not the agent's to claim**: the run's own
`install_report` must carry an entry that mentions serena and does not look like
a success. An `unavailable` beside a clean install report is a FAIL.

The reason for the exemption is that serena is the one capability whose install
crosses a network and touches a third-party index; a site with no route to it
must still be able to run this package and get a truthful report. The reason for
the cross-check is `.claude/CLAUDE.md`'s first principle: a stage once reported
fourteen tasks and ten validators green over a run in which every result was
zero, and the general form of that failure is a producer being trusted about its
own environment.

The entry-shape matching is **tolerant on purpose**. `env_mgr`'s `Outcome` is a
level, a message and an extra mapping, but the install report reaches here
through the agent's JSON and this body does not own that schema. So an entry
counts when its serialised form mentions serena and does not look like a
success — which cannot be satisfied by an entry reporting a clean serena
install, and cannot be satisfied at all by a report that never mentions serena.

## Why `strong`, and why `cost: minutes`

`strong` because every rule is a comparison between a string the agent wrote and
a string this body derived from the installed artefact. Nothing is approximate
and nothing is judged. `strength` qualifies a PASS (`validator` spec §5.4), and
the PASS is qualified further, per capability, in *What a PASS means* above —
which is where the narrowness lives, rather than in a weaker label that would
hide it.

`cost: minutes` is an order of magnitude, not a measurement: two subprocesses,
one import and a few file reads is seconds of work in the good case, and a
server that hangs is bounded by `replay_timeout_seconds` rather than by luck.
The tag's job is to put this validator **second**, behind a shape check that
costs nothing, and one honest step up from `seconds` is what does that.

## Two things this body deliberately does not do

**It does not print the nonce.** `ENVCHK_NONCE` is compared, hashed and passed
to two subprocesses in their environment — where they already read it from — and
its value never reaches a message, a file or an argument. `ScriptBodyRunner`
folds a body's stderr tail into an exception message, and an exception message
travels into the event stream.

**It does not fall back to an empty nonce.** An unset `ENVCHK_NONCE` would
produce seven mismatches and send a reader to look at the agent, so it is named
as its own failure instead. On this package's only phase the variable is present
— `validator` spec §8.2's PRODUCER row is `Prepared.environment`, into which
`env_mgr.material.deploy` merged the agent spec's `env` block — and the guard is
for whoever adds a consumer and reaches the other row.

## Layout

`entry.sh` is the command, `check.py` is the implementation,
`../lib/zone.py` is the four body-facing zone files, and `../lib/envchk.py` is
the capability register and the token scheme — shared with
`check_env_report_shape`, so the two cannot disagree about which seven
capabilities exist.
