# The o11y panel: AgentsView

`agent_sys` ships an observability panel — [AgentsView](https://github.com/kenn-io/agentsview),
an external Go binary — over the session transcripts a run produces, at
**`http://127.0.0.1:18888`**. Deploying `agent_sys` starts it; it stays resident
across runs and binds loopback only. **AgentsView's own code is never modified.**

Design, and the measurement behind every claim here: **`design.md`**, beside this
file.

## Using it

| | |
|---|---|
| `--agentsview-port N` | the port. Then `AGENTSVIEW_PORT`, then `18888` |
| `--no-agentsview` | do not start it. Not one external call is made |

`--dry-run` and `--clean` are exempt: a dry run that leaves a resident daemon
behind has broken its only promise.

You will see two lines on the way past:

```
      o11y  panel at http://127.0.0.1:18888
      o11y  this run is project 'run.20260904T121032_b65238' on the panel
```

## The four things an operator should be told plainly

**Your own `~/.claude` is never read, written, or reconfigured.** Measured
against the live daemon: of the 122 session roots AgentsView would scan, 122 are
inside `~/.infera_agent_sys` and 0 outside.

**Each run is one project.** AgentsView names a project after the session's
deepest path segment, and every attempt runs in its own nested zone, so a run
would otherwise arrive as a dozen unrelated entries. The label lives in a
mapping row, not on the session — so deleting the row un-names that run at the
next full re-sync. There is no automatic cleanup, on purpose.

**A pre-existing AgentsView of yours is never adopted.** Reuse needs two gates:
it answers `/api/v1/agents` with JSON, **and** a live `daemon.<pid>.json` in our
own data directory names that port. If you expected reuse and got "port in use",
that is why.

**The panel cannot fail your run.** Binary missing, port taken, daemon wedged,
health timed out, mapping refused: each is one warning and a skip. Install
failure too, through the recipe's `importance: suggested`.

## Where things are

| | |
|---|---|
| `agentsview.py` | the daemon — install, port, launch, ownership, health |
| `mapping.py` | what the panel shows — one project per run |
| `../../recipes/agentsview.o11y.yaml` | the pinned release and its checksum |
| `../../prefix.py` | `~/.infera_agent_sys`, an `env_mgr` layout this is the first consumer of |
