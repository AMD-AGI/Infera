# Worked examples — sanitised kits from runs that passed

One file per model. Each is the delivery kit an earlier run of this task
actually produced, **stripped of local and process information**, kept so the
next run has a shape to match rather than a shape to invent.

They are read by the agent (the brief points here) and by anybody deciding
whether a new kit is good enough. They are **not** commands to copy: the ports,
paths and hostnames that made them work were one site's.

## What "sanitised" means, exactly

An exemplar is derived from a real kit by removing two classes of content and
nothing else.

**Local information — removed, because it is one machine's and pretending
otherwise is how a kit stops travelling.**

| removed | replaced with |
|---|---|
| hostnames, node names, cluster names | `$HOST` / a role word (`the node`) |
| absolute paths carrying a user name or a site layout | the variable the kit already uses — `$MODEL_PATH`, `$WORK_ROOT` |
| IP addresses | `$MY_IP` |
| chosen port numbers, container names, run tags | the `${VAR:=…}` form the kit itself must use |
| `agent_sys` run ids, zone paths, handoff uuids | dropped |
| anything site-specific in `environment.md` — driver build, image digest of a locally committed image | kept as the *fact category* with the value marked as this-site |

**Process information — removed, because an exemplar is a specification of the
result, not a diary.**

Debugging narrative, the order things were tried in, failed attempts, timings
of the agent's own turns, and any "I then noticed…" prose. The *durable* lesson
that came out of a wrong turn stays — in `notes.md`, stated as a rule rather
than as a story.

**Never removed:** the structure, the section headings, the kinds of evidence
and what each proves, the parameterisation patterns, and the wording of
`Expected output`. Those are the whole point.

## What is not sanitised away, and must not be

- **The `## Result` section's verdict.** An exemplar that no longer says whether
  it worked teaches the wrong thing.
- **The two independent readings of the deployment mode.** That the worker's own
  log and the router's listing are read separately is the property being taught.
- **The published model name.** `Qwen/Qwen3.6-27B` is not local information; the
  path it was loaded from is.

## Rule of thumb

If a reader on a different cluster could copy a line and have it mean the right
thing, keep it. If copying it would silently point them at this site, it is
local information.
