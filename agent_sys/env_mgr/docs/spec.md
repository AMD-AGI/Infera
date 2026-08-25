# Environment Manager — Specification

| | |
|---|---|
| Status | Draft, pending review |
| Revision | 1 — 2026-08-24. Widens the scope of the shipped `env_mgr` from environment recipes to all operating-system interaction |
| Date | 2026-08-24 |
| Scope | All interaction with the Linux system: storage, workspaces, permission zones, local↔remote mapping |
| Source | The task definition §7; the shipped `env_mgr` (PR #123) |
| Part of | [`../../docs/spec.md`](../../docs/spec.md) — the whole-system specification |

---

## 1. Purpose

`env_mgr` owns **everything the system does to the operating system**. Storage,
the agent's own working environment, the remote machine where experiments run,
and the permission boundaries that make the other components' authority rules
real rather than intended.

The shipped `env_mgr` is a layered environment manager: one YAML recipe drives
check / dry-run / install / bootstrap across Python, apt, binaries, and Claude
plugins. That remains, and §9 says what survives unchanged. This document widens
the remit around it.

**Why one component and not several.** These responsibilities look separable —
storage, workspaces, permissions, remote access — and are not, because they share
one fact: the path. A permission zone is a path prefix; a handoff's storage
location is a path; a workspace is a path; the local↔remote mapping is a pair of
paths. Splitting them would put one fact in four places.

### 1.1 In scope

- All storage and persistence.
- The filesystem manager: domain registration, in-domain permission, and
  local↔remote mapping.
- Permission zones and the prefix convention that makes them cheap to check.
- Remote access.
- The agent's local and remote environments: workspace, playground, handoff
  storage.
- Preparing an agent's environment before it works.
- Its own metadata and configuration.

### 1.2 Out of scope

**The runtime environment of the program under test.** sglang, vllm, and infera
environments — and whether one is consistent before and after a change — are
**not** `env_mgr`'s business. Two things follow, and §7 develops both:

- Consistency and its rules belong to the handoffs and validators that name them.
- Deployment guides and cluster conventions enter the system as **knowledge
  handoffs**, not as code here.

Also out of scope: **an agent's own process and durability** (`task_graph` spec
§1.2), and **what a backend does internally** (agent spec §4.1).

---

## 2. Design principles

| # | Principle | Consequence |
|---|---|---|
| 1 | **The path is the fact** | Permission, storage, and mapping are all expressed against paths, in one component. §1 |
| 2 | **Write narrowly, read broadly** | An agent may not write outside its zone; reads are looser. §4.3 |
| 3 | **Enforcement is a hook, not a convention** | A rule an agent is merely told is a rule it will satisfy literally and defeat. §4.4 |
| 4 | **Cheap to check** | A permission decision happens on every tool call, so it must be answerable from the path alone. §4.2 |
| 5 | **Work on a copy** | An agent copies a handoff into its playground and works there. It never edits the stored artefact. §6.2 |
| 6 | **Reuse what shipped** | The existing recipe, layer, and installer machinery is not rewritten. §9 |

---

## 3. What `env_mgr` owns

Five sub-modules. Each is named here and specified below.

| Sub-module | Owns |
|---|---|
| **metadata** | `env_mgr`'s own configuration and the mappings below. Persisted; auto-detectable where possible; producible by a designated task |
| **filesystem** | Domain registration, in-domain permission, local↔remote mapping and sync. §4 |
| **remote** | How to reach a remote machine. §5 |
| **agent environment** | Workspace, playground, and their local/remote correspondence. §6 |
| **handoff storage** | The read/write entry point for handoffs. §6.2 |

### 3.1 Metadata

`env_mgr` has configuration of its own, and it is persisted rather than inferred
fresh each run: the domain registrations, the local↔remote path mappings, the
sync strength of each mapping, and the base configuration each of the above
needs.

Three ways it is populated, in decreasing order of preference: **auto-detected**
where the answer is discoverable (is this path on NFS? is this a git worktree?);
**produced by a designated task**, for answers that require running something;
**declared**, for the rest.

---

## 4. The filesystem manager

### 4.1 Domains

A **domain** is a registered region of the filesystem with a name, a root, and a
kind. Domains are registered — created or reloaded — rather than discovered,
because an unregistered region is one nothing governs.

| Kind | What it holds |
|---|---|
| `handoff_storage` | Handoffs, durable |
| `playground` | An agent's scratch: run logs, debug scripts, progress notes |
| `workspace` | An agent's code checkout |

Registration is idempotent: registering an existing domain reloads it rather than
recreating it, because an agent's playground surviving a restart is what makes
that agent resumable (§6.1).

### 4.2 The permission zone prefix convention

**A governed path is recognisable from its prefix alone.** Directories are laid
out flat, with a common prefix that a hook can match without consulting any
registry:

```
${root}/agent-handoff-permission-zone-prefix.<zone-id>/...
        └──────── the marker ────────────┘ └── the zone ──┘
```

The check is two steps, and the split is the point:

1. **Is this a governed region?** Match the prefix. A path that does not carry
   the marker is not in a permission zone, and the check ends — which is the
   common case and costs a string comparison.
2. **May *this* agent reach *this* zone?** Compare `<zone-id>` against the
   agent's declared zones (agent spec §3.2).

Cheap, because step 1 rejects almost everything and step 2 is a set membership
test against a list the agent spec already carries. A permission decision happens
on every tool call, so anything more expensive would be felt.

Flat rather than nested, because a nested layout makes the zone a function of
depth, and a hook would have to walk the path to find it.

### 4.3 The two directional rules

| Direction | Rule |
|---|---|
| **Write** | An agent may not write outside its zones. Local or remote, no exception |
| **Read** | Looser. A governed path the agent's list does not name is denied; everything else is allowed |

The asymmetry is deliberate. Writes are how an agent affects the world and how it
could corrupt another agent's materials; reads are mostly how it does its job,
and denying them by default would make every agent spec an exhaustive inventory
of the filesystem.

**The read rule is what enforces the validator boundary** (validator spec §8.1):
a validator's checking standard lives in a governed zone that the producing
agent's list does not name, so the read is denied. That is the entire mechanism —
no separate validator-privacy system, just a zone the producer cannot name.

### 4.4 Enforcement

Both rules are enforced by a **`PreToolUse` hook** on the agent's backend, for the
reason agent spec §5.3 records: the SDK's `can_use_tool` callback fires only when
the permission flow falls through to a prompt, and `allowed_tools` entries,
settings rules, and permissive modes all bypass it. `PreToolUse` fires on every
call.

The hook resolves the target path, applies §4.2, and either allows or blocks with
a message saying why. **Blocking silently is not acceptable**: an agent that does
not know why it was blocked retries creatively, and creative retries against a
permission boundary are exactly what the boundary exists to prevent.

### 4.5 Local↔remote mapping

A domain may exist on both a local and a remote machine. The mapping records the
correspondence and its **sync strength**:

| Strength | Mechanism | Meaning |
|---|---|---|
| **strong** | Same NFS mount | The two paths are the same bytes. Nothing to sync |
| **strong** | Same mount | As above |
| **weak** | `rsync` | Two copies. **Synchronised only when explicitly called** |

The strength is recorded per mapping and is not inferred at use time. A caller
that needs the remote side current either has a strong mapping or calls the sync;
there is no automatic background reconciliation, because a sync that happens
without being asked for is a sync that happens at the wrong moment.

`local` and `remote` may be the same machine. Nothing in the model assumes
otherwise, and the mapping then has strong strength trivially.

---

## 5. Remote access

Two mechanisms: **ssh** and **docker exec**. Both are specified as ways to reach
a shell on a remote environment; which one applies is part of the environment's
configuration, not the caller's concern.

Reaching *inside* a container or a pod is part of the remote working convention,
and — per §7 — the convention itself arrives as a knowledge handoff rather than
being hard-coded here.

---

## 6. The agent's environment

### 6.1 Local

Three things, and their relationship is what makes an agent resumable:

| | |
|---|---|
| **workspace** | A **worktree** cut from the main repository. Other dependency repositories are cloned inside it. Working directly in the main checkout is not permitted |
| **playground** | A temporary working area: intermediate files, run logs, debug scripts, progress notes |
| **handoff storage** | The entry point of §6.2. Not a directory the agent walks |

**The resume identity**, stated plainly because it determines what must be
durable:

> An agent's **history** (owned by its harness — agent spec §5.1) plus its
> **playground** plus its **workspace** is enough to resume that agent.

Which is why domain registration is idempotent (§4.1) and why a playground is not
swept automatically (handoff spec §11).

A worktree rather than a clone, because a worktree shares the object store with
the main repository: several agents get isolated checkouts without several
copies of the history, and a branch one agent creates is visible to the operator
without a fetch.

Where a playground lives, when the user has designated an NFS root: a
subdirectory of that root if the agent declares NFS, otherwise a subdirectory of
the workspace.

### 6.2 Handoff storage entry

The interface for reading and writing handoffs. Three rules:

1. **A handoff is local-independent** — no local paths, dependencies declared.
   Specified in handoff spec §7; this component is what makes it enforceable,
   since a handoff written with a path from a governed zone is detectable.
2. **An agent copies before using.** Read from storage into the playground; work
   on the copy. Never edit the stored artefact.
3. **An agent reaches only its permitted entries.** §4.2.

Rule 2 is what makes a re-run comparable to the run before it: the input is the
same bytes both times, because nobody has been editing it in place.

### 6.3 Remote

Mirrors local: a remote workspace, a remote playground, and remote handoff
storage, related to their local counterparts by §4.5's mapping. The two
directional rules (§4.3) apply on both sides.

---

## 7. What `env_mgr` does not own

**`env_mgr` is not responsible for the runtime environment of the program under
test.** The sglang / vllm / infera environment a task needs, and whether it is
the same before and after a change, is not this component's problem.

Where it goes instead:

| Concern | Owner |
|---|---|
| Consistency of the program's runtime environment, and the rules for it | The concrete handoff and validator implementations that name it |
| Deployment guides, cluster usage conventions | **Knowledge handoffs** (handoff spec §4), injected like any other |

The second row has a consequence the task definition names and this spec adopts:
**`env_mgr` is itself a consumer of knowledge handoffs.** When it discovers an
environment or deploys itself, it reads the same cluster conventions an agent
would. Which implies **a small set of system-level tasks, agents, and handoffs**
whose job is to produce and maintain those conventions — they are part of the
system, not configuration beside it.

`env_mgr` does own the agent's *own* dependencies — that is what the shipped
recipe machinery already does (§9).

---

## 8. Preparing an agent's environment

Before an agent works, `env_mgr` has everything ready:

| Prepared | Detail |
|---|---|
| **Handoffs** | The agent's declared inputs, copied into its playground (§6.2) |
| **Workspace** | The worktree cut, dependency repositories cloned |
| **Playground** | Created, or reloaded if this is a resume |
| **Permission hooks** | Installed on the backend, for every storage the agent can reach. §4.4 |

The last row is the one with a hard requirement attached: **every storage is
behind a permission hook.** A domain reachable without a hook is a domain outside
the permission model, and its existence would make every §4.3 guarantee
conditional on nobody having created one.

---

## 9. Relationship to the shipped `env_mgr`

The existing component is reused, not rewritten. The task definition requires
both that it be reused and that the widened design be sufficiently decoupled.

| Shipped | Status |
|---|---|
| `recipe.py` — YAML recipe parsing into `Target` + `[Item]` | **Unchanged.** It solves agent-dependency provisioning, which stays in scope (§7) |
| `layer.py` — the five-layer override model | **Unchanged** |
| `installers/` — uv, apt, bin, oneline, embed, claude | **Unchanged.** Each still shells out to the mature tool it wraps |
| `runner.py`, `outcome.py`, `report.py` — selection, conflict detection, reporting | **Unchanged** |
| `registry.py` — installer name → instance | **Unchanged** |
| `cli.py` — check / dry-run / install / bootstrap | **Extended**, not replaced: new sub-commands for domain and zone inspection |

Genuinely new: the filesystem manager (§4), remote access (§5), the agent
environment (§6), the handoff storage entry (§6.2), and metadata (§3.1).

**The decoupling requirement is met structurally**: nothing new imports the
installer machinery, and nothing in the installer machinery learns about domains
or zones. They share the package and the CLI, which is what the task definition's
"reuse the existing module" asks for, and nothing else.

The shipped README also records three v1 limitations — cross-layer
skip-with-warning unimplemented, the workspace layer stubbed, system apt
detect-and-print only. The second of those overlaps §6.1 and is superseded by it;
the other two stay as recorded.

---

## 10. Acceptance criteria

1. A domain is registered, reloaded idempotently, and its kind
   (`handoff_storage` / `playground` / `workspace`) determines its layout.
2. **A path carrying the zone prefix is recognised as governed from the prefix
   alone**, without consulting a registry — verified by checking a path whose
   zone id does not exist.
3. **A write outside the agent's declared zones is blocked by a `PreToolUse`
   hook**, and the agent receives a message saying why. Verified with the tool
   named in `allowed_tools`, to demonstrate the hook and not the allow list is
   what gates.
4. **A read into a governed zone the agent's list does not name is denied; a read
   outside every governed zone is allowed.** The asymmetry of §4.3 is asserted
   directly, not inferred.
5. **A producing agent cannot read a validator's checking standard**, because the
   standard's zone is absent from its list — the validator boundary
   (validator spec §8.1) resolved entirely by §4.2.
6. A weakly-mapped domain does **not** synchronise until sync is explicitly
   called; a strongly-mapped one needs no call.
7. An agent copies a handoff into its playground and works on the copy: the
   stored artefact is byte-identical afterwards, and its digest still verifies
   (handoff spec §10 criterion 5).
8. A handoff written containing a path from a governed zone fails its
   locality-independence check.
9. **An agent is resumable from history + playground + workspace.** Restart it
   and it continues, with nothing else restored.
10. A workspace is a worktree of the main repository, not a clone, and the main
    checkout is unmodified by the agent's work.
11. Every domain an agent can reach has a permission hook installed before the
    agent starts — asserted by enumerating reachable domains and checking each,
    not by checking the ones the test knows about.
12. `env_mgr` reads its cluster conventions from a knowledge handoff, and
    changing that handoff changes its behaviour without a code change.
13. **The shipped recipe and installer machinery is untouched**: `pytest
    agent_sys/tests/env_mgr` passes unchanged.

---

## 11. Open questions

| Item | Status |
|---|---|
| **Playground retention** | §6.1 makes the playground part of what a resume needs, and handoff spec §11 notes that nothing sweeps it. Those pull in opposite directions and the resolution — an explicit end-of-run archive, probably — is unspecified |
| **Zone identity across machines** | §4.5 maps paths between local and remote. Whether a zone id is the same on both sides, or whether the mapping translates it, is undecided. Same-id is simpler and assumes the operator laid both sides out identically |
| **Concurrent agents in one domain** | Two agents with the same playground domain will collide. Whether a playground is per-agent by construction, or whether the model needs a lock, is unspecified — and main spec §10's multi-graph question is the same question one level up |
| **Sync direction and conflict** | §4.5's weak mapping is `rsync`, which has a direction. Which side wins when both have changed is not specified, and "the caller decides" is an answer that will produce a data loss eventually |
| **Auto-detection scope** | §3.1 prefers auto-detection. What is actually detectable — NFS yes, worktree yes, sync strength probably, remote reachability only by trying — has not been enumerated, and the boundary decides how much configuration an operator must write |
| **System-level tasks** | §7 concludes that maintaining cluster conventions implies system-level tasks, agents, and handoffs. None is specified. They are the first thing that will need one |
