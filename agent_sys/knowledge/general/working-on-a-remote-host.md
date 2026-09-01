# Working on a remote host

**Audience: an AI agent executing a task.** Written from a bring-up that ran on a
different machine from the agent driving it, plus the failures collected on the
way. Every rule here was bought by something going wrong; none is style.

Desensitised: no hostnames, no site paths, no credentials. Where a concrete
value would appear, the shape is given instead.

---

## 1. First question: is the work host the machine your shell is on?

**Look at your tool list before you do anything else.** Not after reading the
task, not after exploring the filesystem — before.

If your tools include a remote-execution tool (a `*_remote_run` /
`*_remote_push` / `*_remote_pull` family, or whatever your harness names them),
then **the work host is probably not the machine your shell runs on**, and those
tools are the whole of how you reach it. If they are absent, the run is local.

**Ask the tool surface where it lands rather than inferring it.** A well-built
remote tool names the far side in its own description; if it does not, call it
with `hostname -f` (or the platform equivalent) and record the answer. Do not
guess from the task text, and do not assume the host named in a document you
were given is the host you are actually connected to — **a document is not a
route.**

**Say which mode you were in, in your report.** A reader of your output cannot
tell local from remote afterwards unless you wrote it down, and the two produce
artefacts that look identical.

### The trap that makes this non-obvious

**A path you can `ls` locally is not evidence that you are on the right
machine.** Shared network exports routinely make the *same* path visible from
both hosts. So "I found the model weights / the dataset / the config" proves
nothing about where you are executing.

Discriminators that actually work, in order of strength:

| | why it works |
|---|---|
| **hardware identity** — GPU architecture, CPU model, device count | cannot be faked by a machine that does not have it |
| **a two-sided process or container listing, taken at the same moment** | present *there* and absent *here* is a differential nobody's prose can forge |
| **`hostname -f` obtained *through the remote tool*** | good, but it is still the far side reporting on itself |
| a path existing | **worthless on a shared mount** |

---

## 2. Executing remotely

**Everything that touches the machine goes through the remote tool.** Hardware,
containers, ports, services, long-running processes. Your own shell is for
reading the repository, thinking, and writing your output.

**Do not reach for `ssh` when you get stuck.** This is the single most common way
a remote task quietly becomes a local one: the remote tool does something
unexpected, `ssh` is right there, it works, and the run no longer demonstrates
anything about the system you were supposed to be exercising. If the remote tool
fails, **report the failure whole before working around it** — a first real
failure through an under-exercised path is usually worth more than a green run.

### Long operations

**Launch detached and poll.** A tool call that blocks through an entire startup
tells you nothing until it is over, and you cannot tell a slow start from a hang
while you are inside it.

    remote_run(["bash", "-lc", "setsid nohup <cmd> > <log> 2>&1 < /dev/null & disown"])
    then poll the log

**Measure any timeout before relying on it.** Layers between you and the far side
may cap a tool call. Do not assume, and do not extrapolate from a short call to a
long one — state the limit you actually measured.

**A startup that repeats "health check failed" is usually not a hang.**
Just-in-time compilation, kernel autotuning and graph capture can run for many
minutes on first start. Killing one of these and retrying is how an hour
disappears. If you need progress, watch the build/cache directory or the log's
byte count, not the absence of a success line.

---

## 3. Paths: yours and theirs are different

**Your working directory is not a valid remote working directory** unless
something has told you the two are the same. Passing a local path as a remote
`cwd` is a classic failure and it fails *silently* when the path happens to exist
on both sides.

Two configurations, and they behave differently:

**Shared filesystem (both sides see one tree).** The same path string works on
both sides. This enables a genuinely useful division: **author locally with your
ordinary file tools, execute remotely.** Write your scripts into your own
workspace, then run them through the remote tool using the same path. Zero bytes
are copied. Your editing tools stay fast and local; only execution crosses.

**Separate filesystems (bytes must move).** Your local root and the remote root
are different strings, and something has to synchronise them. Then:

- **The mirror is a snapshot taken before your work, not of its result.** Files
  you create remotely after the copy do not appear locally, and vice versa. Check
  which direction, and when, before treating either side as authoritative.
- **A file-count difference is not automatically a truncated copy.** Resolve it
  by differential listing — *which* files differ — not by comparing totals. Note
  that directory-size tools count directory inodes differently across filesystem
  types, which manufactures a plausible-looking gap out of nothing.
- **Do not synchronise large read-only inputs.** If a model, dataset or image is
  already reachable on the far side, reference it in place. Copying tens of
  gigabytes to prove that copying works is not the experiment.

---

## 4. A remote host usually belongs to other people too

Assume you are a guest and that someone else's job is running.

- **Record the neighbours' state on arrival, in both directions.** What is
  already running, *and* what is already stopped. Otherwise at teardown a stopped
  neighbour reads as "we destroyed it", and a container that appeared while you
  worked reads as "we left something running". Both misreadings are cheap to
  pre-empt and impossible to disprove afterwards.
- **Ports collide.** The conventional port for anything popular is often already
  taken. **Parameterise every port**; never hardcode one. Check which are free
  before you start, and pick from an unusual range.
- **Identify resource occupancy by process, not by index.** Device indices are
  not one namespace: what a container is told, what the kernel driver reports,
  and what a monitoring tool prints can be three different numbers for the same
  device. Cross-reference the PID holding the resource against the PIDs inside
  your own container. Reasoning across two index namespaces as though they agree
  produces confident, wrong attributions — including "a neighbour is using this"
  when it is you.
- **Delete nothing you did not create.** On a shared export the filesystem will
  not stop you. Never write a recursive delete whose target is a variable: one
  unset variable turns it into a catastrophe. Name paths, or delete nothing.
- **Disk is shared and may be nearly full.** Check free space before copying, and
  prefer local scratch over a shared export for anything large or temporary.

---

## 5. Evidence, because most of it is perishable

**Capture two-sided readings while the thing is up.** "Running there, absent
here" can only be taken during the window in which it is true. Once you tear
down, it is gone and cannot be reconstructed from any log.

**Anything may be torn down and restarted, including by you.** A restart that
clears logs and results is often the *right* thing to do — it makes the shipped
scripts byte-identical to the ones that produced the evidence — but it destroys
the previous run's record. Save what matters to a location outside the working
tree before you restart.

**Read the artefact, not the exit code.** A pipeline reports the *last*
command's status, so `cmd | tail` hides `cmd`'s failure. A suite can report
"all passed" over work that produced nothing. Open the file.

**Assume your instrument is answering a different question than the one you
asked.** This is the most common failure by a wide margin, and it fails
*silently*, usually in the reassuring direction — "still loading", "nothing
running", "no violation" — because nobody investigates good news. Observed
instances, all real:

- a `grep` matching a section header instead of a data row, printing a banner and
  no rows, which reads as "nothing is running";
- a process search matching a defunct (zombie) entry, reporting a dead server as
  alive;
- a process search matching its own command line;
- a filter applied to a field that does not carry the discriminating information
  (matching a container's *name* when the identifying token is in its *image*);
- a port checked from a previous run's configuration rather than this run's, so
  silence was read as "not started yet" when the service was up elsewhere;
- a substring match where a word or field match was meant.

**Guard the instrument, not just the subject.** If a check can return "nothing
found", first assert that it *can* find something — otherwise a reader that
silently returns nothing makes every downstream assertion pass.

**Agreement between tools is not corroboration if they share a cache.** Several
readers can return the same wrong answer from one stale cache entry. On a
networked filesystem, attribute caching means **"I read it and it said X"
carries an implicit "as of up to a minute ago"**. When a read contradicts
something you expect, re-read after forcing a fresh fetch before concluding.

**When a report and an artefact disagree, enumerate hypotheses on both sides.**
It is natural to list ways the *other* party failed; make sure at least one
candidate is *my instrument is wrong*. And do not accept someone else's list of
hypotheses as complete — if you hold the cheapest instrument for the question,
use it.

---

## 6. Reporting a remote run

State plainly:

1. **Which mode** — local or remote — and the evidence, not the assumption.
2. **Which machine**, established by hardware identity or a differential
   reading, not by a path that exists on both.
3. **What the run does *not* show.** If a configuration copied no bytes, it says
   nothing about the copy path. If a capability was disabled, say so. An
   overclaimed result is worse than none, because the next person builds on it.
4. **Anything that had to differ** from the recipe you were given — that list is
   often the most useful thing you produce.

### If you produce a reproduction kit

**A kit is judged by what it made into parameters, not by whether it re-runs
where it was written.** Someone reproducing it will be on a different machine,
with different images available, different writable directories and different
free ports. Every one of those must be an overridable variable with a documented
default, and every shared-namespace identifier — ports, container names, run
tags — must be parameterised or the second copy collides with the first.

**Do not fix an identifier a reader will have to change.** Test this by asking:
if two people ran this kit on one host at the same time, what breaks?
