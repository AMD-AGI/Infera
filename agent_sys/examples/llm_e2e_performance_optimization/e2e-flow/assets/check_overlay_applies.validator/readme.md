# check_overlay_applies

Completeness, strong. The mount plan is well formed, the files it would mount are
in the handoff, they hash as the plan says, and they differ from what they
replace.

**The last clause is the rule that earns the validator.** A patch that applies
cleanly and changes nothing gives two byte-identical arms, and everything
downstream then passes for the wrong reason: the pipeline compares the stock
deployment against itself and reports no regression. There is no symptom of this
anywhere else in the graph, and one hash comparison here catches it.

Python files get a `compile()`. A syntax error in a mounted file takes the worker
down during model import, fifteen minutes later, where it reads as a
model-loading failure and sends the reader to the wrong place.

## What it cannot check

This body runs on the login node and the mounts point at node-local files it
cannot see. So it checks the plan and the copies the handoff carries. Whether the
node-local file is what ends up inside the running container is a different
question, asked of the container itself, by `check_patch_live`.

---

## What changed on the way into `e2e-flow`

`container_roots_from` names the whitelist, and it is loaded **from the package
rather than from the handoff** so a producer cannot widen its own bounds by
shipping a roots file with `/` in it. Without the whitelist this body cannot tell
a container path from a host path, so a missing or unreadable roots file is a
refusal rather than a fallback.

That is the third rule, and it is the one with no other symptom: a patch naming a
host path has confused the machine it was cut on with the image it is to be
applied to, the mount is still made, the container still starts, the engine still
imports the file it always imported, and the run reports no regression for a
change that was never applied.
