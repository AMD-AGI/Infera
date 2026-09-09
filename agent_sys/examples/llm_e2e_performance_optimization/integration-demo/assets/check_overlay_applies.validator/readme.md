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
