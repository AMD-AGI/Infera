# check_patch_live

Trustworthiness, strong. The only validator here that does not ask whether a
record is complete — it asks whether what the record describes actually happened.

## The failure it exists for

A patch that is mounted and never executed produces two arms with identical
numbers and a comparison that reports "no regression": a green result for a
change that was never tested. Nothing else in the graph can see it. The
deployment is healthy, the eval scores well, the replay is fast — and every one
of those is equally true of a deployment running stock code.

## Three rules, in increasing order of what they prove

**`docker inspect` shows the mount, read-only.** Cheapest; catches the plan never
reaching the `docker run`.

**The file hashes, inside the running container, to `sha256_patched`.** This is
the rule people expect to be redundant and it is the only static one that works.
A bind mount does not change the path inside the container, so
`sglang.srt.models.glm5_next.__file__` reads identically on both arms — measured,
not assumed. `__file__` proves nothing; the hash proves the bytes.

**The declared markers appear in the engine log.** The import marker says the
interpreter compiled the mounted bytes. The first-call marker says a real request
entered the patched code. Only the second answers the question this stage exists
to ask.

## The hole, stated

`runtime_marker` is optional in the contract, so a patch declaring none gets the
first two rules and a printed finding saying what could not be shown. Requiring
markers would mean refusing every KernelForge patch that does not know about this
package. Whether to close it is an open question in `DESIGN.md` section 11.

A third layer was designed and not built: cut a short profiler window on the
patched arm and check the kernel symbols against the stock arm's. The machinery
exists in `profiling-demo`; the cost is a graph-off restart per arm.
