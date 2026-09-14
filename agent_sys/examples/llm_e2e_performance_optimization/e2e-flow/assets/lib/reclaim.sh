#!/bin/sh
# Give a container-written output back to the user who owns the zone.
#
# **Every body in this package that runs work inside a container hits this**,
# not one module. Found by m3 on the first real GPU run, and it is the second
# half of a problem whose first half (`__pycache__`) `PYTHONDONTWRITEBYTECODE=1`
# already solves.
#
# The container runs as **root**, and it has to: a framework compiling kernels
# on first call cannot write its cache as a user who does not exist inside the
# image. So every file the body writes into `$AGENT_SYS_OUTPUT_<KIND>` is
# root-owned.
#
# **Reading is fine and that is what makes it easy to miss.** The files are 644,
# so `copy_out` works, the seal works, every validator reads them, and the run
# goes green. What fails is *later*: the zone's own user cannot clean up, and
# the symptom lands on the **next** run rather than the one that caused it —
# m3 had to reclaim their scratch from inside a root container to tidy it.
#
# The honest fix is operational rather than clever: chown from inside the same
# container that created the files, because that is the only context with the
# privilege to do it. `analyze-demo` reached the same conclusion independently.
#
#   reclaim.sh <container-name> <path-inside-container> [<path> ...]
#
# Idempotent, and a no-op when the container is gone — a body should be able to
# call it in a `finally` without deciding first whether it will work.
set -eu

ctr="${1:?usage: reclaim.sh <container> <path-in-container> [...]}"
shift

if ! docker inspect "${ctr}" >/dev/null 2>&1; then
  echo "reclaim: no container ${ctr}; nothing to do" >&2
  exit 0
fi

# The host identity to hand ownership to. Taken from the *runner's* process
# rather than from a variable: a body that got this wrong would hand the files
# to a uid that does not exist and the failure would look identical.
uid="$(id -u)"
gid="$(id -g)"

for path in "$@"; do
  # `|| true` per path: a path the body never created is not an error, and a
  # reclaim that aborts halfway leaves a half-owned tree — worse than the
  # unowned one it was fixing.
  docker exec "${ctr}" sh -c "chown -R ${uid}:${gid} '${path}' 2>/dev/null" || true
  echo "reclaim: ${ctr}:${path} -> ${uid}:${gid}" >&2
done
