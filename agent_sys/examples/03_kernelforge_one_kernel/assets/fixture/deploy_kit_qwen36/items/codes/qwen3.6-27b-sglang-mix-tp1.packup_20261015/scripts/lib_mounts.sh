#!/usr/bin/env bash
# measure_mount_for <root>  -> prints the directory to bind at the same path,
# or fails naming the accepted forms.
#
# Stages after this one `docker exec` into the engine container and cannot add
# mounts, so the container must see the run root and the scratch root
# read-write at the same path inside as outside.
#
# The first three cases are the derivation in
# assets/build_workset.task/measure_in_container.sh, i.e. the forms measured
# accepted by the cluster's docker authorization plugin (spur-authz):
#     /shared_nfs/...            -> -v /shared_nfs:/shared_nfs
#     /home/<user>/...           -> -v /home/<user>:/home/<user>
#     /mnt/m2m_nobackup/...      -> -v <path>:<same path>   (node-local scratch)
#     -v /home:/home             is REFUSED by that plugin.
# The mount is derived from the ROOT, never from $HOME (a validation zone's
# $HOME may be /home, the one refused form).
#
# One more case, and it is a measured fact about the host, not a guess: when the
# daemon runs NO authorization plugin (`docker info` -> Plugins.Authorization is
# null; measured on smc300x-ccs-aus-a16-19 at 2026-10-15T06:50Z) a same-path bind
# of a non-top-level directory is accepted -- it is what this kit's own
# bring-up used. Bare top-level directories are still refused, because binding
# /var or /tmp read-write hands the container every tenant's files.
measure_mount_for() {
  local root="$1" authz
  case "$root" in
    /shared_nfs/*)        echo "/shared_nfs"; return 0 ;;
    /home/*/*)            echo "/home/$(printf '%s' "${root#/home/}" | cut -d/ -f1)"; return 0 ;;
    /mnt/m2m_nobackup/*)  echo "$root"; return 0 ;;
  esac
  case "$root" in
    /|/home|/home/|/tmp|/tmp/|/var|/var/|/var/tmp|/var/tmp/|/data|/data/|/mnt|/mnt/|/opt|/usr|/etc|/root)
      echo "lib_mounts: refusing to bind a shared top-level directory: $root" >&2; return 1 ;;
  esac
  authz=$(docker info --format '{{json .Plugins.Authorization}}' 2>/dev/null)
  case "$authz" in
    null|"[]"|"") echo "$root"; return 0 ;;
  esac
  echo "lib_mounts: cannot derive a mount the docker authorization plugin ($authz) will accept for $root." >&2
  echo "  accepted forms (measured on the cluster): -v /shared_nfs:/shared_nfs | -v /home/<user>:/home/<user> | -v /mnt/m2m_nobackup/...:<same path>" >&2
  echo "  refused: -v /home:/home. Set E2E_KIT_MEASURE_ROOTS to roots under an accepted form." >&2
  return 1
}
