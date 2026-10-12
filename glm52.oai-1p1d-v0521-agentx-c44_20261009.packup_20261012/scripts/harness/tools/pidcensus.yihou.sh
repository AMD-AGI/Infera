#!/usr/bin/env bash
# Run ON the node (read-only): host PID / namespace PID of every process in the two engine
# containers, their @hsa* abstract sockets (/proc/net/unix, host netns), and namespace-PID
# collisions between the legs. ROCr's default IPC binds "\0hsa<pid>" with the NAMESPACE pid;
# two private PID namespaces on one host netns can collide -> hipIpcGetMemHandle EINVAL.
# Usage: pidcensus.yihou.sh CONTAINER_PREFIX
set -uo pipefail
pfx="${1:?container prefix}"
# Abstract names are NUL-padded (shown as trailing '@'s, seen on 138): keep "@hsa<digits>".
socks="$(grep -ao '@hsa[0-9]*' /proc/net/unix 2>/dev/null | sort)"
declare -A nsof=() rankof=()
for leg in prefill decode; do
  c="$pfx-$leg-0"
  mode="$(docker inspect -f '{{.HostConfig.PidMode}}' "$c" 2>/dev/null)" || { echo "-- $c absent"; continue; }
  echo "-- $c PidMode=${mode:-<private>} legacy_ipc=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" | grep -c '^HSA_ENABLE_IPC_MODE_LEGACY=1$')"
  # Namespace pids come from inside the container: /proc/<hostpid> is not visible when this
  # runs under spur exec (job PID namespace; seen on 102), so NSpid cannot be read from here.
  nproc=0
  while read -r ns comm; do
    [[ -n "$ns" ]] || continue
    nproc=$((nproc + 1))
    nsof[$leg]+=" $ns"
    [[ "$comm" == sglang::schedul* ]] && rankof[$leg]+=" $ns"
    hit="$(printf '%s\n' "$socks" | grep -cx "@hsa$ns")"
    printf '   ns_pid=%-8s @hsa%s=%s  %s\n' "$ns" "$ns" "$hit" "$comm"
  done < <(docker exec "$c" sh -c 'for d in /proc/[0-9]*; do printf "%s %s\n" "${d#/proc/}" "$(cat $d/comm 2>/dev/null)"; done' 2>/dev/null |
           awk '$2 !~ /^(tini|docker-init|sh|bash|sleep|cat)$/')
  echo "   ($nproc processes; host pids: $(docker top "$c" -eo pid 2>/dev/null | awk 'NR>1' | tr '\n' ' '))"
done
coll="$(comm -12 <(tr ' ' '\n' <<< "${nsof[prefill]:-}" | grep . | sort -u) <(tr ' ' '\n' <<< "${nsof[decode]:-}" | grep . | sort -u) | tr '\n' ' ')"
echo "   @hsa sockets in host netns: $(printf '%s\n' "$socks" | grep -c .)  (duplicated names: $(printf '%s\n' "$socks" | uniq -d | tr '\n' ' '))"
rcoll="$(comm -12 <(tr ' ' '\n' <<< "${rankof[prefill]:-}" | grep . | sort -u) <(tr ' ' '\n' <<< "${rankof[decode]:-}" | grep . | sort -u) | tr '\n' ' ')"
echo "   scheduler rank PIDs: prefill [${rankof[prefill]:-} ] decode [${rankof[decode]:-} ]"
echo "   rank-PID overlap prefill/decode: ${rcoll:-none}  ($([[ -z "$rcoll" ]] && echo 'OK: disjoint' || echo 'FAIL: CLR hipIpcOpenMemHandle PID guard -> 201'))"
echo "   ns-PID collisions prefill/decode (all processes): ${coll:-none}"
echo "   (harmless iff no @hsa<pid> socket is bound, i.e. legacy IPC on both legs, or --pid host; with"
echo "    default ROCr IPC on private PID namespaces a shared ns pid is the hipIpcGetMemHandle EINVAL race)"
