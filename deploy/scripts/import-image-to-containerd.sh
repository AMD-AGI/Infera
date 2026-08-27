#!/usr/bin/env bash
# Import a locally built image into a node's containerd, without root on the node.
#
# Kubelet does not ask the docker daemon what images exist -- it asks containerd,
# and only inside containerd's `k8s.io` namespace. So on a containerd cluster
# (RKE2, k3s) a `docker build` result is invisible to the scheduler and the Pod
# fails with ErrImageNeverPull on an image that is plainly there.
#
# With root on the nodes, one line does it and you do not need this script:
#
#   docker save IMAGE | sudo ctr -n k8s.io images import -
#
# This is for the case with cluster-admin but no host root: a privileged Pod per
# node mounts the docker and ctr binaries plus both sockets and pipes `docker save`
# straight into `ctr images import`, so no multi-GB intermediate tar is written.
#
#   deploy/scripts/import-image-to-containerd.sh IMAGE:TAG NODE [NODE...]
#
# The image must already be in *that node's* docker image store -- this moves it
# between two runtimes on one host, it does not copy between hosts.
set -euo pipefail

usage() {
  # The header comment above is the usage text; print it up to the first line
  # that is not a comment, so the two cannot drift apart.
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
  cat <<'EOF'

Environment:
  CTR_SOCK     containerd socket on the node.
               Default: /run/k3s/containerd/containerd.sock (RKE2, k3s).
               Plain containerd: /run/containerd/containerd.sock
  CTR_BIN      ctr binary on the node.     Default: /usr/bin/ctr
  DOCKER_BIN   docker binary on the node.  Default: /usr/bin/docker
  NAMESPACE    namespace for the importer Pods. Default: default
  HELPER_IMAGE image the Pod itself runs; must be pullable by the node.
               Default: docker.io/library/busybox:1.36
  TIMEOUT      seconds to wait per node for the import itself. Default: 3600
               (a 100 GB image takes several minutes even locally)
  PENDING_GRACE
               seconds to wait for a Pod to start running before giving up on it
               and printing its events. Default: 60. This is separate from
               TIMEOUT because a Pod that cannot start does not need an hour to
               prove it.
  KEEP_PODS    set to 1 to keep the importer Pods after success, for debugging.
EOF
}

case "${1:-}" in
  -h | --help) usage; exit 0 ;;
  "") usage >&2; exit 1 ;;
esac

IMAGE="$1"; shift
[ $# -ge 1 ] || { echo "error: at least one node is required" >&2; usage >&2; exit 1; }
NODES=("$@")

CTR_BIN="${CTR_BIN:-/usr/bin/ctr}"
DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
NAMESPACE="${NAMESPACE:-default}"
HELPER_IMAGE="${HELPER_IMAGE:-docker.io/library/busybox:1.36}"
TIMEOUT="${TIMEOUT:-3600}"
PENDING_GRACE="${PENDING_GRACE:-60}"
KEEP_PODS="${KEEP_PODS:-0}"

command -v kubectl >/dev/null || { echo "error: kubectl not in PATH" >&2; exit 1; }

# The socket path is a property of the node, and nothing readable from here says
# which one it is: the Kubernetes API reports the runtime version but not its
# socket. So this defaults to the RKE2/k3s path and says so, rather than
# pretending to detect it -- a wrong guess surfaces as a Pod that cannot mount
# its hostPath, and the failure message names the alternative.
CTR_SOCK="${CTR_SOCK:-/run/k3s/containerd/containerd.sock}"

# Pod names must be DNS-1123 labels: lowercase alphanumerics and dashes, first and
# last character alphanumeric, 63 characters at most. Two nodes must also not
# collapse onto one name, or the second import overwrites the first and one node
# silently never gets the image -- which truncating a long name reintroduces, so
# the full name's checksum goes on the end rather than being cut off. Worst case
# here is 10 + 40 + 1 + 10 = 61 characters.
pod_name_for() {
  local sanitized sum
  sanitized=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
  sanitized=$(printf '%s' "${sanitized:0:40}" | sed -e 's/^-*//' -e 's/-*$//')
  sum=$(printf '%s' "$1" | cksum | cut -d' ' -f1)
  printf 'imgimport-%s-%s' "${sanitized:-node}" "$sum"
}

echo "image     $IMAGE"
echo "nodes     ${NODES[*]}"
echo "socket    $CTR_SOCK"
echo

pods=()
for node in "${NODES[@]}"; do
  pod=$(pod_name_for "$node")
  pods+=("$pod")
  kubectl -n "$NAMESPACE" delete pod "$pod" --ignore-not-found --wait >/dev/null 2>&1 || true
  kubectl apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  namespace: $NAMESPACE
  labels: {app: infera-image-import}
spec:
  nodeName: $node
  restartPolicy: Never
  containers:
    - name: import
      image: $HELPER_IMAGE
      imagePullPolicy: IfNotPresent
      securityContext: {privileged: true}
      command: ["/bin/sh", "-c"]
      args:
        - |
          set -e
          /host/docker save "$IMAGE" \
            | /host/ctr -a "$CTR_SOCK" -n k8s.io images import -
          /host/ctr -a "$CTR_SOCK" -n k8s.io images ls -q | grep -F "$IMAGE"
          echo IMPORT_OK
      volumeMounts:
        - {name: ctrbin, mountPath: /host/ctr, readOnly: true}
        - {name: dockerbin, mountPath: /host/docker, readOnly: true}
        - {name: dockersock, mountPath: /var/run/docker.sock}
        - {name: ctrsock, mountPath: $CTR_SOCK}
  volumes:
    - {name: ctrbin, hostPath: {path: $CTR_BIN, type: File}}
    - {name: dockerbin, hostPath: {path: $DOCKER_BIN, type: File}}
    - {name: dockersock, hostPath: {path: /var/run/docker.sock, type: Socket}}
    - {name: ctrsock, hostPath: {path: $CTR_SOCK, type: Socket}}
YAML
  echo "[start] $node -> pod/$pod"
done

# Every node runs at once: the imports are independent and each is slow.
failed=0
for i in "${!NODES[@]}"; do
  node="${NODES[$i]}" pod="${pods[$i]}"
  deadline=$((SECONDS + TIMEOUT))
  watch_start=$SECONDS
  phase=""
  started=0
  stuck=0
  while [ $SECONDS -lt $deadline ]; do
    phase=$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    case "$phase" in
      Succeeded | Failed) break ;;
      Running) started=1 ;;
    esac
    # TIMEOUT budgets the import, which is slow and legitimately quiet. A Pod that
    # has not started at all is a different thing, and none of its causes get
    # better by waiting: a nodeName no scheduler can place, a CTR_SOCK that does
    # not exist on the node (FailedMount), a HELPER_IMAGE the node cannot pull.
    # Each of those names itself in the Pod's events, and spending TIMEOUT first
    # only delays reading them.
    if [ "$started" -eq 0 ] && [ $((SECONDS - watch_start)) -ge "$PENDING_GRACE" ]; then
      stuck=1
      break
    fi
    sleep 10
  done

  if [ "$phase" = Succeeded ]; then
    echo "[ok]    $node"
    [ "$KEEP_PODS" = 1 ] || kubectl -n "$NAMESPACE" delete pod "$pod" --wait=false >/dev/null 2>&1 || true
    continue
  fi

  failed=1
  if [ "$stuck" -eq 1 ]; then
    echo "[FAIL]  $node -- never started (phase=${phase:-unknown} after ${PENDING_GRACE}s), Pod kept. Events:"
  else
    echo "[FAIL]  $node (phase=${phase:-unknown}) -- Pod kept for inspection:"
    kubectl -n "$NAMESPACE" logs "$pod" 2>&1 | sed 's/^/          /' || true
  fi
  kubectl -n "$NAMESPACE" describe pod "$pod" 2>&1 | sed -n '/Events:/,$p' | sed 's/^/          /' || true
done

if [ "$failed" -ne 0 ]; then
  cat >&2 <<EOF

At least one import failed. The usual causes, in order:
  - the image is not in that node's docker store (this script moves it between
    runtimes on one host; it does not copy between hosts)
  - CTR_SOCK=$CTR_SOCK is not this cluster's socket. RKE2 and k3s use
    /run/k3s/containerd/containerd.sock, plain containerd
    /run/containerd/containerd.sock. A FailedMount event above says so.
  - CTR_BIN=$CTR_BIN or DOCKER_BIN=$DOCKER_BIN is not where the node keeps it
  - $HELPER_IMAGE is not pullable by the node
EOF
  exit 1
fi

cat <<EOF

Imported on all ${#NODES[@]} node(s). Verify, from a node with ctr access:
  ctr -a $CTR_SOCK -n k8s.io images ls -q | grep -F $IMAGE

Apply the workload now rather than later. Kubelet garbage-collects any image no
Pod references once the node is above the image-GC high threshold (85% disk by
default), and the minimum-age protection is only two minutes -- a large image on
a full node can disappear within that window, after the import reported success.
EOF
