#!/usr/bin/env bash
# Stream a local docker image into the node's containerd (namespace k8s.io) so
# kubernetes can run it, without needing root on the host.
#
# RKE2 pods pull through containerd, which does not see the docker daemon's image
# store, and /run/k3s/containerd/containerd.sock is root-owned. A privileged pod
# is the one lever a cluster-admin has that does not need host sudo: it mounts
# both sockets plus the two static Go binaries and pipes `docker save` straight
# into `ctr images import`, so no 100 GB intermediate tarball is ever written.
#
#   bash import-image-to-containerd.sh <image[:tag]> <node> [node...]
set -euo pipefail

IMAGE="${1:?usage: import-image-to-containerd.sh <image:tag> <node>...}"; shift
[ $# -ge 1 ] || { echo "at least one node required" >&2; exit 1; }

DOCKER_BIN="${DOCKER_BIN:-$(command -v docker)}"
CTR_BIN="${CTR_BIN:-/usr/bin/ctr}"
CTR_SOCK="${CTR_SOCK:-/run/k3s/containerd/containerd.sock}"
NS="${NS:-default}"

for node in "$@"; do
  pod="imgimport-${node##*-}"
  kubectl -n "$NS" delete pod "$pod" --ignore-not-found --wait=true >/dev/null 2>&1 || true
  cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  namespace: $NS
spec:
  nodeName: $node
  hostNetwork: true
  restartPolicy: Never
  containers:
    - name: import
      image: docker.io/library/busybox:1.36
      imagePullPolicy: IfNotPresent
      securityContext:
        privileged: true
      command: ["/bin/sh", "-c"]
      args:
        - |
          set -e
          echo "saving $IMAGE from docker and importing into containerd k8s.io"
          /host/docker save "$IMAGE" | /host/ctr -a "$CTR_SOCK" -n k8s.io images import -
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
  echo "[import] $node -> pod/$pod"
done

echo "follow with: kubectl get pod -n $NS -l '' -o wide | grep imgimport"
