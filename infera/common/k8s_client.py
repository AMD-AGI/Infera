###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Minimal in-cluster Kubernetes API access over httpx.

Mirrors the dependency-light style already used for the etcd v3 JSON gateway
(``infera.common.discovery``): no ``kubernetes`` client / no protobuf, just
httpx against the API server using the pod's mounted ServiceAccount token + CA.

Used by the ``kubernetes`` discovery backend (server watches worker Pods) and
the worker-side self-registration (PATCH own Pod annotation). The control-plane
etcd is never touched directly — discovery rides the API server, exactly like
the dynamo ``discoveryBackend: kubernetes`` model.
"""

from __future__ import annotations

import os

import httpx

_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
_TOKEN_PATH = os.path.join(_SA_DIR, "token")
_CA_PATH = os.path.join(_SA_DIR, "ca.crt")
_NAMESPACE_PATH = os.path.join(_SA_DIR, "namespace")


def in_cluster_namespace(default: str = "default") -> str:
    """Read the pod's namespace from the mounted ServiceAccount, else default."""
    try:
        with open(_NAMESPACE_PATH, encoding="utf-8") as fh:
            ns = fh.read().strip()
            return ns or default
    except OSError:
        return default


def api_server_base() -> str:
    """Return the in-cluster API server base URL from the standard env vars."""
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    return f"https://{host}:{port}"


def _read_token() -> str:
    with open(_TOKEN_PATH, encoding="utf-8") as fh:
        return fh.read().strip()


def make_client(*, timeout: float | None = 10.0) -> httpx.AsyncClient:
    """Build an httpx client authenticated to the in-cluster API server.

    ``timeout=None`` is used for the long-lived watch stream; a finite timeout
    for one-shot list/patch calls.
    """
    token = _read_token()
    verify: object = _CA_PATH if os.path.exists(_CA_PATH) else True
    return httpx.AsyncClient(
        base_url=api_server_base(),
        headers={"Authorization": f"Bearer {token}"},
        verify=verify,
        timeout=timeout,
    )
