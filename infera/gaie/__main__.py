###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Entry point for the GAIE Endpoint Picker (ext_proc) server.

Reuses the same routing brain as the in-process router: it builds the policy
(``kv-aware`` by default), wires worker discovery (kubernetes or etcd) so the
``WorkerPool`` and the kv-aware cache view stay live, and serves the Envoy
ext_proc protocol so an Inference Gateway (kGateway) routes by Infera's
kv-cache locality + load model.

Run:
    python -m infera.gaie \
        --router-tokenizer-path Qwen/Qwen3-0.6B \
        --discovery-backend kubernetes \
        --k8s-label-selector 'infera.amd.com/deployment=<idep>' \
        --nats-server nats://<broker>:4222
"""

from __future__ import annotations

import asyncio
import logging

import grpc

from infera.common.discovery import Registry
from infera.common.discovery_k8s import KubernetesRegistry
from infera.common.tokenizer import resolve_tokenizer_path
from infera.common.worker_pool import WorkerInfo
from infera.gaie.args import parse_epp_args
from infera.gaie.endpoint_picker import EndpointPicker
from infera.gaie.ext_proc_server import ExtProcServicer
from infera.gaie.proto import ext_proc_pb2_grpc as pb_grpc
from infera.router.policy.factory import build_policy

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Health service name the k8s grpc probe checks (matches dynamo-epp.yaml).
_HEALTH_SERVICE = "inference-extension"


def _self_signed_credentials() -> grpc.ServerCredentials:
    """Generate an in-memory self-signed cert/key and return gRPC server TLS
    credentials. Inference Gateways connect to the EPP over TLS with the cert
    verification skipped, so a self-signed cert is sufficient and avoids any
    cert provisioning/secret plumbing for the EndpointPicker."""
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "infera-epp")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("infera-epp")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return grpc.ssl_server_credentials([(key_pem, cert_pem)])


def _build_registry(args, on_added, on_removed):
    if args.discovery_backend == "kubernetes":
        if not args.k8s_label_selector:
            raise SystemExit(
                "--discovery-backend=kubernetes requires --k8s-label-selector "
                "(or $INFERA_K8S_LABEL_SELECTOR)"
            )
        logger.info(
            "kubernetes discovery: selector=%r namespace=%s",
            args.k8s_label_selector,
            args.k8s_namespace or "<pod-namespace>",
        )
        return KubernetesRegistry(
            label_selector=args.k8s_label_selector,
            namespace=args.k8s_namespace,
            on_worker_added=on_added,
            on_worker_removed=on_removed,
        )
    if not args.etcd_endpoint:
        raise SystemExit("--discovery-backend=etcd requires --etcd-endpoint")
    logger.info("etcd discovery: endpoint=%s prefix=%s", args.etcd_endpoint, args.etcd_prefix)
    return Registry(
        endpoint=args.etcd_endpoint,
        prefix=args.etcd_prefix,
        on_worker_added=on_added,
        on_worker_removed=on_removed,
    )


def _add_health_service(server: grpc.aio.Server) -> bool:
    """Register the grpc_health.v1 service (SERVING) on an existing server.
    Returns False if grpcio-health-checking isn't installed."""
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc
    except ImportError:
        logger.warning("grpcio-health-checking not installed; skipping gRPC health service")
        return False
    servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(servicer, server)
    # Mark both the named service and the overall server SERVING.
    servicer.set(_HEALTH_SERVICE, health_pb2.HealthCheckResponse.SERVING)
    servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    return True


async def _serve_health(host: str, port: int) -> grpc.aio.Server | None:
    """Best-effort standalone gRPC health server (grpc_health.v1) on a dedicated
    port, for k8s probes that target a separate health port. Returns None if
    grpcio-health-checking isn't installed."""
    server = grpc.aio.server()
    if not _add_health_service(server):
        return None
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    logger.info("gRPC health serving on %s:%d (service=%s)", host, port, _HEALTH_SERVICE)
    return server


async def main() -> None:
    args = parse_epp_args()

    tokenizer_path = resolve_tokenizer_path(args.router_tokenizer_path)
    policy = build_policy(
        args.router_policy,
        overlap_weight=args.kv_overlap_weight,
        prefill_overlap_weight=args.kv_prefill_overlap_weight,
        decode_overlap_weight=args.kv_decode_overlap_weight,
        tokenizer_path=tokenizer_path,
        kv_event_transport=args.kv_event_transport,
        nats_server=args.nats_server,
    )
    logger.info("router-policy=%s tokenizer=%s", args.router_policy, tokenizer_path)

    # The policy's kv-aware cache view is driven by worker add/remove events.
    def on_worker_added(info: WorkerInfo) -> None:
        try:
            policy.on_worker_added(info)
        except Exception:
            logger.exception("policy.on_worker_added failed for %s", info.worker_id)

    def on_worker_removed(worker_id: str) -> None:
        try:
            policy.on_worker_removed(worker_id)
        except Exception:
            logger.exception("policy.on_worker_removed failed for %s", worker_id)

    registry = _build_registry(args, on_worker_added, on_worker_removed)
    await registry.start()

    picker = EndpointPicker(registry.pool, policy, destination_port=args.destination_port)
    server = grpc.aio.server()
    pb_grpc.add_ExternalProcessorServicer_to_server(ExtProcServicer(picker), server)
    # Serve the gRPC health service on the SAME port as ext_proc: kgateway's
    # InferencePool health-checks the EndpointPicker on its endpointPickerRef
    # port and fails requests closed (HTTP 500) if the probe can't connect.
    _add_health_service(server)
    bind = f"{args.host}:{args.grpc_port}"
    if args.secure_serving:
        server.add_secure_port(bind, _self_signed_credentials())
        mode = "TLS"
    else:
        server.add_insecure_port(bind)
        mode = "plaintext"
    await server.start()
    logger.info(
        "GAIE ext_proc EPP serving on %s:%d (%s, +grpc health)", args.host, args.grpc_port, mode
    )

    # Keep a dedicated health port too (for k8s grpc probes that target it).
    health_server = (
        await _serve_health(args.host, args.grpc_health_port)
        if args.grpc_health_port and args.grpc_health_port != args.grpc_port
        else None
    )

    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        if health_server is not None:
            await health_server.stop(grace=1)
        await registry.stop()
        await policy.aclose()


if __name__ == "__main__":
    asyncio.run(main())
