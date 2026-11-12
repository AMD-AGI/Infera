# The 11-09 SGLang ROCm nightly (sglang e60f7e909b) plus Infera 0.3.3 from PyPI; the A/B mounts sglang main + #48047 over it.
ARG BASE=lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261109@sha256:e728a52213cce18493a46ac8ac62d5c9eaee66591941871acd213fe989e55957
FROM ${BASE}
RUN python3 -m pip install --no-cache-dir --no-deps nats-py==2.16.0 amd-infera==0.3.3 \
 && python3 -c "import infera.engine.sglang; from infera.server.launch_rust import _find_binary; print(_find_binary())"
# Host libionic provider at start (Infera's entrypoint, deploy/docker/scripts/infera_inject_host_ionic.sh): without it the
# image's libionic does not match the host ionic ABI, Mooncake finds 0 HCAs and the KV goes over the HIP transport (11-12).
COPY infera_inject_host_ionic.sh /usr/local/bin/infera-inject-host-ionic
RUN chmod +x /usr/local/bin/infera-inject-host-ionic
ENTRYPOINT ["/usr/local/bin/infera-inject-host-ionic"]
