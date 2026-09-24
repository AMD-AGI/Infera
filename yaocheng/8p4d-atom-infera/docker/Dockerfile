# Infera ATOM image: infera, its KV-event site hook and the host-libionic
# entrypoint on top of an ATOM nightly. The base already ships Mooncake
# v0.3.14-rc1 built with multi-protocol support; keep it rather than rebuilding
# an older pin. ATOM PD is routed by the Python router only, so no Rust router.
ARG ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202609221542
FROM ${ATOM_BASE_IMAGE}

# One diff per fix: atom-*.diff apply to /app/ATOM, infera-*.diff to /opt/infera.
COPY yaocheng/8p4d-atom-infera/patch/ /tmp/patch/
RUN for p in /tmp/patch/atom-*.diff; do git -C /app/ATOM apply "$p"; done

WORKDIR /opt/infera
COPY pyproject.toml README.md ./
COPY infera ./infera
# The base's opentelemetry-proto requires protobuf<7.
RUN for p in /tmp/patch/infera-*.diff; do git apply "$p"; done \
    && pip install --no-cache-dir ".[atom]" "protobuf<7" \
    && rm -rf /tmp/patch

# ATOM owns its BlockManager in a spawned EngineCore, so the KV-event publisher
# is installed from a site .pth; it stays inactive unless
# INFERA_ATOM_KV_EVENTS_ENDPOINT is set.
RUN SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    && echo "import infera.engine.atom.hooks.kv_event_bootstrap" \
        > "${SITE_PKGS}/zzz_infera_atom_bootstrap.pth"

COPY deploy/docker/scripts/infera_inject_host_ionic.sh /usr/local/bin/infera-inject-host-ionic
ENTRYPOINT ["/usr/local/bin/infera-inject-host-ionic"]
CMD ["/bin/bash"]
