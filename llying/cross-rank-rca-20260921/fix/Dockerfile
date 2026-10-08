# Base image fd7220a5 plus Mooncake faae8dd4 rebuilt with the destination-local
# rail, pinned destination HCA and RC ACK timeout patches. build_image.sh
# asserts the base image ID.
ARG BASE_IMAGE=infera-sglang:v0519-yihou-0917-nextnfix-hicache
FROM ${BASE_IMAGE}
COPY build_mooncake_sglang.sh mooncake-dest-local-rail.diff mooncake-ib-timeout.diff \
    mooncake-dest-pinned-hca.diff /tmp/mc-fix/
RUN MOONCAKE_GIT_REF=faae8dd4a6309c3ecd47e0721a83b0250d686fa2 \
        MOONCAKE_PATCHES="/tmp/mc-fix/mooncake-dest-local-rail.diff /tmp/mc-fix/mooncake-ib-timeout.diff /tmp/mc-fix/mooncake-dest-pinned-hca.diff" \
        bash /tmp/mc-fix/build_mooncake_sglang.sh \
    && so="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')" \
    && strings "$so" | grep -q MC_ENABLE_DEST_LOCAL_RAIL \
    && strings "$so" | grep -q MC_IB_TIMEOUT \
    && rm -rf /tmp/mc-fix
# 18 = 1.07 s, NCCL's default: a sender NIC reading GPU memory across the
# socket stalls past Mooncake's 67 ms under saturation.
ENV MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18
