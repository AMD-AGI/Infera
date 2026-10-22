# Task dcpx 2026-10-22: infera-sglang:v0522-1020-yihou-dcp-mc4340 (437497f5 on 135/136) with Mooncake faae8dd4 rebuilt
# with the same #4340 port PLUS llying's rail patches (cross-rank-rca-20260921/fix): destination-local rail, pinned
# destination HCA, RC ACK timeout. Nothing else changes. Env stays unset here; the harness config sets
# MC_ENABLE_DEST_LOCAL_RAIL / MC_IB_TIMEOUT explicitly.
ARG BASE_IMAGE=infera-sglang:v0522-1020-yihou-dcp-mc4340
FROM ${BASE_IMAGE}
COPY build_mooncake_sglang.mc4340.v0522.yihou.sh mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff \
     mooncake-dest-local-rail.diff mooncake-ib-timeout.diff mooncake-dest-pinned-hca.diff /tmp/yihou-mc-fix/
RUN MOONCAKE_GIT_REF=faae8dd4a6309c3ecd47e0721a83b0250d686fa2 \
        MOONCAKE_PATCHES="/tmp/yihou-mc-fix/mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff /tmp/yihou-mc-fix/mooncake-dest-local-rail.diff /tmp/yihou-mc-fix/mooncake-ib-timeout.diff /tmp/yihou-mc-fix/mooncake-dest-pinned-hca.diff" \
        bash /tmp/yihou-mc-fix/build_mooncake_sglang.mc4340.v0522.yihou.sh \
    && so="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')" \
    && strings "$so" | grep -q MC_ENABLE_DEST_LOCAL_RAIL \
    && strings "$so" | grep -q MC_IB_TIMEOUT \
    && strings "$so" | grep -q "extends past its allocation"
