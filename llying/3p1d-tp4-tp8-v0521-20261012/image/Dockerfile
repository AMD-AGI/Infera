# infera-sglang:v0521-yihou-1010-mc4340 (sglang v0.5.21 recipe, Mooncake faae8dd4 + #4340 port)
# with Mooncake rebuilt from the same ref plus the cross-rank v3 fix from
# llying/cross-rank-rca-20260921/fix: destination-local rail, pinned destination
# HCA, configurable RC ACK timeout. Only engine.so (and libasio) change.
ARG BASE_IMAGE=infera-sglang:v0521-yihou-1010-mc4340
FROM ${BASE_IMAGE}
COPY build_mooncake_sglang.mc4340.yihou.sh \
     mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff \
     mooncake-dest-local-rail.diff mooncake-ib-timeout.diff mooncake-dest-pinned-hca.diff \
     /tmp/mc-fix/
SHELL ["/bin/bash", "-c"]
RUN MOONCAKE_GIT_REF=faae8dd4a6309c3ecd47e0721a83b0250d686fa2 \
        MOONCAKE_PATCHES="/tmp/mc-fix/mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff /tmp/mc-fix/mooncake-dest-local-rail.diff /tmp/mc-fix/mooncake-ib-timeout.diff /tmp/mc-fix/mooncake-dest-pinned-hca.diff" \
        bash /tmp/mc-fix/build_mooncake_sglang.mc4340.yihou.sh \
    && so="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')" \
    && s="$(strings "$so")" \
    && grep -q MC_ENABLE_DEST_LOCAL_RAIL <<<"$s" \
    && grep -q MC_IB_TIMEOUT <<<"$s" \
    && grep -q "extends past its allocation" <<<"$s" \
    && echo "[mc-fix] dest-local-rail + ib-timeout + #4340 present in $so ($(sha256sum "$so" | cut -c1-16))" \
    && rm -rf /tmp/mc-fix /tmp/mooncake-upstream
# 18 = 1.07 s (NCCL's default); Mooncake's built-in 14 = 67 ms stalls under
# saturation when a sender NIC reads GPU memory across the socket.
ENV MC_ENABLE_DEST_LOCAL_RAIL=1 MC_IB_TIMEOUT=18
