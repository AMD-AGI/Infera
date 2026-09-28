# Build from this task directory:
#   docker build -f docker/Dockerfile --build-arg BASE=<base image> -t <tag> .
# Adds the decode-radix-spec opt-in; behavior is unchanged unless the decode
# container sets SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1.
ARG BASE=infera-sglang:v0519-yihou-0917-nextnfix-hicache
FROM ${BASE}
COPY patches/01-sglang-decode-radix-allow-eagle.patch /tmp/p01.patch
COPY patches/02-infera-decode-radix-spec-opt-in.patch /tmp/p02.patch
# Infera exists twice: /opt/infera (the WorkingDir, first on sys.path for the
# engine's `python3 -m`) and the installed site-packages copy (used from any
# other directory). Patch both so the result does not depend on the cwd.
RUN cd /sgl-workspace/sglang && git apply /tmp/p01.patch \
 && cd /opt/infera && git apply --include='infera/*' /tmp/p02.patch \
 && cd "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
 && git apply --include='infera/*' /tmp/p02.patch \
 && rm /tmp/p01.patch /tmp/p02.patch
