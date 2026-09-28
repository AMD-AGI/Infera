FROM infera-sglang:aus-0922-reqtrace
COPY p01.patch p02.patch /tmp/
RUN cd /sgl-workspace/sglang && git apply /tmp/p01.patch \
 && cd /opt/infera && git apply --include='infera/*' /tmp/p02.patch \
 && cd "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
 && git apply --include='infera/*' /tmp/p02.patch \
 && rm /tmp/p01.patch /tmp/p02.patch
