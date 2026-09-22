FROM infera-sglang:v0519-llying-aus-0922-nextnfix-hicache
COPY docker/aus_diag.py /sgl-workspace/sglang/python/sglang/srt/observability/aus_diag.py
COPY docker/apply_patch.py /tmp/aus_apply_patch.py
RUN python3 /tmp/aus_apply_patch.py
