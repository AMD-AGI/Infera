FROM infera-sglang:v0519-yihou-0917-nextnfix-hicache

COPY docker/apply_request_trace_patch.py /tmp/apply_request_trace_patch.py
RUN python3 /tmp/apply_request_trace_patch.py \
    && python3 -m py_compile \
       /sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py \
    && rm -f /tmp/apply_request_trace_patch.py
