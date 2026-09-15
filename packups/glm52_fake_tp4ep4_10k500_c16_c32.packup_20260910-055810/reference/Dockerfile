FROM lmsysorg/sglang:v0.5.18-rocm720-mi35x

# The fused-kernel stack, all merged:
#   sglang  #31 FlyDSL sparse MLA prefill and decode   (needs aiter #11)
#           #36 unified Triton router on ROCm
#           #37 four-kernel fused DSA indexer
#           #38 fused fp8 q prep
#           #40 FlyDSL sparse MLA decode at 8 q heads
#           #41 ... and prefill
#   aiter   #11 gfx950 FP8 sparse MLA kernels
#           #14 fp8_mqa_logits BLOCK_M selection
#           #13 FlyDSL bf16 a16w16 skinny GEMM (merged after #14)
ARG SGLANG_SHA=402df1e1e453e1e85ec0f5ac4052d36598cc691a
ARG AITER_SHA=2c71811b32c8ce2e1266aedaec199df7d90f597d

# The bashrc line puts the stock aiter ahead of ours in every `docker exec` shell.
RUN pip uninstall -y sglang amd-aiter \
 && rm -rf /sgl-workspace/sglang /sgl-workspace/aiter \
 && sed -i '\|^export PYTHONPATH=/sgl-workspace/aiter:|d' /etc/bash.bashrc

RUN git clone https://github.com/xiaobochen-amd/sglang.git /sglang \
 && cd /sglang && git checkout "$SGLANG_SHA" \
 && cp python/pyproject_other.toml python/pyproject.toml \
 && pip install -e "python[srt_hip]" --no-deps

RUN git clone https://github.com/xiaobochen-amd/aiter.git /aiter \
 && cd /aiter && git checkout "$AITER_SHA" \
 && git submodule sync && git submodule update --init --recursive \
 && AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 python3 setup.py develop

# /sgl-workspace shadows the sglang package when it is cwd.
WORKDIR /
