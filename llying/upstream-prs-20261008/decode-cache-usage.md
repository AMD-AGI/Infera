## Usage

Build from this PR's checkout:

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  --build-arg APPLY_SGLANG_DECODE_RADIX_SPEC_PATCH=1 \
  -t infera-sglang:mtp-decode-cache .
```

Run the following **inside that rebuilt mi35x container**, with model files and
ROCm/RDMA devices available. These are Decode launch commands; a matching
Prefill worker, etcd and a Router using HTTP requests/ZMQ KV events must already
be running. Replace the example model path and addresses with your deployment's
values. The Prefill worker must advertise the same served model name.

Common Decode arguments (Bash):

```bash
MODEL_PATH=/models/GLM-5.2-MXFP4
DECODE_IP=10.0.0.12
ETCD_ENDPOINT=10.0.0.10:2379

export SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1

DECODE_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name glm5.2-mxfp4
  --trust-remote-code
  --host 0.0.0.0 --port 30000
  --advertise-host "$DECODE_IP"
  --discovery-backend etcd --etcd-endpoint "$ETCD_ENDPOINT"
  --request-transport http
  --enable-kv-events --kv-events on --kv-event-transport zmq
  --disaggregation-mode decode
  --disaggregation-transfer-backend mooncake
  --tp-size 8 --dp-size 8 --enable-dp-attention
  --speculative-algorithm EAGLE
  --speculative-eagle-topk 1
  --speculative-num-steps 5 --speculative-num-draft-tokens 6
  --mem-fraction-static 0.85
  --enable-metrics
  --json-model-override-args '{"index_share_for_mtp_iteration":false}'
)
```

Choose one of the two modes below.

**Decode radix + MTP, without HiCache:**

```bash
python -m infera.engine.sglang "${DECODE_ARGS[@]}"
```

KV events cause Infera to append the Decode radix flag. The patched SGLang hook
accepts the explicit MTP opt-in for EAGLE/NEXTN top-k 1.

**Decode radix + HiCache + MTP:**

```bash
python -m infera.engine.sglang "${DECODE_ARGS[@]}" \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5 \
  --hicache-write-policy write_through \
  --hicache-io-backend kernel \
  --hicache-mem-layout page_first
```

HiCache's required radix flag is set **before** `ServerArgs.from_cli_args`
performs cache compatibility checks, and is also forwarded to the engine.
Users do not need to add `--disaggregation-decode-enable-radix-cache` manually.
An explicit HiCache request implies this prerequisite even with KV events off;
this does not turn KV-event publication back on. Explicit radix flags remain
supported and are not duplicated.

The HiCache settings above reproduce the earlier configuration; size the host
pool for the node's memory. They are not a performance recommendation: the
historical C40 `write_through` test reduced total-token throughput by about 4.5%
and increased mean ITL by about 6.2% compared with Decode radix alone.

An old benchmark harness may still reject `DECODE_MTP=1` plus `DECODE_HICACHE=1`
before invoking Infera. These direct engine commands do not pass through that
harness guard; updating such scripts is separate from engine support.
