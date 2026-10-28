Closing as superseded by the merged #41395 after reviewing upstream main `449617bc94a5d4ae23ad62d0da2f842e711ca983`.

`TransferKVChunk.wait_event`, one-time forwarding from `CommonKVSender.send`, queue preservation, and Mooncake worker synchronization before reading/staging KV are already present. #41395 also added regression coverage for event propagation, wait-before-read, abort-during-wait, failed rooms, and the no-event path.

The remaining hunk here records an event in `process_batch_result_disagg_prefill`. On current main this function already synchronizes `result.copy_done` before its overlap-mode chunk send; scheduler result readback is ordered after `forward_stream`. Adding another event at that site would be redundant.

This does not claim all PP ordering is solved. The distinct PP early-send gap is tracked in #45524, and non-CUDA/NIXL event handling is tracked in #43299. I will track those rather than turn this older PR into a competing fix. The original cross-node results remain historical evidence, not fresh validation of current main.
