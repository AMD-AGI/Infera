# REPRODUCE

1. Image (on the prefill node; network needed for the base image, Mooncake, libionic, pip, cargo):
   `git archive --format=tar.gz -o infera-fdbafb60.tar.gz fdbafb60` (AMD-AGI/Infera) next to `image/build_ll.yihou.sh`, edit
   `D`/`C` paths, run it on the prefill node: builds `infera-sglang:v0522-fdbafb60` and `docker save | ssh <decode> docker load`.
2. Kit: copy `kit/sglang_2p1d_glm5.2` to a path visible at the same path on both nodes (we used NFS); InferenceX 918524ff
   under `$WORK_DIR/cache/InferenceX` (or let the kit clone it).
3. Site: edit a `scripts/llr*.site.env` (node IPs, DECODE_SSH, WORK_DIR, GPUs, CTR), then on the prefill node:
   `setsid nohup bash scripts/run_llr.yihou.sh <site.env> <run-id> > <log> 2>&1 &`
   (it waits for both nodes to have no running containers, removes only exited containers named `$CTR-*`, runs
   `engine/up.sh` + `engine/agentx/agentx.sh`, saves `docker logs` into the run dir, docker-stops).
4. Summary: `python3 -I scripts/llsum.yihou.py $WORK_DIR/runs/<run-id>`.
