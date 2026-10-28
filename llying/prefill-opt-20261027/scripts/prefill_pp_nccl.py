"""PP4 prefill: when the hidden-state (proxy) NCCL transfer of each microbatch actually runs, from the torch-profiler
traces of all PP ranks (same node clock, step i = same microbatch on every rank). For hop r -> r+1 and step i:
  post  = CPU time rank r posted the proxy send of step i (record_function "send_proxy_dict_to_next_stage")
          minus rank r's GPU end of step i (< 0: posted while the forward still ran, as intended)
  sendk = GPU start of that send kernel minus rank r's GPU end of step i (the NCCL kernel waits on the forward)
  recvk = end of rank r+1's matching recv kernel minus rank r's GPU end of step i
  start = rank r+1's main-stream start of step i minus the recv kernel end
NCCL kernels are matched to the python call that launched them through the runtime launch correlation id.
  python3 prefill_pp_nccl.py TRACE_PP0 .. TRACE_PP3"""

import argparse
import bisect
import gzip
import json
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("traces", nargs=4)
a = ap.parse_args()


def load(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    ann = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")), key=lambda e: e["ts"])
    tid = ann[0]["tid"]
    ext = [e for e in ann if not e["name"].startswith("step[DRAFT")]
    ext_ts = [e["ts"] for e in ext]
    ann_ts = [e["ts"] for e in ann]
    ann_step = [bisect.bisect_right(ext_ts, e["ts"]) - 1 for e in ann]
    gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cnt = {}
    for k in gpu:
        if "nccl" not in k["name"].lower():
            cnt[k["args"].get("stream")] = cnt.get(k["args"].get("stream"), 0) + 1
    main = max(cnt, key=cnt.get)
    ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
    rt = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
          and "correlation" in e.get("args", {}) and e.get("tid") == tid}
    g0, g1 = [None] * len(ext), [None] * len(ext)
    for i, k in enumerate(ops):
        lt = rt.get(k["args"].get("correlation"))
        if lt is None:
            continue
        ai = bisect.bisect_right(ann_ts, lt) - 1
        if ai < 0 or lt > ann[ai]["ts"] + ann[ai]["dur"] or ann_step[ai] < 0:
            continue
        si = ann_step[ai]
        en = min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"]
        g0[si] = k["ts"] if g0[si] is None else min(g0[si], k["ts"])
        g1[si] = en if g1[si] is None else max(g1[si], en)
    # python-side intervals that launch proxy sends / receives
    send_iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("tid") == tid and e.get("ph") == "X"
                     and e["name"] == "send_proxy_dict_to_next_stage")
    recv_iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("tid") == tid and e.get("ph") == "X"
                     and e.get("cat") == "python_function" and re.search(r"scheduler_pp_mixin\.py\(\d+\): _pp_recv_proxy_tensors$", e["name"]))
    nccl = sorted((k for k in gpu if "nccl" in k["name"].lower()), key=lambda k: k["ts"])

    def kernels_in(iv):
        out = []
        starts = [s for s, _ in iv]
        for k in nccl:
            lt = rt.get(k["args"].get("correlation"))
            if lt is None:
                continue
            j = bisect.bisect_right(starts, lt) - 1
            if j >= 0 and lt <= iv[j][1]:
                out.append((j, k["ts"], k["ts"] + k["dur"]))
        by = {}
        for j, s, e in out:
            x = by.setdefault(j, [s, e])
            x[0], x[1] = min(x[0], s), max(x[1], e)
        return by

    sk, rk = kernels_in(send_iv), kernels_in(recv_iv)

    def step_of(t):
        return bisect.bisect_right(ext_ts, t) - 1

    send = {}  # step -> (post time, kernel start, kernel end); the send of step i is posted after step i's launch
    for j, (s, e) in enumerate(send_iv):
        if j in sk:
            send[step_of(s)] = (s, sk[j][0], sk[j][1])
    recv = {}  # step -> (kernel start, kernel end); the proxy recv of step i comes right before step i's launch
    for j, (s, e) in enumerate(recv_iv):
        if j in rk:
            recv[step_of(e) + 1] = (rk[j][0], rk[j][1])
    del ev, gpu, ops, rt, nccl
    return dict(n=len(ext), g0=g0, g1=g1, send=send, recv=recv)


R = [load(p) for p in a.traces]
ms = lambda x: x / 1e3


def summ(name, xs):
    xs = sorted(xs)
    if not xs:
        return f"{name} n=0"
    return f"{name} p50 {st.median(xs):7.1f} mean {st.mean(xs):7.1f} p90 {xs[int(.9 * len(xs))]:7.1f}"


for r in range(3):
    up, dn = R[r], R[r + 1]
    post, sendk, sende, recvk, start, n = [], [], [], [], [], 0
    for i in range(1, min(up["n"], dn["n"])):
        if up["g1"][i] is None or dn["g0"][i] is None or i not in up["send"] or i not in dn["recv"]:
            continue
        if dn["g0"][i] - up["g1"][i] > 5e6:
            continue
        n += 1
        ps, ks, ke = up["send"][i]
        rs, re_ = dn["recv"][i]
        post.append(ms(ps - up["g1"][i]))
        sendk.append(ms(ks - up["g1"][i]))
        sende.append(ms(ke - ks))
        recvk.append(ms(re_ - up["g1"][i]))
        start.append(ms(dn["g0"][i] - re_))
    print(f"hop PP{r}->PP{r + 1} ({n} steps):")
    print(f"   {summ('send posted - up GPU end', post)}")
    print(f"   {summ('send kernel start - up GPU end', sendk)}")
    print(f"   {summ('send kernel duration', sende)}")
    print(f"   {summ('down recv kernel end - up GPU end', recvk)}")
    print(f"   {summ('down GPU start - recv kernel end', start)}")
