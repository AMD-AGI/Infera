"""bench.py <port> <req.jsonl> <mode greedy|sampled> <out.jsonl> [concurrency]
Sends chat completions with return_spec_tokens_details; records completion tokens, verify steps,
correct-draft histogram, and the output text per request."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, reqf, mode, outf = sys.argv[1:5]
conc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
reqs = [json.loads(l) for l in open(reqf)]

def one(r):
    body = {"model": "default", "messages": r["messages"], "max_tokens": r["max_tokens"], "ignore_eos": r["ignore_eos"],
            "stream": False, "return_spec_tokens_details": True}
    if mode == "greedy":
        body["temperature"] = 0.0
    else:
        body["temperature"] = 1.0; body["top_p"] = 0.95; body["seed"] = 1234
    t = time.time()
    try:
        x = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
        d = json.loads(x.read())
    except Exception as e:
        return {"id": r["id"], "error": repr(e)}
    dt = time.time() - t
    sd = (d.get("sglext") or {}).get("spec_tokens_details") or {}
    msg = d["choices"][0]["message"]
    n = d["usage"]["completion_tokens"]; v = sd.get("spec_verify_ct", 0)
    return {"id": r["id"], "mode": mode, "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": n, "verify": v,
            "acc": n / max(v, 1), "hist": sd.get("spec_correct_drafts_histogram"), "srv_accept_length": sd.get("spec_accept_length"),
            "finish": d["choices"][0].get("finish_reason"), "wall": dt,
            "reasoning": (msg.get("reasoning_content") or "")[:1500], "content": (msg.get("content") or "")[:1500]}

with ThreadPoolExecutor(conc) as ex, open(outf, "w") as fo:
    T = tn = tv = 0
    for res in ex.map(one, reqs):
        fo.write(json.dumps(res) + "\n"); fo.flush()
        if "error" in res:
            print(res["id"], "ERROR", res["error"], flush=True); continue
        tn += res["completion_tokens"]; tv += res["verify"]
        print(f"{res['id']} {mode} in={res['prompt_tokens']} tok={res['completion_tokens']} ver={res['verify']} acc={res['acc']:.2f} "
              f"srv={res['srv_accept_length']} fin={res['finish']} wall={res['wall']:.1f}s", flush=True)
    print(f"TOTAL {mode} tok={tn} verify={tv} accept_len={tn/max(tv,1):.3f}", flush=True)
