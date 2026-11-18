#!/usr/bin/env python3
"""Cost split of the engine-side chat-template render for AgentX-sized conversations: Jinja (Python, holds the GIL)
vs tokenization (Rust, releases it). Builds the accumulated messages of some main conversations from AIPerf's saved
dataset (deltas only; server responses are not in the dataset, so assistant turns are placeholders) up to ~TARGET
tokens.   render_cost.py DATASET_DIR MODEL [TARGET=110000] [N=5]"""
import json, sys, time
from transformers import AutoTokenizer
ddir, model = sys.argv[1:3]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 110000
n = int(sys.argv[4]) if len(sys.argv) > 4 else 5
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
idx = json.load(open(f"{ddir}/index.dat")); data = open(f"{ddir}/dataset.dat", "rb")
done = 0
for cid, o in idx["offsets"].items():
    if "::" in cid:
        continue
    data.seek(o["offset"]); c = json.loads(data.read(o["size"]))
    msgs = []
    for t in c["turns"]:
        msgs += [m for m in (t.get("raw_messages") or []) if isinstance(m, dict)]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if len(text) > target * 3:
            break
        msgs.append({"role": "assistant", "content": "ok"})
    t0 = time.perf_counter(); text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    t1 = time.perf_counter(); ids = tok.encode(text, add_special_tokens=False); t2 = time.perf_counter()
    print(f"{cid[:12]} messages {len(msgs)} chars {len(text):,} tokens {len(ids):,}: jinja {1e3*(t1-t0):.0f} ms, "
          f"tokenize {1e3*(t2-t1):.0f} ms")
    done += 1
    if done >= n:
        break
