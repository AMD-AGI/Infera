import json, sys, time, urllib.request
port = sys.argv[1]
prompts = [
 "The history of the Roman Empire can be summarized as follows:",
 "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
 "Here is a detailed explanation of how TCP congestion control works:",
 "Write a short story about a robot who learns to paint.\n\n",
 "The following is a list of the 20 largest countries by area:\n1.",
 "In linear algebra, an eigenvector of a matrix is",
 "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n",
 "Recipe for chocolate chip cookies.\nIngredients:\n",
 "SELECT customers.name, SUM(orders.amount) FROM",
 "Q: What are the main causes of inflation?\nA:",
 "The theory of general relativity, proposed by Albert Einstein,",
 "Translate to French: The weather is nice today and I want to go for a walk in the park.\n",
]
tot_tok = tot_ver = 0; tot_t = 0.0
for i, p in enumerate(prompts):
    body = {"text": p, "sampling_params": {"max_new_tokens": 512, "temperature": 0}}
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
    d = json.loads(r.read()); dt = time.time() - t
    m = d["meta_info"]; n = m["completion_tokens"]; v = m.get("spec_verify_ct", 0)
    tot_tok += n; tot_ver += v; tot_t += dt
    print(f"[{i}] tok={n} verify={v} acc={n/max(v,1):.2f} wall={dt:.2f}s text={d['text'][:80]!r}")
print(f"TOTAL tok={tot_tok} verify={tot_ver} accept_len={tot_tok/max(tot_ver,1):.3f} ms_per_verify_step={1000*tot_t/max(tot_ver,1):.2f} (includes prefill+http)")
