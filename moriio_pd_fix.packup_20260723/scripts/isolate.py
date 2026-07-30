import json,urllib.request
MODEL="DeepSeek-V4-Pro"
def ask(url,p,n=8):
    body=json.dumps({"model":MODEL,"prompt":p,"max_tokens":n,"temperature":0}).encode()
    req=urllib.request.Request(f"{url}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=180) as r:
        return json.load(r)["choices"][0]["text"]
ROUTER="http://10.2.122.10:8000"
PDIRECT="http://10.2.122.10:30001"
tests=["The capital of France is","The capital of Japan is","Water is made of","The opposite of hot is","2 plus 2 equals"]
for p in tests:
    pd=ask(ROUTER,p); dr=ask(PDIRECT,p)
    print(f"[{p!r}]\n   PD    -> {pd!r}\n   Pdir  -> {dr!r}")
