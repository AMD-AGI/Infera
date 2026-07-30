import json,urllib.request,sys
ROUTER="http://10.2.122.10:8000"; MODEL="DeepSeek-V4-Pro"
p=sys.argv[1]
body=json.dumps({"model":MODEL,"prompt":p,"max_tokens":10,"temperature":0}).encode()
req=urllib.request.Request(f"{ROUTER}/v1/completions",data=body,headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=180) as r:
    d=json.load(r)
print("OUT:",repr(d["choices"][0]["text"]), "| prompt_tok:", d.get("usage",{}).get("prompt_tokens"))
