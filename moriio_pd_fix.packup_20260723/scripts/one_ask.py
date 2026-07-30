import json,sys,urllib.request
ROUTER="http://10.2.122.10:8000"; MODEL="DeepSeek-V4-Pro"
prompt=sys.argv[1] if len(sys.argv)>1 else "The capital of France is"
body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":8,"temperature":0}).encode()
req=urllib.request.Request(f"{ROUTER}/v1/completions",data=body,headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req,timeout=180) as r:
    print(repr(json.load(r)["choices"][0]["text"]))
