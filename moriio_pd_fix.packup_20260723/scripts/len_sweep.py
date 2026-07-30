import json,urllib.request
ROUTER="http://10.2.122.10:8000"; MODEL="DeepSeek-V4-Pro"
# prompts of increasing token length; each factual so garbage is obvious
tests=[
 ("5tok_France","The capital of France is"),
 ("5tok_China","The capital of China is"),
 ("2+2","Question: What is 2+2? Answer:"),
 ("~20tok","The quick brown fox jumps over the lazy dog. The capital city of the country France is"),
 ("~40tok","In the field of geography it is widely known and taught in schools everywhere that the capital city of the European country called France happens to be the city named"),
 ("~80tok","Geography lesson for today. We will review several facts. The sun rises in the east. Water is made of hydrogen and oxygen. The largest ocean is the Pacific. Now, a simple question that every student should know the answer to: the capital city of the country France, a nation in western Europe, is the city called"),
]
def ask(p):
    body=json.dumps({"model":MODEL,"prompt":p,"max_tokens":10,"temperature":0}).encode()
    req=urllib.request.Request(f"{ROUTER}/v1/completions",data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=180) as r:
        return json.load(r)["choices"][0]["text"]
for name,p in tests:
    print(f"[{name}] -> {ask(p)!r}")
