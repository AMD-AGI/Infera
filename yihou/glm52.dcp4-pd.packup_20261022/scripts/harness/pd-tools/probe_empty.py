import json, urllib.request, sys
R = sys.argv[1]; res = []
for i, p in enumerate(["Hello, who are you?", "Write a haiku about the sea.", "List three colors.", "What is 2+2?", "Explain TCP in one sentence."] * 2):
    for stream in (False,):
        body = {"model": "glm5.2-mxfp4", "messages": [{"role": "user", "content": p}], "max_tokens": 64 if i < 5 else 1, "temperature": 0}
        req = urllib.request.Request(R + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        j = json.load(urllib.request.urlopen(req, timeout=300)); m = j["choices"][0]["message"]
        print(i, "max_tokens", body["max_tokens"], "finish", j["choices"][0].get("finish_reason"), "content_len", len(m.get("content") or ""), "reasoning_len", len(m.get("reasoning_content") or ""), "completion_tokens", j["usage"]["completion_tokens"])
