"""Build real-text agentic-coding requests: real source files (sglang in the image) + coding tasks."""
import json, random, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou/req_real.jsonl"
S = "/sgl-workspace/sglang/python/sglang/srt/"
SYS = ("You are an expert software engineering agent working in a Python repository (SGLang, an LLM serving engine). "
       "You read code carefully, reason about it, and answer with precise, concrete code. When you propose changes, "
       "give them as unified diffs or complete functions.")
files = ["managers/schedule_batch.py", "managers/scheduler.py", "mem_cache/radix_cache.py", "speculative/eagle_worker_v2.py",
         "layers/sampler.py", "managers/tokenizer_manager.py", "entrypoints/openai/serving_chat.py", "server_args.py",
         "model_executor/model_runner.py", "layers/radix_attention.py", "managers/detokenizer_manager.py",
         "mem_cache/memory_pool.py", "sampling/sampling_params.py", "layers/logits_processor.py",
         "disaggregation/decode.py", "disaggregation/prefill.py"]
tasks = ["Explain in detail what the main class in this file does, its key invariants, and how its methods interact.",
         "Write a comprehensive set of pytest unit tests for the most important functions in this file. Use mocks where needed.",
         "Review this file for bugs, race conditions, and edge cases. For each issue, show the problematic code and a fixed version.",
         "Refactor the longest function in this file into smaller well-named helper functions. Output the full refactored code.",
         "Add complete type hints and Google-style docstrings to every public function in this file. Output the modified code.",
         "Implement a new feature: add a per-request metric that counts how many times the request was preempted or retracted, and expose it in the response. Show the diff."]
rng = random.Random(7)
reqs = []
for i, f in enumerate(files):
    src = open(S + f).read()
    src = src[:240_000]  # ~60k tokens max
    t = tasks[i % len(tasks)]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Here is `python/sglang/srt/{f}`:\n\n```python\n{src}\n```\n\n{t}"}]
    reqs.append({"id": f"rc{i:02d}", "file": f, "task": t[:40], "max_tokens": 1024, "ignore_eos": False, "messages": msgs})
with open(OUT, "w") as fo:
    for x in reqs:
        fo.write(json.dumps(x) + "\n")
print("wrote", len(reqs))
