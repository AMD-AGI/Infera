#!/usr/bin/env python3
"""Concurrent use of one HF fast tokenizer, as the P4D4 API servers and the Python router do, to see whether
tokenizers 0.23.3 / transformers 5.22.0 crash (SIGSEGV / SIGILL) without any GPU or engine involved.
Run inside the image on a compute node (CPU only), e.g.
  docker run --rm -v $MODEL:$MODEL:ro -v $PWD:/s -e PYTHONFAULTHANDLER=1 --entrypoint python3 \
      infera-vllm:v0340-0c5e5e89 /s/tokenizer_stress.py $MODEL 16 1800
  tokenizer_stress.py MODEL [THREADS=16] [SECONDS=1800] [TOKENS=110000]
Each thread loops over: encode of a long text, chat-template tokenization of a two-turn conversation, and
(on odd threads) encode with split_special_tokens / truncation settings different from the even threads, which makes
transformers rewrite the shared Rust tokenizer's state on every call. A clean exit prints the iteration counts.
"""
import faulthandler
import random
import sys
import threading
import time

from transformers import AutoTokenizer

faulthandler.enable(all_threads=True)
model = sys.argv[1]
n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 16
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 110000
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
words = ("def class return import from self None True False for while if else elif try except with as lambda yield "
         "intermittent tokenizer request response stream chunk offload prefill decode cache block").split()
rng = random.Random(0)
texts = [" ".join(rng.choice(words) for _ in range(int(tokens * 0.8))) for _ in range(4)]
counts = [0] * n_threads
stop = time.time() + seconds


def work(i):
    r = random.Random(i)
    while time.time() < stop:
        text = texts[r.randrange(len(texts))]
        if i % 2:
            tok(text, split_special_tokens=True, truncation=True, max_length=tokens // 2)
        else:
            tok(text, add_special_tokens=False)
        msgs = [{"role": "user", "content": text[: len(text) // 2]}, {"role": "assistant", "content": text[len(text) // 2:]}]
        tok.apply_chat_template(msgs, tokenize=True)
        counts[i] += 1


ts = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"no crash in {seconds:.0f} s, {n_threads} threads, iterations {counts}", flush=True)
