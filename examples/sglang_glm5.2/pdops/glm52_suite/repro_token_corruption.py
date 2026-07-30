#!/usr/bin/env python3
"""复现长上下文冷 prefill 下的偶发 token 损坏（token id 0 / '!'）。详见 KNOWN_ISSUES.md。

每次请求前 flush 一次 prefix cache——这是触发的必要条件：暖缓存下同一 prompt 稳定正确，
只有 KV 重算时才会损坏。

发生率只有百分之几，样本量小时可能一次都不出现。要判定"修好了"，至少跑到 100 次以上
无失败（--iters 12 以上），否则很容易得到假阴性结论。

用法:
  ./repro_token_corruption.py                      # 36 次请求
  ./repro_token_corruption.py --iters 12           # 108 次，用于验证修复
  ./repro_token_corruption.py --lengths 12288      # 只测出现过失败的长度
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import requests

# 刻意选用不含任何数字的平淡散文：这样整段 context 里唯一的数字就是 needle，
# 取回数字即为"确实读到了 needle"的无歧义证据。
FILLER = [
    "The archive room stayed cool even in summer, and the shelves were always full.",
    "Maintenance crews swept the corridors before the building opened each morning.",
    "A narrow window near the stairwell let in a thin band of afternoon light.",
    "The catalogue was reorganised twice, first by subject and later by author.",
    "Visitors often remarked that the reading desks were unusually comfortable.",
    "Rain collected in the courtyard and drained slowly through the old channels.",
    "The east wing housed periodicals never transferred to microfilm.",
    "Librarians kept a spare key behind the counter for the basement door.",
    "Every autumn the heating pipes made a low knocking sound for a week or so.",
    "The reference desk was staffed by rotation, and the schedule changed monthly.",
    "Boxes of uncatalogued correspondence waited in the annex for years.",
    "A hand-drawn floor plan hung near the entrance, faded but still legible.",
    "Study carrels along the north wall were claimed early on busy days.",
    "The garden outside was tended by a volunteer who came on alternate weekends.",
]

QUESTION = (
    "\n\nBased on the text above, what is the secret access code for Ravenna? "
    "Reply with just the number."
)


def build_prompt(length: int, depth: int) -> tuple[str, int]:
    """构造 needle prompt。seed 由 (length, depth) 决定，因此各次迭代的 prompt 完全一致，
    失败是否复现只取决于运行时数值行为。"""
    rng = random.Random(length * 1000 + depth)
    code = rng.randint(1_000_000, 9_999_999)
    needle = f"Important: the secret access code for Ravenna is {code}. Remember this number."
    body = [rng.choice(FILLER) for _ in range(max(4, int(length / 16)))]  # 约 16 token/句
    body.insert(min(int(len(body) * depth / 100), len(body)), needle)
    return " ".join(body) + QUESTION, code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--iters", type=int, default=4, help="每个 (长度,深度) 组合重复次数")
    ap.add_argument("--lengths", default="12288,16384", help="上下文长度，逗号分隔")
    ap.add_argument("--depths", default="10,50,90", help="needle 深度百分比")
    ap.add_argument("--no-flush", action="store_true",
                    help="不 flush cache（对照用，预期不复现）")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    lengths = [int(x) for x in args.lengths.split(",") if x]
    depths = [int(x) for x in args.depths.split(",") if x]

    sess = requests.Session()
    model = sess.get(f"{base}/v1/models", timeout=30).json()["data"][0]["id"]
    print(f"server : {base}")
    print(f"model  : {model}")
    print(f"flush  : {'off (对照)' if args.no_flush else 'on (每次请求前)'}")

    cases = [(l, d) for l in lengths for d in depths]
    total = len(cases) * args.iters
    print(f"计划   : {len(cases)} 组 x {args.iters} 次 = {total} 次请求\n")

    fails = []
    t0 = time.time()
    for it in range(args.iters):
        for length, depth in cases:
            if not args.no_flush:
                sess.post(f"{base}/flush_cache", timeout=60)
            prompt, code = build_prompt(length, depth)
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            r = sess.post(f"{base}/v1/chat/completions", json=body, timeout=args.timeout)
            r.raise_for_status()
            d = r.json()
            content = (d["choices"][0]["message"].get("content") or "").strip()
            if str(code) not in content.replace(",", "").replace(" ", ""):
                fails.append((it, length, depth, content, code))
                print(f"  FAIL iter{it} {length // 1024}k d{depth}: "
                      f"got={content!r} want={code} "
                      f"ctok={d['usage']['completion_tokens']} "
                      f"ptok={d['usage']['prompt_tokens']}"
                      f"{'  <-- token id 0' if '!' in content else ''}")
        print(f"  iter {it} 完成（累计失败 {len(fails)}）")

    ok = total - len(fails)
    print(f"\n{'=' * 60}")
    print(f"结果: {ok}/{total} 通过，{len(fails)} 失败"
          f"（{100.0 * len(fails) / total:.1f}%），耗时 {time.time() - t0:.0f}s")
    if fails:
        print("失败均为长上下文冷 prefill 下首个 decode step 采样到 token id 0，"
              "详见 KNOWN_ISSUES.md")
    elif total < 100:
        print(f"警告: 样本仅 {total} 次，发生率约百分之几，0 失败不足以说明问题已修复。"
              f"验证修复请用 --iters 12 以上。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
