#!/usr/bin/env python3
"""GLM-5.2-FP8 正确性验证套件（ROCm / MI325X + SGLang，TP=8）。

本文件由 inference_glm5p2/verify_correctness.py 改编而来。原版针对的是打了未合并补丁
PR #48886 的 vLLM；这里换成 SGLang（tilelang DSA backend + fp8_e4m3 KV cache），因此：

  - 默认端口 8100 -> 30000；
  - 代码语料从 vLLM 源码换成 SGLang 源码（本容器里没有 vllm 模块）；
  - weights 检查不再对应某个补丁分支，改为 checkpoint 级的反量化自洽性校验。

为什么不能只做短 prompt 测试
----------------------------
模型的 index_topk 是 2048：任何短于 2048 token 的 prompt，sparse indexer 的 top-k 都会
选中整个 prompt，行为等价于 dense attention。**短 prompt 因此完全测不到 sparse 路径。**
GLM-5.2 的 DSA sparse attention 在 gfx942 上走的是 tilelang 的非 gfx950 分支（按 304 CU
调优），这条路径的失效模式是"周围文字依然流畅、但深处的事实取不回来"，普通 smoke test
抓不到。只有远超 2048 token 的检索类任务才能。

检查项
------
  weights        Tier 0，离线、不占 GPU：校验 indexer FP8 scale 的反量化自洽性
  basic          Tier 1：短问答（低于 index_topk，仅证明没有严重崩坏）
  determinism    Tier 1：greedy 解码必须可复现
  idle           Tier 1：空闲后首个请求是否输出损坏（回归检查）
  needle         Tier 2：合成散文的长上下文检索
  humaneval      代码 A：HumanEval 子集，短上下文，执行官方测试得 pass@1
  humaneval-long 代码 B：同一批题 + 8k 真实源码填充，与 A 的差值隔离 sparse 的影响
  code-retrieval 代码 C：跨文件符号检索（ground truth 由 ast 自动抽取）
  deep-api       代码 D：context 深处的自造 API，要求生成代码正确调用

用法
----
  ./verify_correctness.py --checks weights            # 离线，不占 GPU
  ./verify_correctness.py                             # 全部
  ./verify_correctness.py --checks humaneval,humaneval-long --n-humaneval 30
  ./verify_correctness.py --quick                      # 最小集，约 2 分钟
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

MODEL_UNSET = "__auto__"
DEFAULT_INDEX_TOPK = 2048


# ── 结果记录 ──────────────────────────────────────────────────────────────────
class Result:
    def __init__(self, name: str, ok: bool, got: str = "", want: str = "", **extra):
        self.name, self.ok, self.got, self.want, self.extra = name, ok, got, want, extra

    def line(self) -> str:
        tag = " ok " if self.ok else "FAIL"
        s = f"    [{tag}] {self.name}"
        if self.extra:
            s += "  " + "  ".join(f"{k}={v}" for k, v in self.extra.items())
        if not self.ok and self.got:
            s += f"\n           got={self.got!r} want={self.want!r}"
        return s


# ── server 客户端 ─────────────────────────────────────────────────────────────
class Server:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.model = model if model != MODEL_UNSET else self._discover_model()

    def _discover_model(self) -> str:
        r = self.session.get(f"{self.base_url}/v1/models", timeout=30)
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def chat(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        thinking: bool = False,
    ) -> dict:
        """一次 greedy 对话。默认关掉 thinking，这样答案简短、判定无歧义；
        推理链本身不是这里的验证对象。"""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        t0 = time.time()
        r = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout
        )
        r.raise_for_status()
        d = r.json()
        msg = d["choices"][0]["message"]
        return {
            "content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "finish_reason": d["choices"][0]["finish_reason"],
            "prompt_tokens": d["usage"]["prompt_tokens"],
            "completion_tokens": d["usage"]["completion_tokens"],
            "elapsed_s": time.time() - t0,
        }


# ── Tier 0：离线权重检查 ──────────────────────────────────────────────────────
def check_weights(model_dir: str) -> list[Result]:
    """校验 indexer FP8 scale 的反量化自洽性（checkpoint 级，与推理框架无关）。

    indexer 决定 sparse attention 能看到哪些 token，它的反量化一旦分组错了，症状是
    静默的检索质量下降而不是崩溃。这里确认两点：scale 全为 2 维（即统一走 block 反量化，
    不存在需要 per-channel 特殊处理的张量），且由 weight/scale 形状反推出的 block 尺寸
    与 config 声明的一致。

    只读 safetensors 的 header 取 shape，不加载权重数据。
    """
    from safetensors import safe_open

    out: list[Result] = []
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or {}
    declared_block = tuple(qcfg.get("weight_block_size") or ())

    index_path = Path(model_dir) / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    scales = {k: v for k, v in weight_map.items() if k.endswith(".weight_scale_inv") and ".indexer." in k}

    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for name, shard in scales.items():
        by_shard[shard].append(name)

    shape_counts: collections.Counter = collections.Counter()
    ndim1: list[str] = []
    block_mismatch: list[str] = []
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(Path(model_dir) / shard), "pt") as f:
            for n in names:
                scale_shape = tuple(f.get_slice(n).get_shape())
                w_shape = tuple(f.get_slice(n.replace(".weight_scale_inv", ".weight")).get_shape())
                kind = n.split(".indexer.")[1].rsplit(".weight_scale_inv", 1)[0]
                shape_counts[(kind, w_shape, scale_shape)] += 1
                if len(scale_shape) == 1:
                    ndim1.append(n)
                elif declared_block:
                    # 2 维 scale 走原有分支：block_size = w.shape[1] // scale.shape[1]，
                    # 该值必须与 config 里声明的 block 尺寸一致，否则反量化的分组是错的。
                    derived = w_shape[1] // scale_shape[1]
                    rows = -(-w_shape[0] // declared_block[0])
                    if derived != declared_block[1] or rows != scale_shape[0]:
                        block_mismatch.append(f"{n}: w={w_shape} scale={scale_shape} derived_block={derived}")

    for (kind, w_shape, s_shape), cnt in sorted(shape_counts.items()):
        print(f"      {kind:6s} weight={str(w_shape):14s} scale={str(s_shape):10s} "
              f"ndim={len(s_shape)}  x{cnt} layers")

    out.append(
        Result(
            "indexer scale 张量全部为 2 维（统一走 block 反量化）",
            ok=not ndim1,
            got=f"{len(ndim1)} 个 1 维 scale",
            want="0 个 1 维 scale",
            tensors=len(scales),
        )
    )
    out.append(
        Result(
            f"2 维 scale 推导出的 block 尺寸与 config 声明的 {declared_block} 一致",
            ok=not block_mismatch,
            got="; ".join(block_mismatch[:3]),
            want="全部一致",
        )
    )
    return out


# ── Tier 1 ───────────────────────────────────────────────────────────────────
def check_basic(srv: Server) -> list[Result]:
    """短、无歧义的任务。全部低于 index_topk，所以只能证明模型没有严重崩坏，
    对 sparse 检索路径不构成任何证据。"""
    cases = [
        ("What is 17 * 23? Reply with just the number.", ["391"]),
        ("What is 1234 + 5678? Reply with just the number.", ["6912"]),
        ("What is the capital of France? Reply with one word.", ["paris"]),
        ("In which year did Apollo 11 land humans on the Moon? Reply with just the year.", ["1969"]),
        ("Reply with exactly the word BANANA and nothing else.", ["banana"]),
        ("What is the sum of the squares of the integers 1 through 10? Reply with just the number.", ["385"]),
        ("Sort ascending, reply comma-separated only: 8, 3, 15, 1, 9", ["1, 3, 8, 9, 15", "1,3,8,9,15"]),
    ]
    out = []
    for prompt, expected in cases:
        try:
            got = srv.chat(prompt, max_tokens=64)["content"]
            ok = any(e.lower() in got.lower() for e in expected)
        except Exception as e:  # noqa: BLE001
            got, ok = f"EXCEPTION {e}", False
        out.append(Result(prompt[:48], ok, got[:70], expected[0]))
    return out


def check_idle(srv: Server, trials: int, idle_s: float) -> list[Result]:
    """空闲后首个请求是否输出损坏——这是 vLLM 那套上曾实测到的缺陷，在 SGLang 上作
    回归检查（此处尚未观察到，预期三项全过）。

    当时的行为：引擎运行队列排空后，下一个请求返回 token 级乱码（服务端日志无任何
    报错），紧随其后的同样请求则完全正常。这个失效模式值得一直测下去，因为：

      - 背靠背压测永远发现不了它，benchmark 会 0 失败通过；
      - 但 code agent 这类请求之间天然有思考间隔的场景会大面积中招。

    这里同时测三种条件，把"空闲触发"和"随机低概率损坏"区分开。
    """
    probe = "What is 17 * 23? Reply with just the number."
    sane = lambda c: "391" in c  # noqa: E731

    # 背靠背基线：先跑一个请求让引擎变热，再连续发。
    srv.chat(probe, max_tokens=32)
    hot = [sane(srv.chat(probe, max_tokens=32)["content"]) for _ in range(trials)]

    cold = []
    for _ in range(trials):
        time.sleep(idle_s)
        cold.append(sane(srv.chat(probe, max_tokens=32)["content"]))

    warmed = []
    for _ in range(trials):
        time.sleep(idle_s)
        srv.chat("hi", max_tokens=8)  # 丢弃的预热请求
        warmed.append(sane(srv.chat(probe, max_tokens=32)["content"]))

    return [
        Result("背靠背请求输出正常（基线）", all(hot), f"{sum(hot)}/{trials} 正常", f"{trials}/{trials}"),
        Result(f"空闲 {idle_s:g}s 后首个请求输出正常", all(cold),
               f"{sum(cold)}/{trials} 正常", f"{trials}/{trials}"),
        Result(f"空闲 {idle_s:g}s + 预热请求后输出正常（规避方案）", all(warmed),
               f"{sum(warmed)}/{trials} 正常", f"{trials}/{trials}"),
    ]


def check_determinism(srv: Server, repeats: int) -> list[Result]:
    """greedy 解码必须可复现。这里出现分歧会指向 attention metadata 未初始化或推导错误，
    而不是采样问题。"""
    prompt = "List the first 15 prime numbers, comma-separated, and nothing else."
    outs = [srv.chat(prompt, max_tokens=128)["content"] for _ in range(repeats)]
    return [
        Result(
            f"greedy 复现性（{repeats} 次）",
            ok=len(set(outs)) == 1,
            got=f"{len(set(outs))} 种不同输出",
            want="1 种",
        )
    ]


# ── Tier 2：合成散文 needle ───────────────────────────────────────────────────
# 刻意选用平淡、且完全不含数字的散文，这样整段 prompt 里唯一的数字就是 needle，
# 取回数字即为"确实读到了 needle"的无歧义证据。
_FILLER = [
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


def check_needle(srv: Server, lengths: list[int], depths: list[int], index_topk: int) -> list[Result]:
    """长上下文检索。这是真正压到补丁 sparse 路径的检查：超过 index_topk 之后，
    indexer 必须从整个 prompt 里挑出 needle 所在的 token；indexer 若不对，症状就是
    周围文字依然流畅、但数字取不回来。"""
    out = []
    for length in lengths:
        for depth in depths:
            seed = length * 1000 + depth
            rng = random.Random(seed)
            code = rng.randint(1_000_000, 9_999_999)
            needle = f"Important: the secret access code for Ravenna is {code}. Remember this number."
            n_sent = max(4, int(length / 16))  # 本语料约 16 token/句
            body = [rng.choice(_FILLER) for _ in range(n_sent)]
            body.insert(min(int(len(body) * depth / 100), len(body)), needle)
            prompt = " ".join(body) + (
                "\n\nBased on the text above, what is the secret access code for Ravenna? "
                "Reply with just the number."
            )
            try:
                r = srv.chat(prompt, max_tokens=64)
                got, ptok = r["content"], r["prompt_tokens"]
                ok = str(code) in got.replace(",", "").replace(" ", "")
                regime = "sparse" if ptok > index_topk else "dense-eq"
            except Exception as e:  # noqa: BLE001
                got, ok, ptok, regime = f"EXCEPTION {e}", False, 0, "?"
            out.append(
                Result(f"needle ~{length // 1024}k tok, 深度 {depth}%", ok, got[:50], str(code),
                       prompt_tok=ptok, regime=regime)
            )
    return out


# ── 代码场景：真实源码语料 ────────────────────────────────────────────────────
def _corpus_files(min_bytes: int = 3000, max_bytes: int = 40000) -> list[Path]:
    """拿本地 SGLang 源码作真实 Python 语料。比合成代码更贴近 code agent 的真实输入。

    原版用的是 vLLM 源码，但本容器只装了 SGLang。语料只作干扰/检索用，换成哪个大型
    Python 仓库都不影响结论。
    """
    try:
        import sglang

        root = Path(sglang.__file__).parent
    except Exception:  # noqa: BLE001
        return []
    files = []
    for p in sorted(root.rglob("*.py")):
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if min_bytes <= size <= max_bytes:
            files.append(p)
    return files


def _build_padding(files: list[Path], target_tokens: int, seed: int, tokenizer) -> str:
    """拼接真实源码到接近 target_tokens。纯干扰用，与任务无关。

    最后一个文件按行裁剪，否则整文件追加会把长度冲高到目标的 1.5 倍以上，各题的
    prompt 长度也就参差不齐，A/B 对比的"唯一变量是上下文长度"就不成立了。
    """
    rng = random.Random(seed)
    pool = files[:]
    rng.shuffle(pool)
    chunks, total = [], 0
    for p in pool:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        n = len(tokenizer.encode(text, add_special_tokens=False))
        if total + n > target_tokens:
            remaining = target_tokens - total
            lines = text.splitlines()
            # 按 token/行 的平均密度估算可保留的行数，宁少勿多。
            keep = max(1, int(len(lines) * remaining / max(n, 1)))
            text = "\n".join(lines[:keep])
            chunks.append(f"# ===== file: {p.name} =====\n{text}")
            break
        chunks.append(f"# ===== file: {p.name} =====\n{text}")
        total += n
    return "\n\n".join(chunks)


_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.findall(text)
    if m:
        return max(m, key=len).strip()
    return text.strip()


def _prompt_preamble(problem: dict) -> str:
    """取原始 prompt 中目标函数定义**之前**的部分。

    部分题目（如 HumanEval/10、/38）会先定义辅助函数再定义目标函数。对话模型通常只
    返回目标函数，于是辅助函数缺失、测试报 NameError——这是判分脚本的缺陷而非模型
    出错。这里补回 import 和辅助函数；不能直接拼整个 prompt，因为它末尾那个只有
    docstring、没有函数体的 def 会导致 SyntaxError。
    """
    prompt, ep = problem["prompt"], problem["entry_point"]
    matches = list(re.finditer(rf"^def\s+{re.escape(ep)}\s*\(", prompt, re.M))
    return prompt[: matches[-1].start()] if matches else ""


def _run_humaneval_case(code: str, problem: dict, timeout: float = 20.0) -> tuple[bool, str]:
    """在子进程里执行生成的代码 + 官方测试。

    注意：这会执行模型生成的代码。仅在受信任的基准场景下这样做，且加了超时和临时
    工作目录；不要拿它跑不可信输入。
    """
    program = (
        "from typing import List, Tuple, Dict, Set, Optional, Any\n"
        "import math, re, itertools, collections, string, heapq\n\n"
        f"{_prompt_preamble(problem)}\n\n"
        f"{code}\n\n{problem['test']}\ncheck({problem['entry_point']})\n"
    )
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "case.py"
        f.write_text(program)
        try:
            r = subprocess.run(
                [sys.executable, str(f)], cwd=td, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
    if r.returncode == 0:
        return True, ""
    tail = (r.stderr or "").strip().splitlines()
    return False, (tail[-1] if tail else "non-zero exit")[:110]


_HUMANEVAL_INSTR = (
    "Complete the following Python function. Reply with the complete function "
    "definition inside a single ```python code block, and nothing else.\n\n"
)


def check_humaneval(
    srv: Server, problems: list[dict], pad_tokens: int, tokenizer, thinking: bool
) -> list[Result]:
    """代码 A / B：HumanEval 子集，可执行判分。

    pad_tokens>0 时在题目前塞入无关的真实源码，把 prompt 推到 sparse 区间。A 与 B 用
    的是**同一批题**，因此两者 pass@1 的差值直接隔离出长上下文/sparse attention 的
    影响，排除了"模型本来就做不对这题"这个混淆因素。
    """
    out = []
    files = _corpus_files() if pad_tokens else []
    for i, prob in enumerate(problems):
        task = _HUMANEVAL_INSTR + prob["prompt"]
        if pad_tokens:
            pad = _build_padding(files, pad_tokens, seed=1000 + i, tokenizer=tokenizer)
            task = (
                "以下是本仓库的一些源码片段，仅作背景参考，与下面的任务无关：\n\n"
                f"{pad}\n\n===== 背景结束 =====\n\n" + task
            )
        try:
            r = srv.chat(task, max_tokens=1024, thinking=thinking)
            code = _extract_code(r["content"])
            ok, err = _run_humaneval_case(code, prob)
            ptok = r["prompt_tokens"]
        except Exception as e:  # noqa: BLE001
            ok, err, ptok = False, f"EXCEPTION {e}", 0
        out.append(Result(prob["task_id"], ok, err, "tests pass", prompt_tok=ptok))
    return out


def _pick_retrieval_target(
    files: list[Path], seed: int, budget_tokens: int, tokenizer
) -> tuple[list[Path], dict] | None:
    """用 ast 从真实源码里自动挑一个检索目标，ground truth 由解析得出而非手写。

    目标必须满足：函数名在所选文件集合里唯一，且至少有一个带默认值的关键字参数。
    文件按 token 预算而非个数选取，否则几个大文件就能把 context 推到 60k+ token，
    既拖慢验证又偏离我们要复现的 8k 场景。
    """
    rng = random.Random(seed)
    pool = files[:]
    rng.shuffle(pool)

    chosen, total = [], 0
    for p in pool:
        try:
            n = len(tokenizer.encode(p.read_text(errors="ignore"), add_special_tokens=False))
        except OSError:
            continue
        chosen.append(p)
        total += n
        if total >= budget_tokens:
            break

    defs: dict[str, list[tuple[Path, ast.FunctionDef]]] = collections.defaultdict(list)
    for p in chosen:
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                defs[node.name].append((p, node))

    candidates = []
    for name, hits in defs.items():
        if len(hits) != 1:
            continue  # 名字不唯一 -> 问题有歧义，跳过
        path, node = hits[0]
        args = node.args
        n_defaults = len(args.defaults)
        if n_defaults == 0:
            continue
        pos = args.args[-n_defaults:]
        for arg, default in zip(pos, args.defaults):
            try:
                val = ast.literal_eval(default)
            except (ValueError, SyntaxError):
                continue
            if isinstance(val, (bool, int, float, str)) or val is None:
                candidates.append({"func": name, "file": path.name, "param": arg.arg, "value": val})
    if not candidates:
        return None
    return chosen, rng.choice(candidates)


def check_code_retrieval(srv: Server, seed: int, budget_tokens: int, tokenizer) -> list[Result]:
    """代码 C：跨文件符号检索——code agent 最高频的操作。context 里放多个真实源文件，
    问某函数定义在哪个文件、以及它某个参数的默认值。ground truth 由 ast 自动抽取，
    避免我手写答案时出错。"""
    files = _corpus_files()
    picked = _pick_retrieval_target(files, seed, budget_tokens, tokenizer)
    if picked is None:
        return [Result("code-retrieval 目标选取", False, "语料中找不到合适目标", "至少 1 个")]
    chosen, target = picked

    context = "\n\n".join(
        f"# ===== file: {p.name} =====\n{p.read_text(errors='ignore')}" for p in chosen
    )
    out = []

    q1 = (
        f"{context}\n\n===== 以上是若干源文件 =====\n\n"
        f"函数 `{target['func']}` 定义在上面哪个文件里？只回答文件名。"
    )
    try:
        r = srv.chat(q1, max_tokens=64)
        ok = target["file"].lower() in r["content"].lower()
        out.append(Result(f"定位 {target['func']} 所在文件", ok, r["content"][:60],
                          target["file"], prompt_tok=r["prompt_tokens"]))
    except Exception as e:  # noqa: BLE001
        out.append(Result(f"定位 {target['func']} 所在文件", False, f"EXCEPTION {e}", target["file"]))

    want = str(target["value"])
    q2 = (
        f"{context}\n\n===== 以上是若干源文件 =====\n\n"
        f"在函数 `{target['func']}` 中，参数 `{target['param']}` 的默认值是什么？"
        f"只回答这个值本身。"
    )
    try:
        r = srv.chat(q2, max_tokens=64)
        got = r["content"]
        ok = want.lower().strip("'\"") in got.lower().replace('"', "").replace("'", "")
        out.append(Result(f"读出 {target['func']}({target['param']}=...) 默认值", ok,
                          got[:60], want, prompt_tok=r["prompt_tokens"]))
    except Exception as e:  # noqa: BLE001
        out.append(Result(f"读出 {target['func']}({target['param']}=...) 默认值", False,
                          f"EXCEPTION {e}", want))
    return out


# 自造 API：名字刻意生僻，确保模型不可能从预训练里"猜"出签名，只能真的从 context 读。
_FAKE_API = '''
def zt_compute_payload_checksum(payload: bytes, *, salt: str, rounds: int = 3) -> str:
    """Return the ZT checksum of ``payload``.

    Args:
        payload: raw bytes to digest.
        salt: mandatory salt string, must be passed as a keyword argument.
        rounds: number of mixing rounds; defaults to 3.
    """
    acc = salt.encode()
    for _ in range(rounds):
        acc = bytes((b * 31 + 7) & 0xFF for b in (acc + payload))
    return acc.hex()[:32]
'''


def check_deep_api(srv: Server, depth_pct: int, pad_tokens: int, tokenizer) -> list[Result]:
    """代码 D：把一个自造 API 埋进大段真实源码的深处，要求模型写代码正确调用它。

    因为 API 是虚构的，模型无法靠先验知识蒙对签名——调用正确就证明它真的从 context
    深处把定义读出来了。判定用 ast 检查调用节点，而不是脆弱的字符串匹配。
    """
    files = _corpus_files()
    pad = _build_padding(files, pad_tokens, seed=7, tokenizer=tokenizer)
    parts = pad.split("\n\n")
    insert_at = min(int(len(parts) * depth_pct / 100), len(parts))
    parts.insert(insert_at, f"# ===== file: zt_util.py =====\n{_FAKE_API}")
    context = "\n\n".join(parts)

    prompt = (
        f"{context}\n\n===== 以上是本仓库的源码 =====\n\n"
        "请写一个函数 `make_tag(data: bytes) -> str`，它使用上面源码中定义的 ZT 校验和函数，"
        "以 salt=\"zt-prod\"、rounds=7 计算 `data` 的校验和并返回结果。"
        "只输出一个 ```python 代码块。"
    )
    try:
        r = srv.chat(prompt, max_tokens=512)
        code = _extract_code(r["content"])
        ptok = r["prompt_tokens"]
    except Exception as e:  # noqa: BLE001
        return [Result("deep-api 调用生成", False, f"EXCEPTION {e}", "正确调用")]

    # 用 ast 核对：函数名、salt/rounds 以关键字传入、rounds 取值为 7。
    called, kw_ok, rounds_ok = False, False, False
    try:
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if fname != "zt_compute_payload_checksum":
                continue
            called = True
            kws = {k.arg: k.value for k in node.keywords if k.arg}
            kw_ok = "salt" in kws and "rounds" in kws
            rv = kws.get("rounds")
            rounds_ok = isinstance(rv, ast.Constant) and rv.value == 7
    except SyntaxError as e:
        return [Result("deep-api 调用生成", False, f"生成代码语法错误: {e}", "可解析", prompt_tok=ptok)]

    return [
        Result(f"deep-api：调用了深埋于 {depth_pct}% 处的自造函数", called,
               code[:80].replace("\n", " "), "zt_compute_payload_checksum(...)", prompt_tok=ptok),
        Result("deep-api：salt / rounds 以关键字参数传入", kw_ok, "", "salt= 且 rounds="),
        Result("deep-api：rounds=7", rounds_ok, "", "rounds=7"),
    ]


# ── 主流程 ────────────────────────────────────────────────────────────────────
ALL_CHECKS = [
    "weights", "basic", "determinism", "idle", "needle",
    "humaneval", "humaneval-long", "code-retrieval", "deep-api",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default=MODEL_UNSET)
    ap.add_argument("--checks", default=",".join(ALL_CHECKS))
    ap.add_argument("--lengths", default="4096,16384,65536", help="needle 上下文长度")
    ap.add_argument("--depths", default="10,50,90", help="needle 深度百分比")
    ap.add_argument("--repeats", type=int, default=3, help="determinism 重复次数")
    ap.add_argument("--idle-trials", type=int, default=3, help="idle 检查每种条件的试验次数")
    ap.add_argument("--idle-seconds", type=float, default=10.0, help="idle 检查的空闲秒数")
    ap.add_argument("--n-humaneval", type=int, default=20, help="HumanEval 取题数")
    ap.add_argument("--pad-tokens", type=int, default=8192, help="代码用例的填充 token 数")
    ap.add_argument("--context-tokens", type=int, default=8192,
                    help="code-retrieval 的 context token 预算")
    ap.add_argument("--thinking", action="store_true", help="HumanEval 打开思维链")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.quick:
        args.lengths, args.depths, args.n_humaneval = "4096", "50", 5

    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    unknown = [c for c in checks if c not in ALL_CHECKS]
    if unknown:
        print(f"未知检查项: {unknown}\n可用: {ALL_CHECKS}", file=sys.stderr)
        return 2

    needs_server = any(c != "weights" for c in checks)
    srv = None
    model_dir = args.model if args.model != MODEL_UNSET else None
    if needs_server:
        srv = Server(args.base_url, args.model, args.timeout)
        model_dir = srv.model
    elif model_dir is None:
        model_dir = "/wekafs/models/GLM-5.2-FP8"

    index_topk = DEFAULT_INDEX_TOPK
    try:
        index_topk = json.loads((Path(model_dir) / "config.json").read_text()).get(
            "index_topk", DEFAULT_INDEX_TOPK
        )
    except Exception:  # noqa: BLE001
        pass

    print(f"model      : {model_dir}")
    if srv:
        print(f"server     : {args.base_url}")
    print(f"index_topk : {index_topk}（prompt 超过此长度才进入 sparse 区间）")
    print(f"checks     : {', '.join(checks)}")
    print()

    tokenizer = None
    if any(c in checks for c in ("humaneval", "humaneval-long", "code-retrieval", "deep-api")):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 语料缺失会让 humaneval-long / code-retrieval / deep-api 静默退化成短上下文测试
    # 并"通过"，那比直接失败更危险，所以在这里就报错退出。
    if any(c in checks for c in ("humaneval-long", "code-retrieval", "deep-api")):
        if not _corpus_files():
            print("错误：找不到 Python 源码语料（需要能 import sglang），"
                  "长上下文代码检查会退化成短上下文测试。", file=sys.stderr)
            return 2

    problems: list[dict] = []
    if "humaneval" in checks or "humaneval-long" in checks:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test")
        problems = [ds[i] for i in range(min(args.n_humaneval, len(ds)))]

    groups: dict[str, list[Result]] = {}

    def run(name: str, title: str, fn):
        if name not in checks:
            return
        print(f"[{name}] {title}")
        t0 = time.time()
        groups[name] = fn()
        for r in groups[name]:
            print(r.line())
        print(f"      -- {time.time() - t0:.1f}s\n")

    run("weights", "Tier 0 · 离线校验 indexer 反量化自洽性（不占 GPU）",
        lambda: check_weights(model_dir))
    run("basic", "Tier 1 · 短问答（低于 index_topk，仅排除严重崩坏）",
        lambda: check_basic(srv))
    run("determinism", "Tier 1 · greedy 解码复现性",
        lambda: check_determinism(srv, args.repeats))
    run("idle", f"Tier 1 · 空闲后首请求损坏（回归检查，空闲 {args.idle_seconds:g}s）",
        lambda: check_idle(srv, args.idle_trials, args.idle_seconds))
    run("needle", "Tier 2 · 长上下文检索（压 sparse 路径）",
        lambda: check_needle(srv, [int(x) for x in args.lengths.split(",") if x],
                             [int(x) for x in args.depths.split(",") if x], index_topk))
    run("humaneval", f"代码 A · HumanEval x{len(problems)}，短上下文，执行判分",
        lambda: check_humaneval(srv, problems, 0, tokenizer, args.thinking))
    run("humaneval-long", f"代码 B · 同一批题 + {args.pad_tokens} token 真实源码填充",
        lambda: check_humaneval(srv, problems, args.pad_tokens, tokenizer, args.thinking))
    run("code-retrieval", "代码 C · 跨文件符号检索（ground truth 由 ast 抽取）",
        lambda: check_code_retrieval(srv, 42, args.context_tokens, tokenizer))
    run("deep-api", "代码 D · 深埋自造 API，要求生成代码正确调用",
        lambda: check_deep_api(srv, 50, args.pad_tokens, tokenizer))

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    overall = True
    for name, results in groups.items():
        passed = sum(1 for r in results if r.ok)
        total = len(results)
        # 代码生成类用 pass 率呈现：单题失败可能只是模型能力，不必然是 bug。
        soft = name in ("humaneval", "humaneval-long")
        status = "PASS" if passed == total else ("INFO" if soft else "FAIL")
        overall &= soft or passed == total
        rate = f"{100.0 * passed / total:.0f}%" if total else "n/a"
        print(f"{status:>4}  {name:<15} {passed}/{total}  ({rate})")

    a, b = groups.get("humaneval"), groups.get("humaneval-long")
    if a and b:
        pa = sum(1 for r in a if r.ok) / len(a)
        pb = sum(1 for r in b if r.ok) / len(b)
        ptok = max((r.extra.get("prompt_tok", 0) for r in b), default=0)
        print()
        print(f"代码 A/B 对比（同一批 {len(a)} 题）:")
        print(f"  短上下文 pass@1        : {pa:.0%}")
        print(f"  {ptok} token 上下文 pass@1 : {pb:.0%}")
        delta = pb - pa
        print(f"  差值                   : {delta:+.0%}")
        if delta <= -0.20:
            print("  => 长上下文下明显退化，需要排查 sparse attention 路径")
        elif delta <= -0.10:
            print("  => 有一定退化，建议加大题量确认是否为统计噪声")
        else:
            print("  => 未见长上下文退化")

    needle = groups.get("needle") or []
    if needle and not any(r.extra.get("regime") == "sparse" for r in needle):
        print("\n警告：所有 needle prompt 都没超过 index_topk，sparse 路径实际未被触及。")

    print("=" * 70)
    print("结论:", "全部通过" if overall else "存在失败项")

    if args.json_out:
        payload = {
            name: [
                {"name": r.name, "ok": r.ok, "got": r.got, "want": r.want, **r.extra}
                for r in results
            ]
            for name, results in groups.items()
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print("json:", args.json_out)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
