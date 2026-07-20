###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Thin OpenAI-compatible HTTP helpers used by e2e scenarios.

Engine-agnostic: everything here talks to the infera server's public
HTTP surface only, so the same helpers back the sglang, vllm and atom
scenarios unchanged.
"""

from __future__ import annotations

import httpx


async def chat(
    server_url: str,
    model: str,
    content: str,
    *,
    max_tokens: int = 20,
    temperature: float = 0.0,
    timeout: float = 180.0,
    chat_template_kwargs: dict | None = None,
) -> httpx.Response:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # e.g. {"enable_thinking": False} to stop Qwen3 from spending the token
    # budget on a <think> block. Ignored by templates that don't declare it.
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    async with httpx.AsyncClient(timeout=timeout) as c:
        return await c.post(f"{server_url}/v1/chat/completions", json=body)


async def chat_json(server_url: str, model: str, content: str, **kw) -> dict:
    r = await chat(server_url, model, content, **kw)
    assert r.status_code == 200, f"chat failed {r.status_code}: {r.text}"
    return r.json()


async def completion_text(
    server_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: float = 180.0,
) -> str:
    """Raw text-completion (/v1/completions). Unlike chat it has no chat
    template / thinking, so a "1,2,3,4,5," seed is simply continued — robust
    for tiny models. Returns choices[0].text."""
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(
            f"{server_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    assert r.status_code == 200, f"completion failed {r.status_code}: {r.text}"
    return r.json()["choices"][0].get("text") or ""


async def chat_stream_body(
    server_url: str,
    model: str,
    content: str,
    *,
    max_tokens: int = 20,
    timeout: float = 180.0,
) -> bytes:
    """POST a streaming chat completion and return the concatenated raw SSE bytes."""
    chunks: list[bytes] = []
    async with httpx.AsyncClient(timeout=timeout) as c:
        async with c.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200, f"stream failed {resp.status_code}"
            async for chunk in resp.aiter_raw():
                chunks.append(chunk)
    return b"".join(chunks)


async def get_workers(server_url: str, *, timeout: float = 10.0) -> list[dict]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{server_url}/v1/workers")
    assert r.status_code == 200, r.text
    return r.json()["workers"]
