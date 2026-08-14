###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Carrying a half-finished generation from one worker to another.

A worker that goes away mid-generation -- drained, evicted, crashed -- takes an
unfinished response with it. Retrying the request from the top is not equivalent:
the client has already been sent tokens, so a fresh generation would either
repeat them or contradict what it already read.

What moves instead is the generation so far, appended to the prompt with the
token budget reduced by what it cost, so the next worker continues rather than
restarts. The client sees one uninterrupted stream and never learns a worker
changed underneath it.

**Exactly, when the engine allows it.** Given the token ids the engine actually
sampled (see infera.router.token_ids), the continuation carries those ids and
the prompt's, and the next worker resumes from the identical sequence. Without
them the decoded text is carried instead and re-encoded downstream, which can
shift a token boundary: the same words, not provably the same tokens. Exactness
is preferred wherever it is available and the fallback is silent, because an
approximate continuation is still far better than a severed stream.

**What this is not.** The KV cache does not move, so the new worker re-reads the
carried prefix -- kv-aware routing usually lands on a worker that already holds
some of it, which is what keeps that affordable. And no continuation is
byte-identical to what the original worker would have produced: sampling state
does not survive the move, and exact ids do not change that. A caller who needs
reproducible output for a fixed seed should not enable migration.

**Requires the NATS transport.** Over HTTP the router hands the connection
straight to the engine and never sees a frame boundary, so there is nothing to
accumulate from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from infera.router.token_ids import deltas_from_chunk, prompt_from_chunk, strip_token_ids

logger = logging.getLogger(__name__)

# The sentinel that ends an OpenAI stream. Never carried: it belongs to the
# stream the client is reading, not to the generation being moved.
_DONE_SENTINEL = b"data: [DONE]"
_DATA_PREFIX = b"data: "

# Only completions accepts a pre-tokenized prompt; chat has no such entry, so an
# exact continuation of a chat request has to be issued against this path.
COMPLETIONS_PATH = "/v1/completions"

# Request fields whose meaning would not survive being reissued as completions.
# Tools are the obvious one -- a completions request cannot emit a tool call, so
# converting one that might need to would quietly remove the ability halfway
# through the answer.
_CHAT_ONLY_FIELDS = ("tools", "functions", "tool_choice", "function_call", "response_format")


def as_chat_chunk(raw: bytes) -> bytes:
    """Re-shape a completions chunk into the chat form the client is reading.

    An exact chat continuation has to be issued against the completions
    endpoint, which answers in a different shape: ``choices[].text`` where the
    client expects ``choices[].delta.content``. Without this the migration would
    be plainly visible as the response changing format mid-stream.

    Anything unrecognised is passed through untouched. A chunk this cannot
    convert is a chunk it does not understand, and mangling it would be worse
    than letting it through.
    """
    if not raw:
        return raw
    out: list[bytes] = []
    rewritten = False
    for line in raw.split(b"\n"):
        stripped = line.strip()
        if not stripped.startswith(_DATA_PREFIX) or stripped.startswith(_DONE_SENTINEL):
            out.append(line)
            continue
        try:
            obj = json.loads(stripped[len(_DATA_PREFIX) :])
        except ValueError:
            out.append(line)
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("choices"), list):
            out.append(line)
            continue
        obj["object"] = "chat.completion.chunk"
        for choice in obj["choices"]:
            if isinstance(choice, dict) and "text" in choice:
                choice["delta"] = {"content": choice.pop("text") or ""}
                # `logprobs` here describes completions tokens and has a
                # different shape in chat; dropping beats mistranslating.
                choice.pop("logprobs", None)
        out.append(_DATA_PREFIX + json.dumps(obj, separators=(",", ":")).encode())
        rewritten = True
    return b"\n".join(out) if rewritten else raw


@dataclass(frozen=True)
class Continuation:
    """The request that makes another worker finish this generation."""

    body: dict
    path: str
    exact: bool


class MigrationState:
    """What has been produced so far, and what it costs to carry it.

    Fed every chunk on its way to the client, so what is accumulated is what the
    client actually received -- not what a worker claims to have sent. That
    distinction is the point: after a migration the two must agree, or the
    client sees a seam.
    """

    def __init__(self, body: dict, *, limit: int, path: str = COMPLETIONS_PATH) -> None:
        self._original_body = body
        self._path = path
        self._text: list[str] = []
        self._chunks_with_text = 0
        self._output_ids: list[int] = []
        self._prompt_ids: list[int] | None = None
        # Cleared the moment a chunk carries text the engine did not account for
        # in ids. Continuing from ids that cover only part of what the client
        # read would drop the rest, so the whole id path is abandoned instead.
        self._ids_cover_output = True
        self.migrations_left = limit
        # Set once anything unparseable arrives. A generation that cannot be
        # reconstructed exactly must not be migrated at all: continuing from a
        # partial prefix would silently drop output the client already read.
        self.poisoned = False

    @property
    def produced_text(self) -> str:
        return "".join(self._text)

    @property
    def produced_tokens(self) -> int:
        """Tokens generated so far.

        Exact when the engine reported ids. Otherwise the number of chunks that
        carried text, which every engine here emits one token at a time. Only
        used to reduce the remaining budget, where being off by a little changes
        the length of the answer and nothing else.
        """
        if self._has_exact_output():
            return len(self._output_ids)
        return self._chunks_with_text

    def is_exact(self) -> bool:
        """Whether the continuation can be issued as token ids.

        Needs the prompt's ids as well as the output's: resuming from exact
        output ids appended to a re-encoded prompt would just move the ambiguity
        from one end of the sequence to the other.
        """
        return bool(self._prompt_ids) and self._has_exact_output()

    def _has_exact_output(self) -> bool:
        return self._ids_cover_output and bool(self._output_ids)

    def observe(self, chunk: bytes) -> bytes:
        """Record one chunk, and return what should go to the client.

        The ids are asked for by the router, not the caller, so they are taken
        out on the way past: whether migration is enabled must not change the
        shape of the response.
        """
        if self.poisoned or not chunk:
            return chunk
        out_lines: list[bytes] = []
        rewritten = False
        for line in chunk.split(b"\n"):
            stripped = line.strip()
            if not stripped.startswith(_DATA_PREFIX) or stripped.startswith(_DONE_SENTINEL):
                out_lines.append(line)
                continue
            try:
                obj = json.loads(stripped[len(_DATA_PREFIX) :])
            except ValueError:
                # Not JSON: this is not a stream we know how to reconstruct.
                self._poison("chunk is not JSON")
                return chunk
            self._record(obj)
            if strip_token_ids(obj):
                # Rebuild only the lines that carried ids; everything else keeps
                # the engine's own bytes.
                out_lines.append(_DATA_PREFIX + json.dumps(obj, separators=(",", ":")).encode())
                rewritten = True
            else:
                out_lines.append(line)
        return b"\n".join(out_lines) if rewritten else chunk

    def _record(self, obj: object) -> None:
        if self._prompt_ids is None:
            self._prompt_ids = prompt_from_chunk(obj)
        ids = deltas_from_chunk(obj)
        if not ids and self._carries_more_than_text(obj):
            # Output that does not survive being carried as text. Resuming would
            # replay it: the client holds half a tool call and would be sent a
            # whole one after it. Exact ids cover this -- they are the tokens
            # behind whatever the parser produced -- so this only stops the
            # text path.
            self._poison("output is not plain text (tool call or reasoning)")
            return
        if ids:
            self._output_ids.extend(ids)
        delta = self._delta_of(obj)
        if delta is None:
            return
        self._text.append(delta)
        self._chunks_with_text += 1
        if not ids:
            # Text the ids do not account for. Whatever the engine is doing --
            # not reporting ids, or reporting them only sometimes -- the id
            # sequence is no longer a faithful record of what was produced.
            self._ids_cover_output = False

    @staticmethod
    def _carries_more_than_text(obj: object) -> bool:
        """Whether a chunk holds output that concatenated text cannot represent.

        Tool calls arrive under their own key rather than as content, and a
        reasoning parser moves the model's thinking out of it. Either way the
        text the client received is not the whole of what was produced, so a
        continuation built from that text alone is missing part of the answer.
        """
        if not isinstance(obj, dict):
            return False
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return False
        first = choices[0]
        if not isinstance(first, dict):
            return False
        delta = first.get("delta")
        if not isinstance(delta, dict):
            return False
        return bool(delta.get("tool_calls") or delta.get("reasoning_content"))

    @staticmethod
    def _delta_of(obj: object) -> str | None:
        """The text a chunk adds, for chat and completions alike."""
        if not isinstance(obj, dict):
            return None
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        # Chat completions put it under `delta`, completions under `text`.
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            return content if isinstance(content, str) and content else None
        text = first.get("text")
        return text if isinstance(text, str) and text else None

    def _poison(self, why: str) -> None:
        self.poisoned = True
        logger.warning("migration disabled for this request: %s", why)

    def can_migrate(self) -> bool:
        return not self.poisoned and self.migrations_left > 0

    def next_continuation(self) -> Continuation:
        """The request that makes another worker continue this generation."""
        self.migrations_left -= 1
        if self.is_exact() and self._convertible():
            return self._exact_continuation()
        return self._text_continuation()

    def _convertible(self) -> bool:
        """Whether this request can be reissued against the completions path.

        A completions request cannot emit a tool call or honour a chat response
        format, so a request that might need either keeps the text path: a
        continuation that silently loses a capability is worse than one whose
        token boundaries might differ.
        """
        if self._path.endswith("/completions") and not self._path.endswith("/chat/completions"):
            return True
        return not any(self._original_body.get(f) for f in _CHAT_ONLY_FIELDS)

    def _exact_continuation(self) -> Continuation:
        body = self._base_body()
        body.pop("messages", None)
        body["prompt"] = list(self._prompt_ids or []) + self._output_ids
        return Continuation(body=body, path=COMPLETIONS_PATH, exact=True)

    def _text_continuation(self) -> Continuation:
        body = self._base_body()
        carried = self.produced_text
        if body.get("messages") is not None:
            # Chat: an assistant turn holding what has been said so far. Engines
            # continue such a turn rather than starting a new one, which is
            # exactly the semantics needed here.
            messages = list(body["messages"])
            messages.append({"role": "assistant", "content": carried})
            body["messages"] = messages
        else:
            body["prompt"] = f"{body.get('prompt', '')}{carried}"
        return Continuation(body=body, path=self._path, exact=False)

    def _base_body(self) -> dict:
        body = dict(self._original_body)
        # Asked for by the router for its own use; the next worker is told
        # separately whether they are still wanted.
        body.pop("return_token_ids", None)
        for key in ("max_tokens", "max_completion_tokens"):
            budget = body.get(key)
            if isinstance(budget, int):
                # Never below 1: a request for zero tokens is rejected outright,
                # which would turn a migration into an error the client sees.
                body[key] = max(1, budget - self.produced_tokens)
        return body
