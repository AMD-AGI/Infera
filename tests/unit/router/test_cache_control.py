###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/router/cache_control.py.

We parse two request shapes:
  - Anthropic-style `cache_control` blocks on messages / system / tools
  - OpenAI-style `prompt_cache_retention` / `prompt_cache_key`

Tests are organized by shape, then a mixed-shape section at the end.
"""

from __future__ import annotations

from infera.router.cache_control import (
    CacheHints,
    Retention,
    parse_cache_hints,
)

# ----------------------------------------------------------------------
# Helpers — request body fragments
# ----------------------------------------------------------------------


def _ephemeral(ttl: str | None = None) -> dict:
    out: dict = {"type": "ephemeral"}
    if ttl is not None:
        out["ttl"] = ttl
    return out


def _text_block(text: str, cache_control: dict | None = None) -> dict:
    out = {"type": "text", "text": text}
    if cache_control is not None:
        out["cache_control"] = cache_control
    return out


# ----------------------------------------------------------------------
# Anthropic shape — system blocks
# ----------------------------------------------------------------------


def test_anthropic_system_with_long_ttl():
    body = {
        "system": [_text_block("stable", _ephemeral(ttl="1h"))],
        "messages": [{"role": "user", "content": "hi"}],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG
    assert hints.explicit_hint_seen is True
    assert hints.session_id is None


def test_anthropic_system_with_default_ephemeral_is_short():
    body = {
        "system": [_text_block("stable", _ephemeral())],
        "messages": [{"role": "user", "content": "hi"}],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.SHORT
    assert hints.explicit_hint_seen is True


def test_anthropic_system_without_cache_control_is_none():
    body = {
        "system": [_text_block("stable")],
        "messages": [{"role": "user", "content": "hi"}],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE
    assert hints.explicit_hint_seen is False


# ----------------------------------------------------------------------
# Anthropic shape — last user message tail block
# ----------------------------------------------------------------------


def test_anthropic_user_message_last_block_ephemeral():
    """OpenClaw's chain-of-warmth: last user content block gets
    cache_control=ephemeral for the next-turn hit."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    _text_block("question"),
                    _text_block("tail", _ephemeral()),
                ],
            }
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.SHORT


# ----------------------------------------------------------------------
# Anthropic shape — tools
# ----------------------------------------------------------------------


def test_anthropic_tools_with_long_ttl():
    """Tool defs are usually marked long-retention (stable across the
    whole agent session)."""
    body = {
        "tools": [
            {
                "name": "search",
                "description": "search the web",
                "cache_control": _ephemeral(ttl="1h"),
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG


# ----------------------------------------------------------------------
# Anthropic shape — max retention across blocks
# ----------------------------------------------------------------------


def test_anthropic_max_retention_wins():
    """If system is long but last user-turn block is short, overall is LONG."""
    body = {
        "system": [_text_block("stable", _ephemeral(ttl="1h"))],
        "messages": [
            {
                "role": "user",
                "content": [_text_block("tail", _ephemeral())],
            }
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG


def test_anthropic_two_short_blocks_stay_short():
    body = {
        "system": [_text_block("a", _ephemeral())],
        "messages": [
            {
                "role": "user",
                "content": [_text_block("b", _ephemeral())],
            }
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.SHORT


# ----------------------------------------------------------------------
# Anthropic shape — thinking blocks NEVER cache
# ----------------------------------------------------------------------


def test_thinking_block_with_cache_control_still_returns_none():
    """OpenClaw strips cache_control from thinking blocks defensively.
    We do the same on the server side — even if a misbehaving client
    sends one, we treat it as NONE for that block."""
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "(reasoning)",
                        "cache_control": _ephemeral(ttl="1h"),  # buggy client
                    },
                ],
            }
        ],
    }
    hints = parse_cache_hints(body)
    # No OTHER block carries cache_control, so overall is NONE.
    assert hints.retention == Retention.NONE


def test_thinking_block_does_not_downgrade_other_blocks():
    """A thinking block is NONE, but if another block is LONG, overall is LONG."""
    body = {
        "system": [_text_block("stable", _ephemeral(ttl="1h"))],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "..."},  # no cache_control
                ],
            }
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG


# ----------------------------------------------------------------------
# OpenAI shape
# ----------------------------------------------------------------------


def test_openai_retention_24h_is_long():
    body = {
        "prompt_cache_key": "session-xyz",
        "prompt_cache_retention": "24h",
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG
    assert hints.session_id == "session-xyz"


def test_openai_retention_1h_is_long():
    body = {"prompt_cache_retention": "1h"}
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG


def test_openai_retention_5m_is_short():
    body = {"prompt_cache_retention": "5m"}
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.SHORT


def test_openai_cache_key_alone_is_short():
    """No retention specified, key present → implicit short."""
    body = {"prompt_cache_key": "session-abc"}
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.SHORT
    assert hints.session_id == "session-abc"


def test_openai_no_fields_is_none():
    body = {"model": "x", "messages": []}
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE
    assert hints.session_id is None
    assert hints.explicit_hint_seen is False


def test_openai_unknown_retention_value_ignored():
    """Unknown retention string shouldn't crash; treat as no hint."""
    body = {"prompt_cache_retention": "garbage"}
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE


# ----------------------------------------------------------------------
# Mixed shapes — proxy / gateway scenarios
# ----------------------------------------------------------------------


def test_mixed_anthropic_long_beats_openai_short():
    body = {
        "system": [_text_block("stable", _ephemeral(ttl="1h"))],
        "messages": [{"role": "user", "content": "hi"}],
        # A proxy slapped these on too
        "prompt_cache_key": "session-x",
        "prompt_cache_retention": "5m",
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG
    assert hints.session_id == "session-x"  # session still propagated


def test_mixed_openai_long_beats_anthropic_short():
    body = {
        "system": [_text_block("stable", _ephemeral())],
        "messages": [{"role": "user", "content": "hi"}],
        "prompt_cache_retention": "24h",
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG


# ----------------------------------------------------------------------
# Defensive — malformed bodies must not crash
# ----------------------------------------------------------------------


def test_non_dict_body_returns_default():
    assert parse_cache_hints(None).retention == Retention.NONE  # type: ignore[arg-type]
    assert parse_cache_hints("not a dict").retention == Retention.NONE  # type: ignore[arg-type]
    assert parse_cache_hints([]).retention == Retention.NONE  # type: ignore[arg-type]


def test_cache_control_with_wrong_type_field_ignored():
    body = {
        "system": [
            _text_block("x", {"type": "persistent", "ttl": "1h"}),  # unknown
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE


def test_messages_with_string_content_no_crash():
    """OpenAI shape allows content as a plain string."""
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE


def test_session_id_falls_back_to_session_id_field():
    body = {"session_id": "fallback-session-1"}
    hints = parse_cache_hints(body)
    assert hints.session_id == "fallback-session-1"


# ----------------------------------------------------------------------
# Retention enum semantics
# ----------------------------------------------------------------------


def test_retention_rank_order():
    assert Retention.NONE.rank() < Retention.SHORT.rank() < Retention.LONG.rank()


def test_cache_hints_dataclass_is_frozen():
    h = CacheHints(retention=Retention.LONG, session_id="x", explicit_hint_seen=True)
    import dataclasses

    try:
        dataclasses.replace(h, retention=Retention.SHORT)
    except dataclasses.FrozenInstanceError:
        raise AssertionError("CacheHints should support dataclasses.replace") from None
    # Direct mutation should fail (frozen):
    try:
        h.retention = Retention.SHORT  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("CacheHints should be frozen (immutable)")


# ----------------------------------------------------------------------
# Multimodal detection
#
# The router-side block hasher is text-only. Image placeholder tokens
# share the same id across different images, so a same-text-different-
# image request would silently collide with the cache entry for another
# image. Detection here is the upstream signal that lets the policy
# fall back to load-balance-only routing for MM requests.
# ----------------------------------------------------------------------


def test_no_mm_when_body_is_pure_text():
    body = {"messages": [{"role": "user", "content": "what's the weather"}]}
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is False


def test_no_mm_when_content_is_list_of_text_blocks_only():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is False


def test_mm_when_anthropic_image_block_present():
    """Anthropic vision: messages[].content[].type == 'image'."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgo...",
                        },
                    },
                ],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_when_openai_image_url_block_present():
    """OpenAI vision: messages[].content[].type == 'image_url'."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/foo.png"},
                    },
                ],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_when_openai_input_audio_block_present():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "<base64>"}},
                ],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_when_document_block_present():
    """Anthropic document support — also non-text, hash-unsafe."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize"},
                    {"type": "document", "source": {}},
                ],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_when_video_block_present():
    """Forward-compat: future vision models with video input."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": {"url": "..."}}],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_when_image_in_system_block():
    """Anthropic now allows images in system blocks too — also MM."""
    body = {
        "system": [
            {"type": "text", "text": "you are helpful"},
            {"type": "image", "source": {}},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is True


def test_mm_detection_independent_of_cache_control_hint():
    """MM detection runs regardless of whether cache_control is set —
    a vision request without a hint is still MM and still hash-unsafe."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.NONE
    assert hints.has_multimodal_content is True


def test_mm_detection_with_cache_control_keeps_retention_signal():
    """Retention + MM can coexist on the same request: client wants long
    caching, but the request has an image. Retention parsed normally,
    has_multimodal_content also True — policy decides what to do."""
    body = {
        "system": [_text_block("rules", _ephemeral(ttl="1h"))],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image", "source": {}},
                ],
            }
        ],
    }
    hints = parse_cache_hints(body)
    assert hints.retention == Retention.LONG
    assert hints.has_multimodal_content is True


def test_no_mm_on_malformed_body():
    """Defensive: malformed body shouldn't crash, shouldn't false-positive."""
    assert parse_cache_hints({}).has_multimodal_content is False
    assert parse_cache_hints({"messages": "not-a-list"}).has_multimodal_content is False
    assert parse_cache_hints({"messages": [{"content": 42}]}).has_multimodal_content is False
    assert (
        parse_cache_hints({"messages": [{"content": [{"type": 123}]}]}).has_multimodal_content
        is False
    )


def test_no_mm_on_unknown_block_type():
    """A future block type we don't know about should NOT trip MM
    detection — better to false-negative on an unknown type than
    false-positive on every harmless future addition."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "weird_new_type", "data": "..."}],
            }
        ]
    }
    hints = parse_cache_hints(body)
    assert hints.has_multimodal_content is False
