///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Small response helpers.

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;

/// Truncate to at most `max` bytes without splitting a character.
///
/// What gets cut here is a worker's error text, which arrives as
/// `from_utf8_lossy` over whatever its engine said: a non-ASCII message or
/// model path is ordinary, and the replacement characters lossy conversion
/// inserts are themselves multi-byte. Slicing a `str` by byte index panics when
/// the index lands inside a character, and every one of these call sites is an
/// error path -- including a detached task, where the panic would take the
/// failure record with it and leave nothing behind.
pub fn truncate_chars(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let end = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max)
        .last()
        .unwrap_or(0);
    &s[..end]
}

/// Build a JSON `{"error": msg}` response with the given status.
pub fn json_error(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({ "error": msg }).to_string();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is always valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncating_never_splits_a_character() {
        // `&s[..200]` on this panics: the 200th byte is inside a character.
        // An engine reporting a failure in Chinese is enough to hit it, and the
        // panic lands on the path that was reporting the failure.
        let s = "模型加载失败".repeat(100);
        assert!(s.len() > 200);
        let cut = truncate_chars(&s, 200);
        assert!(cut.len() <= 200);
        assert!(s.starts_with(cut), "must be a prefix, not a re-encoding");
    }

    #[test]
    fn a_short_string_is_returned_whole() {
        assert_eq!(truncate_chars("short", 200), "short");
        // Exactly at the limit is not truncation.
        assert_eq!(truncate_chars("abcde", 5), "abcde");
    }

    #[test]
    fn a_single_oversized_character_truncates_to_empty() {
        // Nothing fits, and returning a partial character is not an option.
        assert_eq!(truncate_chars("模", 2), "");
    }

    #[test]
    fn ascii_truncates_exactly_at_the_limit() {
        assert_eq!(truncate_chars("abcdefghij", 4), "abcd");
    }
}
