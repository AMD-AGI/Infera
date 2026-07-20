///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Small response helpers.

use axum::body::Body;
use axum::http::{header, StatusCode};
use axum::response::Response;

/// Build a JSON `{"error": msg}` response with the given status.
pub fn json_error(status: StatusCode, msg: &str) -> Response {
    let body = serde_json::json!({ "error": msg }).to_string();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("static response is always valid")
}
