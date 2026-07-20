///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Infera router data plane (Rust): a multi-core drop-in for the Python
//! router's hot path — same etcd workers, same OpenAI API, no single-core
//! ceiling.
//!
//! Supports mixed dispatch + round-robin + kv-aware (DP-attention cache
//! locality) routing + etcd discovery + SSE relay + SGLang-bootstrap PD.
//! Configs outside this set (NATS transport, other PD connectors, edge
//! endpoints) are served by the Python backend.
//!
//! Modules are `pub` so the binary and the `tests/` suite share one API.

pub mod block_hasher;
pub mod cache_control;
pub mod config;
pub mod disagg;
pub mod discovery;
pub mod dp;
pub mod handlers;
pub mod hasher;
pub mod kv_event;
pub mod policy;
pub mod pool;
pub mod protocol;
pub mod proxy;
pub mod tiktoken;
pub mod util;
