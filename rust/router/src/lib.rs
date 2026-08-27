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
//! locality) routing, etcd and Kubernetes discovery, HTTP and NATS request
//! transports, ZMQ and NATS kv-event feeds, SSE relay, and SGLang-bootstrap PD
//! -- which together cover what the operator deploys by default. Configs
//! outside this set (other PD connectors, GAIE direct mode, the profiling
//! control plane) are served by the Python backend.
//!
//! Modules are `pub` so the binary and the `tests/` suite share one API.

pub mod block_hasher;
pub mod breaker;
pub mod cache_control;
pub mod config;
pub mod disagg;
pub mod discovery;
pub mod discovery_k8s;
pub mod dp;
pub mod encoding_dsv4;
pub mod encoding_k3;
pub mod handlers;
pub mod hasher;
pub mod k8s;
pub mod kv_event;
pub mod kv_event_nats;
pub mod kv_selfheal;
pub mod nats_request;
pub mod policy;
pub mod pool;
pub mod protocol;
pub mod proxy;
pub mod tiktoken;
pub mod util;
