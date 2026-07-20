///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Binary entry point. The router itself lives in the `infera_router` library
//! crate (see `lib.rs`); this just wires config → discovery → server.

use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwap;
use tracing_subscriber::EnvFilter;

use infera_router::block_hasher::BlockHasher;
use infera_router::config::Config;
use infera_router::handlers::{app, AppState};
use infera_router::kv_event::KvEventClient;
use infera_router::policy::{KvEventAwarePolicy, Policy, RoundRobin};
use infera_router::pool::Snapshot;
use infera_router::{discovery, proxy};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::parse_and_validate()?;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tracing::info!(?cfg, "starting infera-router (rust data plane)");

    // Build the routing policy from config. kv-aware owns a kv-event subscriber
    // + tokenizer; round-robin is stateless.
    let policy: Arc<dyn Policy> = if cfg.router_policy == "kv-aware" {
        let kv = Arc::new(KvEventClient::new());
        let hasher = match &cfg.kv_tokenizer_path {
            Some(p) => BlockHasher::load(p),
            None => BlockHasher::disabled(),
        };
        Arc::new(KvEventAwarePolicy::new(
            kv,
            hasher,
            cfg.kv_overlap_weight,
            cfg.kv_prefill_overlap_weight,
            cfg.kv_decode_overlap_weight,
        ))
    } else {
        Arc::new(RoundRobin::new())
    };

    // Lock-free pool: discovery swaps an immutable Snapshot; handlers load it
    // without locking, so reads scale across cores.
    let pool = Arc::new(ArcSwap::from_pointee(Snapshot::empty()));

    {
        let pool = pool.clone();
        let policy = policy.clone();
        let base = cfg.etcd_base();
        let prefix = cfg.etcd_prefix.clone();
        tokio::spawn(async move { discovery::run(base, prefix, pool, policy).await });
    }

    let state = AppState {
        pool,
        policy,
        http: proxy::build_upstream_client()?,
        started: Instant::now(),
        retries: cfg.request_max_retries,
    };

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("infera-router listening on http://{}", addr);

    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received; draining");
}
