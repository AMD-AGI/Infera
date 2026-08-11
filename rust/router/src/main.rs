///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Binary entry point. The router itself lives in the `infera_router` library
//! crate (see `lib.rs`); this just wires config → discovery → server.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use tracing_subscriber::EnvFilter;

use infera_router::block_hasher::BlockHasher;
use infera_router::breaker;
use infera_router::config::Config;
use infera_router::handlers::{app, AppState};
use infera_router::kv_event::KvEventClient;
use infera_router::policy::{KvEventAwarePolicy, Policy, RoundRobin};
use infera_router::pool::Snapshot;
use infera_router::{discovery, discovery_k8s, k8s, nats_request, proxy};

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

    // Built before discovery starts: the reconcile loop prunes its entries
    // against the live fleet, the same way it reconciles the policy's.
    let breaker = Arc::new(breaker::CircuitBreaker::new(
        cfg.breaker_failure_threshold,
        Duration::from_secs_f64(cfg.breaker_cooldown_s),
        Duration::from_secs_f64(cfg.breaker_max_cooldown_s),
    ));

    {
        let pool = pool.clone();
        let policy = policy.clone();
        let breaker = breaker.clone();
        if cfg.discovery_backend == "kubernetes" {
            // validate() has already established the selector is present.
            let selector = cfg.k8s_label_selector.clone().unwrap_or_default();
            let ns = cfg
                .k8s_namespace
                .clone()
                .unwrap_or_else(k8s::in_cluster_namespace);
            tracing::info!("discovery: kubernetes, selector {selector:?} in ns {ns}");
            tokio::spawn(
                async move { discovery_k8s::run(selector, ns, pool, policy, breaker).await },
            );
        } else {
            let base = cfg.etcd_base();
            let prefix = cfg.etcd_prefix.clone();
            tracing::info!("discovery: etcd at {base}");
            tokio::spawn(async move { discovery::run(base, prefix, pool, policy, breaker).await });
        }
    }

    let nats = if cfg.request_transport == "nats" {
        Some(Arc::new(
            nats_request::NatsRequestClient::connect(
                cfg.nats_server.as_deref(),
                cfg.nats_req_idle_timeout_s,
                cfg.nats_req_max_duration_s,
                cfg.nats_req_max_pending,
            )
            .await?,
        ))
    } else {
        None
    };

    let state = AppState {
        pool,
        policy,
        http: proxy::build_upstream_client()?,
        started: Instant::now(),
        retries: cfg.request_max_retries,
        breaker,
        nats,
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
