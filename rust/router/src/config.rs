///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! CLI / env configuration. Flag names mirror `infera.server.args` so the
//! Python `--router-backend rust` shim can translate 1:1.

use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "infera-router", about = "Infera router data plane (Rust)")]
pub struct Config {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    #[arg(long, default_value = "127.0.0.1:2379")]
    pub etcd_endpoint: String,

    #[arg(long, default_value = "/infera/workers/")]
    pub etcd_prefix: String,

    /// Failover attempts to alternate workers on a pre-first-byte failure.
    #[arg(long, default_value_t = 1)]
    pub request_max_retries: usize,

    /// Consecutive pre-first-byte worker faults before a worker is taken out
    /// of rotation. Failover alone forgets between requests; this remembers.
    #[arg(long, default_value_t = 3)]
    pub breaker_failure_threshold: u32,

    /// Seconds a tripped worker is excluded before one probe is admitted.
    #[arg(long, default_value_t = 5.0)]
    pub breaker_cooldown_s: f64,

    /// Ceiling for the cooldown, which doubles on each failed probe.
    #[arg(long, default_value_t = 60.0)]
    pub breaker_max_cooldown_s: f64,

    /// `round-robin` or `kv-aware` (DP-attention cache-locality routing).
    #[arg(long, default_value = "round-robin")]
    pub router_policy: String,

    /// `etcd` (external) or `kubernetes` (workers publish into their own Pod
    /// annotation and the API server is watched).
    #[arg(long, default_value = "etcd")]
    pub discovery_backend: String,

    /// kubernetes discovery: label selector identifying the fleet's worker
    /// Pods. Required for that backend -- an empty selector would match every
    /// Pod in the namespace.
    #[arg(long, env = "INFERA_K8S_LABEL_SELECTOR")]
    pub k8s_label_selector: Option<String>,

    /// kubernetes discovery: namespace to watch. Defaults to the Pod's own.
    #[arg(long)]
    pub k8s_namespace: Option<String>,

    /// `http` (dial the worker directly) or `nats` (publish onto the worker's
    /// own subject and stream the reply back over an inbox).
    #[arg(long, default_value = "http")]
    pub request_transport: String,

    /// NATS broker URL. Falls back to `NATS_SERVER`, then to localhost.
    #[arg(long, env = "NATS_SERVER")]
    pub nats_server: Option<String>,

    /// Seconds to wait for the *next* reply chunk before giving up on a
    /// request. Reset on every chunk, so a long generation that keeps producing
    /// tokens never trips it -- only a stall does. 0 disables it.
    #[arg(long, default_value_t = 900.0, env = "INFERA_NATS_REQ_IDLE_TIMEOUT")]
    pub nats_req_idle_timeout_s: f64,

    /// Hard cap on a whole request's wall clock regardless of token flow, for
    /// runaway generations. 0 (the default) disables it.
    #[arg(long, default_value_t = 0.0, env = "INFERA_NATS_REQ_MAX_DURATION")]
    pub nats_req_max_duration_s: f64,

    /// Refuse to dispatch to a worker whose in-NATS backlog has reached this
    /// many messages, answering 429. Turning it on makes the request path
    /// JetStream-backed, which is what makes the backlog measurable. 0 (the
    /// default) keeps the transport pure core NATS.
    #[arg(long, default_value_t = 0, env = "INFERA_NATS_REQ_MAX_PENDING")]
    pub nats_req_max_pending: usize,

    /// kv-aware only: path to the model's HF fast tokenizer (`tokenizer.json` or
    /// its dir). Required for cache locality — without it kv-aware degrades to
    /// pure load balancing (block hashes can't be computed).
    #[arg(long)]
    pub kv_tokenizer_path: Option<String>,

    /// kv-aware: base overlap weight in `cost = w*(blocks-hits) + active`.
    #[arg(long, default_value_t = 1.0)]
    pub kv_overlap_weight: f64,

    /// kv-aware: overlap weight for prefill workers (compute-bound; weight cache
    /// locality aggressively). Defaults to `kv_overlap_weight`.
    #[arg(long)]
    pub kv_prefill_overlap_weight: Option<f64>,

    /// kv-aware: overlap weight for decode workers (memory-bound; route by load).
    /// Defaults to `kv_overlap_weight`.
    #[arg(long)]
    pub kv_decode_overlap_weight: Option<f64>,
}

impl Config {
    pub fn parse_and_validate() -> anyhow::Result<Self> {
        let c = Config::parse();
        c.validate()?;
        Ok(c)
    }

    /// Reject config outside the Rust backend's supported set.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.router_policy != "round-robin" && self.router_policy != "kv-aware" {
            anyhow::bail!(
                "rust backend supports --router-policy round-robin|kv-aware (got {:?})",
                self.router_policy
            );
        }
        if self.router_policy == "kv-aware" && self.kv_tokenizer_path.is_none() {
            tracing::warn!(
                "--router-policy kv-aware without --kv-tokenizer-path: block hashes \
                 can't be computed, so routing degrades to pure load balancing"
            );
        }
        match self.discovery_backend.as_str() {
            "etcd" => {}
            "kubernetes" => {
                // Without a selector this would watch every Pod in the
                // namespace and register anything carrying the annotation.
                if self.k8s_label_selector.as_deref().unwrap_or("").is_empty() {
                    anyhow::bail!("--discovery-backend kubernetes requires --k8s-label-selector");
                }
            }
            other => anyhow::bail!(
                "rust backend supports --discovery-backend etcd|kubernetes (got {other:?})"
            ),
        }
        if self.request_transport != "http" && self.request_transport != "nats" {
            anyhow::bail!(
                "rust backend supports --request-transport http|nats (got {:?})",
                self.request_transport
            );
        }
        Ok(())
    }

    /// Normalize the etcd endpoint to a base URL for the v3 HTTP/JSON gateway.
    pub fn etcd_base(&self) -> String {
        let ep = &self.etcd_endpoint;
        if ep.starts_with("http://") || ep.starts_with("https://") {
            ep.trim_end_matches('/').to_string()
        } else if ep.contains(':') {
            format!("http://{ep}")
        } else {
            format!("http://{ep}:2379")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_endpoint(ep: &str) -> Config {
        Config::try_parse_from(["infera-router", "--etcd-endpoint", ep]).unwrap()
    }

    #[test]
    fn etcd_base_normalizes_forms() {
        assert_eq!(
            with_endpoint("127.0.0.1:2379").etcd_base(),
            "http://127.0.0.1:2379"
        );
        // bare host gets the default etcd port
        assert_eq!(
            with_endpoint("etcd-host").etcd_base(),
            "http://etcd-host:2379"
        );
        // explicit scheme is preserved, trailing slash trimmed
        assert_eq!(
            with_endpoint("https://etcd:2379/").etcd_base(),
            "https://etcd:2379"
        );
        assert_eq!(
            with_endpoint("http://etcd:2379").etcd_base(),
            "http://etcd:2379"
        );
    }

    #[test]
    fn validate_rejects_unsupported_subset() {
        // unknown policy is rejected
        let bad =
            Config::try_parse_from(["infera-router", "--router-policy", "least-load"]).unwrap();
        assert!(bad.validate().is_err());
        // round-robin (default) and kv-aware are both accepted
        let ok = Config::try_parse_from(["infera-router"]).unwrap();
        assert!(ok.validate().is_ok());
        let kva = Config::try_parse_from([
            "infera-router",
            "--router-policy",
            "kv-aware",
            "--kv-tokenizer-path",
            "/tmp/tok.json",
        ])
        .unwrap();
        assert!(kva.validate().is_ok());
        // unsupported discovery backend still rejected
        let bad_disc =
            Config::try_parse_from(["infera-router", "--discovery-backend", "k8s"]).unwrap();
        assert!(bad_disc.validate().is_err());
    }
}
