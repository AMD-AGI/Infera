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

    /// `round-robin` or `kv-aware` (DP-attention cache-locality routing).
    #[arg(long, default_value = "round-robin")]
    pub router_policy: String,

    /// Only `etcd` is supported by the Rust backend.
    #[arg(long, default_value = "etcd")]
    pub discovery_backend: String,

    /// Only `http` is supported by the Rust backend.
    #[arg(long, default_value = "http")]
    pub request_transport: String,

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
        if self.discovery_backend != "etcd" {
            anyhow::bail!(
                "rust backend supports only --discovery-backend etcd (got {:?})",
                self.discovery_backend
            );
        }
        if self.request_transport != "http" {
            anyhow::bail!(
                "rust backend supports only --request-transport http (got {:?})",
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
