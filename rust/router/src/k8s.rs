///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Minimal in-cluster Kubernetes API access.
//!
//! Mirrors `infera.common.k8s_client`, and the dependency-light style the etcd
//! backend already uses: no kube-rs, no protobuf, just reqwest against the API
//! server with the Pod's mounted ServiceAccount token and CA.

use std::time::Duration;

use anyhow::{Context, Result};

const SA_DIR: &str = "/var/run/secrets/kubernetes.io/serviceaccount";

/// The Pod's own namespace, or `default` when not running in a cluster.
pub fn in_cluster_namespace() -> String {
    std::fs::read_to_string(format!("{SA_DIR}/namespace"))
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "default".to_string())
}

pub fn api_server_base() -> String {
    let host = std::env::var("KUBERNETES_SERVICE_HOST")
        .unwrap_or_else(|_| "kubernetes.default.svc".to_string());
    let port = std::env::var("KUBERNETES_SERVICE_PORT").unwrap_or_else(|_| "443".to_string());
    format!("https://{host}:{port}")
}

/// A client authenticated to the in-cluster API server.
///
/// `timeout` is `None` for the long-lived watch stream and finite for one-shot
/// lists. The token is read on every call rather than cached: projected
/// ServiceAccount tokens are rotated (an hour by default), so a client built
/// once at startup would start getting 401s mid-run. Rebuilding per connection
/// is what the Python side does for the same reason.
pub fn make_client(timeout: Option<Duration>) -> Result<reqwest::Client> {
    let token = std::fs::read_to_string(format!("{SA_DIR}/token"))
        .context("reading the ServiceAccount token (is this running in a Pod?)")?;
    let mut headers = reqwest::header::HeaderMap::new();
    let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token.trim()))
        .context("ServiceAccount token is not a valid header value")?;
    auth.set_sensitive(true);
    headers.insert(reqwest::header::AUTHORIZATION, auth);

    let mut builder = reqwest::Client::builder().default_headers(headers);
    // The API server's certificate is signed by the cluster CA, which is not in
    // the system roots.
    if let Ok(pem) = std::fs::read(format!("{SA_DIR}/ca.crt")) {
        builder = builder.add_root_certificate(
            reqwest::Certificate::from_pem(&pem).context("parsing the cluster CA")?,
        );
    }
    builder = match timeout {
        Some(t) => builder.timeout(t),
        // A watch is meant to stay open, so it gets no total timeout at all --
        // reqwest applies none unless asked. Connecting is still bounded, and
        // reconnection is the caller's job.
        None => builder.connect_timeout(Duration::from_secs(10)),
    };
    builder.build().context("building the Kubernetes client")
}
