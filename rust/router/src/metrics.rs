///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Prometheus metrics for the Rust router data plane.
//!
//! Names and semantics match the Python frontend (`infera/server/metrics.py`)
//! so a single Grafana dashboard scrapes either backend. Percentiles
//! (p50/p90/p99) are derived in Prometheus via `histogram_quantile`.

use std::collections::HashSet;
use std::sync::OnceLock;
use std::time::Instant;

use prometheus::core::Collector;

use prometheus::{
    register_counter_vec, register_gauge, register_gauge_vec, register_histogram_vec,
    register_int_gauge, CounterVec, Encoder, Gauge, GaugeVec, HistogramVec, IntGauge, TextEncoder,
};

struct Metrics {
    request_duration: HistogramVec,
    request_inflight: GaugeVec,
    requests_total: CounterVec,
    ttft: HistogramVec,
    itl: HistogramVec,
    isl: HistogramVec,
    osl: HistogramVec,
    prompt_tokens: CounterVec,
    generation_tokens: CounterVec,
    router_picks: CounterVec,
    pick_cache_hits: HistogramVec,
    pick_request_blocks: HistogramVec,
    prefix_blocks_hit: CounterVec,
    prefix_blocks_total: CounterVec,
    engine_kv_cache_usage: GaugeVec,
    engine_prefix_cache_hit_rate: GaugeVec,
    engine_kv_transfer_queue_reqs: GaugeVec,
    active_workers: Gauge,
    uptime_seconds: IntGauge,
    started: Instant,
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

fn m() -> &'static Metrics {
    METRICS.get_or_init(|| Metrics {
        request_duration: register_histogram_vec!(
            "infera_request_duration_seconds",
            "End-to-end server-observed request latency",
            &["router", "outcome"],
            vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,]
        )
        .expect("register request_duration"),
        request_inflight: register_gauge_vec!(
            "infera_request_inflight",
            "Currently in-flight requests at the server",
            &["router"]
        )
        .expect("register request_inflight"),
        requests_total: register_counter_vec!(
            "infera_requests_total",
            "Completed requests by router and outcome",
            &["router", "outcome"]
        )
        .expect("register requests_total"),
        ttft: register_histogram_vec!(
            "infera_time_to_first_token_seconds",
            "Server-observed time from dispatch to the first reply byte. Labeled by the prefill worker; decode time is infera_inter_token_latency_seconds.",
            &["router", "model", "prefill_worker"],
            vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )
        .expect("register ttft"),
        itl: register_histogram_vec!(
            "infera_inter_token_latency_seconds",
            "Mean per-request inter-token latency after TTFT",
            &["router", "model", "decode_worker"],
            vec![0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.0]
        )
        .expect("register itl"),
        isl: register_histogram_vec!(
            "infera_input_sequence_tokens",
            "Prompt length in tokens (ISL)",
            &["router", "model"],
            vec![64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0]
        )
        .expect("register isl"),
        osl: register_histogram_vec!(
            "infera_output_sequence_tokens",
            "Generated length in tokens (OSL)",
            &["router", "model"],
            vec![4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0]
        )
        .expect("register osl"),
        prompt_tokens: register_counter_vec!(
            "infera_prompt_tokens_total",
            "Prompt tokens observed on successful requests",
            &["router", "model", "prefill_worker"]
        )
        .expect("register prompt_tokens"),
        generation_tokens: register_counter_vec!(
            "infera_generation_tokens_total",
            "Generated tokens observed on successful requests",
            &["router", "model", "decode_worker"]
        )
        .expect("register generation_tokens"),
        router_picks: register_counter_vec!(
            "infera_router_picks_total",
            "Total pick() decisions",
            &["role", "worker_id"]
        )
        .expect("register router_picks"),
        pick_cache_hits: register_histogram_vec!(
            "infera_router_pick_cache_hits",
            "Cache blocks the picked worker already had",
            &["role"],
            vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,]
        )
        .expect("register pick_cache_hits"),
        pick_request_blocks: register_histogram_vec!(
            "infera_router_pick_request_blocks",
            "Request block count at pick time",
            &["role"],
            vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,]
        )
        .expect("register pick_request_blocks"),
        prefix_blocks_hit: register_counter_vec!(
            "infera_prefix_cache_blocks_hit_total",
            "Cached KV blocks credited at pick time",
            &["role"]
        )
        .expect("register prefix_blocks_hit"),
        prefix_blocks_total: register_counter_vec!(
            "infera_prefix_cache_blocks_total",
            "Request KV blocks seen at pick time",
            &["role"]
        )
        .expect("register prefix_blocks_total"),
        engine_kv_cache_usage: register_gauge_vec!(
            "infera_engine_kv_cache_usage",
            "Engine-reported KV-cache pool occupancy fraction (0-1)",
            &["worker_id", "engine", "disagg_mode"]
        )
        .expect("register engine_kv_cache_usage"),
        engine_prefix_cache_hit_rate: register_gauge_vec!(
            "infera_engine_prefix_cache_hit_rate",
            "Engine-reported prefix cache hit rate when published",
            &["worker_id", "engine", "disagg_mode"]
        )
        .expect("register engine_prefix_cache_hit_rate"),
        engine_kv_transfer_queue_reqs: register_gauge_vec!(
            "infera_engine_kv_transfer_queue_reqs",
            "Engine PD KV transfer / bootstrap queue depth from worker /metrics",
            &["worker_id", "queue"]
        )
        .expect("register engine_kv_transfer_queue_reqs"),
        active_workers: register_gauge!(
            "infera_router_active_workers",
            "Number of ACTIVE workers known to this router"
        )
        .expect("register active_workers"),
        uptime_seconds: register_int_gauge!(
            "infera_router_uptime_seconds",
            "Seconds since this router process started"
        )
        .expect("register uptime"),
        started: Instant::now(),
    })
}

/// Render Prometheus text exposition (default registry).
pub fn render() -> (Vec<u8>, &'static str) {
    let metrics = m();
    metrics
        .uptime_seconds
        .set(metrics.started.elapsed().as_secs() as i64);
    let encoder = TextEncoder::new();
    let families = prometheus::gather();
    let mut buf = Vec::new();
    encoder.encode(&families, &mut buf).expect("encode metrics");
    (buf, "text/plain; version=0.0.4; charset=utf-8")
}

pub fn set_active_workers(n: f64) {
    m().active_workers.set(n);
}

pub fn record_pick(role: &str, worker_id: &str, cache_hits: usize, request_blocks: usize) {
    let metrics = m();
    metrics
        .router_picks
        .with_label_values(&[role, worker_id])
        .inc();
    metrics
        .pick_cache_hits
        .with_label_values(&[role])
        .observe(cache_hits as f64);
    metrics
        .pick_request_blocks
        .with_label_values(&[role])
        .observe(request_blocks as f64);
    if request_blocks > 0 {
        metrics
            .prefix_blocks_hit
            .with_label_values(&[role])
            .inc_by(cache_hits as f64);
        metrics
            .prefix_blocks_total
            .with_label_values(&[role])
            .inc_by(request_blocks as f64);
    }
}

static SCRAPE_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();

/// Serialize one /metrics scrape. Engine gauges are cleared, workers are
/// fetched, then the registry is rendered; a second scrape must not clear
/// gauges the first one has already filled.
pub async fn scrape_lock() -> tokio::sync::MutexGuard<'static, ()> {
    SCRAPE_LOCK
        .get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// Drop SLA series whose worker id is no longer in `active`.
///
/// Histogram and counter children stay registered until removed, and a
/// replaced pod receives a new worker id. An empty worker label is kept.
pub fn prune_departed_workers(active: &HashSet<String>) {
    let metrics = m();
    for values in stale_label_values(
        &metrics.ttft,
        &["router", "model", "prefill_worker"],
        &["prefill_worker"],
        active,
    ) {
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let _ = metrics.ttft.remove_label_values(&refs);
    }
    for values in stale_label_values(
        &metrics.itl,
        &["router", "model", "decode_worker"],
        &["decode_worker"],
        active,
    ) {
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let _ = metrics.itl.remove_label_values(&refs);
    }
    for values in stale_label_values(
        &metrics.prompt_tokens,
        &["router", "model", "prefill_worker"],
        &["prefill_worker"],
        active,
    ) {
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let _ = metrics.prompt_tokens.remove_label_values(&refs);
    }
    for values in stale_label_values(
        &metrics.generation_tokens,
        &["router", "model", "decode_worker"],
        &["decode_worker"],
        active,
    ) {
        let refs: Vec<&str> = values.iter().map(String::as_str).collect();
        let _ = metrics.generation_tokens.remove_label_values(&refs);
    }
}

fn stale_label_values(
    metric: &impl Collector,
    label_names: &[&str],
    worker_labels: &[&str],
    active: &HashSet<String>,
) -> Vec<Vec<String>> {
    let mut stale = Vec::new();
    for family in metric.collect() {
        for sample in family.get_metric() {
            let pairs: Vec<(String, String)> = sample
                .get_label()
                .iter()
                .map(|label| (label.get_name().to_string(), label.get_value().to_string()))
                .collect();
            let departed = pairs.iter().any(|(name, value)| {
                worker_labels.iter().any(|want| name == want)
                    && !value.is_empty()
                    && !active.contains(value)
            });
            if !departed {
                continue;
            }
            stale.push(
                label_names
                    .iter()
                    .map(|name| {
                        pairs
                            .iter()
                            .find(|(n, _)| n == name)
                            .map(|(_, value)| value.clone())
                            .unwrap_or_default()
                    })
                    .collect(),
            );
        }
    }
    stale
}

/// Drop engine gauges before a scrape round so departed workers leave the exposition.
pub fn clear_engine_gauges() {
    let metrics = m();
    metrics.engine_kv_cache_usage.reset();
    metrics.engine_prefix_cache_hit_rate.reset();
    metrics.engine_kv_transfer_queue_reqs.reset();
}

/// Update engine gauges from one worker's Prometheus text exposition.
pub fn apply_engine_scrape(worker_id: &str, engine: &str, disagg_mode: &str, text: &str) {
    let metrics = m();
    let mode = if disagg_mode.is_empty() {
        "mixed"
    } else {
        disagg_mode
    };
    let eng = engine.to_ascii_lowercase();

    let kv_names: &[&str] = match eng.as_str() {
        "sglang" => &["sglang:token_usage"],
        "vllm" => &["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
        _ => &[],
    };
    for name in kv_names {
        if let Some(v) = mean_metric(text, name) {
            metrics
                .engine_kv_cache_usage
                .with_label_values(&[worker_id, engine, mode])
                .set(v);
            break;
        }
    }

    if eng == "sglang" {
        if let Some(v) = mean_metric(text, "sglang:cache_hit_rate") {
            metrics
                .engine_prefix_cache_hit_rate
                .with_label_values(&[worker_id, engine, mode])
                .set(v);
        }
        for (queue, name) in [
            (
                "prefill_bootstrap",
                "sglang:num_prefill_bootstrap_queue_reqs",
            ),
            ("prefill_inflight", "sglang:num_prefill_inflight_queue_reqs"),
            ("decode_prealloc", "sglang:num_decode_prealloc_queue_reqs"),
            ("decode_transfer", "sglang:num_decode_transfer_queue_reqs"),
        ] {
            if let Some(v) = sum_metric(text, name) {
                metrics
                    .engine_kv_transfer_queue_reqs
                    .with_label_values(&[worker_id, queue])
                    .set(v);
            }
        }
    }
}

fn sum_metric(text: &str, name: &str) -> Option<f64> {
    let mut total = 0.0;
    let mut found = false;
    for (metric_name, value) in metric_samples(text) {
        if metric_name == name {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn mean_metric(text: &str, name: &str) -> Option<f64> {
    let mut total = 0.0;
    let mut n = 0usize;
    for (metric_name, value) in metric_samples(text) {
        if metric_name == name {
            total += value;
            n += 1;
        }
    }
    (n > 0).then_some(total / n as f64)
}

/// Yield `(metric_name, value)` for each non-comment sample line.
fn metric_samples(text: &str) -> impl Iterator<Item = (&str, f64)> + '_ {
    text.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (name_part, value_part) = line.rsplit_once(char::is_whitespace)?;
        let value: f64 = value_part.trim().parse().ok()?;
        let name = name_part
            .split_once('{')
            .map(|(n, _)| n)
            .unwrap_or(name_part)
            .trim();
        Some((name, value))
    })
}

const VLLM_FEDERATE: &[&str] = &[
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens",
    "vllm:generation_tokens_total",
    "vllm:generation_tokens",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:request_success_total",
    "vllm:request_success",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_max_num_generation_tokens",
];

const SGLANG_FEDERATE: &[&str] = &[
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:time_to_first_token_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_used_tokens",
];

fn federate_families(engine: &str) -> &'static [&'static str] {
    match engine.to_ascii_lowercase().as_str() {
        "vllm" => VLLM_FEDERATE,
        "sglang" => SGLANG_FEDERATE,
        _ => &[],
    }
}

fn metric_family(name: &str) -> &str {
    for suffix in ["_bucket", "_sum", "_count", "_created"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

fn family_allowed(families: &[&str], name: &str) -> bool {
    let base = metric_family(name);
    families.contains(&base)
}

fn escape_prom_label(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

fn inject_labels(line: &str, worker_id: &str, engine: &str) -> String {
    let Some((name_part, value_part)) = line.rsplit_once(char::is_whitespace) else {
        return line.to_string();
    };
    let wid = escape_prom_label(worker_id);
    let eng = escape_prom_label(engine);
    if let Some((name, labels)) = name_part.split_once('{') {
        let inner = labels.trim_end_matches('}');
        let has_worker = inner.contains("worker_id=\"");
        let has_engine = inner.contains("engine=\"");
        let mut extras = String::new();
        if !has_worker {
            extras.push_str(&format!(r#"worker_id="{wid}""#));
        }
        if !has_engine {
            if !extras.is_empty() {
                extras.push(',');
            }
            extras.push_str(&format!(r#"engine="{eng}""#));
        }
        if extras.is_empty() {
            return line.to_string();
        }
        if inner.is_empty() {
            format!("{name}{{{extras}}} {value_part}")
        } else {
            format!("{name}{{{inner},{extras}}} {value_part}")
        }
    } else {
        format!(r#"{name_part}{{worker_id="{wid}",engine="{eng}"}} {value_part}"#)
    }
}

/// Filter allowlisted engine series and stamp `worker_id` / `engine` labels.
pub fn federate_engine_metrics(text: &str, worker_id: &str, engine: &str) -> String {
    let families = federate_families(engine);
    if families.is_empty() || text.is_empty() {
        return String::new();
    }
    let eng_label = engine.to_ascii_lowercase();
    let mut out = String::new();
    for raw in text.lines() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            let mut parts = rest.split_whitespace();
            let kind = parts.next().unwrap_or("");
            let name = parts.next().unwrap_or("");
            if (kind == "HELP" || kind == "TYPE") && family_allowed(families, name) {
                out.push_str(line);
                out.push('\n');
            }
            continue;
        }
        let name = line
            .split_once('{')
            .map(|(n, _)| n)
            .or_else(|| line.split_whitespace().next())
            .unwrap_or("");
        if !family_allowed(families, name) {
            continue;
        }
        out.push_str(&inject_labels(line, worker_id, &eng_label));
        out.push('\n');
    }
    out
}

/// Join per-worker federated text, keeping one HELP/TYPE per family.
pub fn merge_federated_exposition(parts: &[String]) -> String {
    let mut seen_meta: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    let mut out = String::new();
    for part in parts {
        if part.is_empty() {
            continue;
        }
        for raw in part.lines() {
            let line = raw.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                let mut parts = rest.split_whitespace();
                let kind = parts.next().unwrap_or("").to_ascii_uppercase();
                let name = parts.next().unwrap_or("");
                if kind != "HELP" && kind != "TYPE" {
                    continue;
                }
                let family = metric_family_name(name);
                let key = (kind, family.to_string());
                if !seen_meta.insert(key) {
                    continue;
                }
                out.push_str(line);
                out.push('\n');
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Strip Prometheus suffix so HELP/TYPE keys match sample families.
fn metric_family_name(name: &str) -> &str {
    for suffix in ["_bucket", "_sum", "_count", "_created"] {
        if let Some(base) = name.strip_suffix(suffix) {
            return base;
        }
    }
    name
}

const MAX_PARTIAL_FRAME: usize = 1 << 16;

/// Per-request lifetime tracker (mirrors Python ``track_request`` + observer).
pub struct RequestTracker {
    router: &'static str,
    model: String,
    start: Instant,
    ttft: Option<f64>,
    frames: u64,
    isl: Option<u64>,
    osl: Option<u64>,
    outcome: String,
    closed: bool,
    /// Trailing SSE bytes that did not end on a newline in the last chunk.
    partial: Vec<u8>,
    prefill_worker: String,
    decode_worker: String,
}

impl RequestTracker {
    pub fn start(router: &'static str, model: &str) -> Self {
        m().request_inflight.with_label_values(&[router]).inc();
        Self {
            router,
            model: model.to_string(),
            start: Instant::now(),
            ttft: None,
            frames: 0,
            isl: None,
            osl: None,
            outcome: "error".into(),
            closed: false,
            partial: Vec::new(),
            prefill_worker: String::new(),
            decode_worker: String::new(),
        }
    }

    /// Record the workers chosen for this request (per-worker SLA labels).
    pub fn set_workers(&mut self, prefill_worker: &str, decode_worker: &str) {
        if !prefill_worker.is_empty() {
            self.prefill_worker = prefill_worker.to_string();
        }
        if !decode_worker.is_empty() {
            self.decode_worker = decode_worker.to_string();
        }
    }

    /// Mixed (colocated) request: both labels point at the same worker.
    pub fn start_mixed(model: &str, worker_id: &str) -> Self {
        let mut t = Self::start("mixed", model);
        t.set_workers(worker_id, worker_id);
        t
    }

    /// PD-disaggregated request: label by the picked prefill and decode workers.
    pub fn start_disagg(model: &str, prefill_worker: &str, decode_worker: &str) -> Self {
        let mut t = Self::start("disagg", model);
        t.set_workers(prefill_worker, decode_worker);
        t
    }

    pub fn set_outcome(&mut self, outcome: &str) {
        self.outcome = outcome.to_string();
    }

    pub fn set_input_tokens(&mut self, n: u64) {
        if n > 0 {
            self.isl = Some(n);
        }
    }

    pub fn set_output_tokens(&mut self, n: u64) {
        if n > 0 {
            self.osl = Some(n);
        }
    }

    /// First reply byte (streaming TTFT proxy). Idempotent.
    pub fn first_byte(&mut self) {
        if self.ttft.is_none() {
            self.ttft = Some(self.start.elapsed().as_secs_f64());
        }
    }

    /// Count one SSE data frame as a generated token estimate.
    pub fn note_frame(&mut self) {
        self.first_byte();
        self.frames = self.frames.saturating_add(1);
    }

    pub fn observe_stream_chunk(&mut self, chunk: &[u8]) {
        // Carry a trailing partial line across HTTP chunk boundaries so a split
        // usage frame is not counted as a token and is parsed when complete.
        self.partial.extend_from_slice(chunk);
        let buf = std::mem::take(&mut self.partial);
        let mut frames: Vec<&[u8]> = buf.split(|&b| b == b'\n').collect();
        let tail = frames.pop().unwrap_or_default();
        self.partial = if tail.len() <= MAX_PARTIAL_FRAME {
            tail.to_vec()
        } else {
            Vec::new()
        };
        for frame in frames {
            let line = trim_ascii(frame);
            if !line.starts_with(b"data:") || line.starts_with(b"data: [DONE]") {
                continue;
            }
            if let Some(usage) = parse_usage_tokens(line) {
                if let Some(p) = usage.0 {
                    self.set_input_tokens(p);
                }
                if let Some(c) = usage.1 {
                    self.set_output_tokens(c);
                }
                continue;
            }
            self.note_frame();
        }
    }

    pub fn finish(mut self) {
        self.close(/* emit */ true);
    }

    /// Drop in-flight accounting without emitting request/duration counters.
    /// Used when a worker attempt fails before commit and the caller will retry.
    pub fn discard(mut self) {
        self.close(/* emit */ false);
    }
}

impl Drop for RequestTracker {
    fn drop(&mut self) {
        // Failed failover attempts must not emit; success paths call finish().
        self.close(/* emit */ false);
    }
}

impl RequestTracker {
    fn close(&mut self, emit: bool) {
        if self.closed {
            return;
        }
        self.closed = true;
        let metrics = m();
        let router = self.router;
        let model = self.model.as_str();
        let outcome = self.outcome.as_str();
        metrics.request_inflight.with_label_values(&[router]).dec();
        if !emit {
            return;
        }
        metrics
            .request_duration
            .with_label_values(&[router, outcome])
            .observe(self.start.elapsed().as_secs_f64());
        metrics
            .requests_total
            .with_label_values(&[router, outcome])
            .inc();
        if outcome != "ok" {
            return;
        }
        let osl = self.osl.unwrap_or(self.frames);
        let prefill = self.prefill_worker.as_str();
        let decode = self.decode_worker.as_str();
        if let Some(isl) = self.isl {
            metrics
                .isl
                .with_label_values(&[router, model])
                .observe(isl as f64);
            metrics
                .prompt_tokens
                .with_label_values(&[router, model, prefill])
                .inc_by(isl as f64);
        }
        if osl > 0 {
            metrics
                .osl
                .with_label_values(&[router, model])
                .observe(osl as f64);
            metrics
                .generation_tokens
                .with_label_values(&[router, model, decode])
                .inc_by(osl as f64);
        }
        if let Some(ttft) = self.ttft {
            metrics
                .ttft
                .with_label_values(&[router, model, prefill])
                .observe(ttft);
            if osl > 1 {
                let decode_time = (self.start.elapsed().as_secs_f64() - ttft).max(0.0);
                metrics
                    .itl
                    .with_label_values(&[router, model, decode])
                    .observe(decode_time / (osl as f64 - 1.0));
            }
        }
    }
}

fn trim_ascii(s: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = s.len();
    while start < end && s[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && s[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &s[start..end]
}

/// Best-effort extract `(prompt_tokens, completion_tokens)` from an SSE data line.
fn parse_usage_tokens(line: &[u8]) -> Option<(Option<u64>, Option<u64>)> {
    let json = line.strip_prefix(b"data:")?;
    let json = trim_ascii(json);
    let v: serde_json::Value = serde_json::from_slice(json).ok()?;
    let usage = v.get("usage")?;
    let prompt = usage.get("prompt_tokens").and_then(|x| x.as_u64());
    let completion = usage.get("completion_tokens").and_then(|x| x.as_u64());
    if prompt.is_none() && completion.is_none() {
        return None;
    }
    Some((prompt, completion))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    #[test]
    fn federate_vllm_keeps_buckets_and_injects_labels() {
        let text = concat!(
            "# TYPE vllm:e2e_request_latency_seconds histogram\n",
            "vllm:e2e_request_latency_seconds_bucket{le=\"0.5\",model_name=\"m\"} 1\n",
            "vllm:num_requests_running{model_name=\"m\"} 2\n",
            "process_cpu_seconds_total 9\n",
        );
        let out = federate_engine_metrics(text, "w1", "vllm");
        assert!(out.contains("vllm:e2e_request_latency_seconds_bucket"));
        assert!(out.contains(r#"worker_id="w1""#));
        assert!(out.contains(r#"engine="vllm""#));
        assert!(!out.contains("process_cpu_seconds_total"));
    }

    #[test]
    fn merge_federated_dedupes_help_and_type() {
        let a = federate_engine_metrics(
            concat!(
                "# HELP vllm:num_requests_running Running\n",
                "# TYPE vllm:num_requests_running gauge\n",
                "vllm:num_requests_running{model_name=\"m\"} 1\n",
            ),
            "w1",
            "vllm",
        );
        let b = federate_engine_metrics(
            concat!(
                "# HELP vllm:num_requests_running Running\n",
                "# TYPE vllm:num_requests_running gauge\n",
                "vllm:num_requests_running{model_name=\"m\"} 2\n",
            ),
            "w2",
            "vllm",
        );
        let out = merge_federated_exposition(&[a, b]);
        assert_eq!(out.matches("# HELP vllm:num_requests_running").count(), 1);
        assert_eq!(out.matches("# TYPE vllm:num_requests_running").count(), 1);
        assert!(out.contains(r#"worker_id="w1""#));
        assert!(out.contains(r#"worker_id="w2""#));
    }

    #[test]
    fn stream_chunk_carries_partial_usage_frame() {
        let mut t = RequestTracker::start("mixed", "m");
        t.observe_stream_chunk(b"data: {\"usage\":{\"prompt_tokens\":3,\"completion_tok");
        t.observe_stream_chunk(b"ens\":7}}\n");
        assert_eq!(t.isl, Some(3));
        assert_eq!(t.osl, Some(7));
        assert_eq!(t.frames, 0);
        t.discard();
    }

    #[test]
    fn ttft_series_follow_the_prefill_worker() {
        for (prefill, decode) in [
            ("p-card-1", "d-card-1"),
            ("p-card-1", "d-card-2"),
            ("p-card-2", "d-card-1"),
            ("p-card-2", "d-card-2"),
        ] {
            let mut tracker = RequestTracker::start("mixed", "card-model");
            tracker.set_workers(prefill, decode);
            tracker.first_byte();
            tracker.set_outcome("ok");
            tracker.finish();
        }
        let (buf, _) = render();
        let text = String::from_utf8(buf).unwrap();
        let lines: Vec<_> = text
            .lines()
            .filter(|line| {
                line.starts_with("infera_time_to_first_token_seconds_count")
                    && line.contains("card-model")
            })
            .collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines.iter().all(|line| !line.contains("decode_worker")));
    }

    #[test]
    fn departed_worker_sla_series_are_dropped() {
        let mut gone = RequestTracker::start("mixed", "prune-model");
        gone.set_workers("p-gone", "d-keep");
        gone.first_byte();
        gone.set_input_tokens(4);
        gone.set_output_tokens(2);
        gone.set_outcome("ok");
        gone.finish();
        let mut stay = RequestTracker::start("mixed", "prune-model");
        stay.set_workers("p-keep", "d-keep");
        stay.first_byte();
        stay.set_outcome("ok");
        stay.finish();

        let (buf, _) = render();
        let text = String::from_utf8(buf).unwrap();
        let mut active = worker_ids_in(&text);
        active.remove("p-gone");
        active.insert("p-keep".to_string());
        active.insert("d-keep".to_string());
        prune_departed_workers(&active);

        let (buf, _) = render();
        let text = String::from_utf8(buf).unwrap();
        assert!(!text.contains("p-gone"));
        assert!(text.contains("p-keep"));
    }

    fn worker_ids_in(text: &str) -> HashSet<String> {
        let mut ids = HashSet::new();
        for key in ["prefill_worker=\"", "decode_worker=\""] {
            for line in text.lines() {
                let Some(start) = line.find(key) else {
                    continue;
                };
                let rest = &line[start + key.len()..];
                let Some(end) = rest.find('"') else {
                    continue;
                };
                if !rest[..end].is_empty() {
                    ids.insert(rest[..end].to_string());
                }
            }
        }
        ids
    }

    #[tokio::test]
    async fn scrapes_do_not_overlap() {
        let _guard = scrape_lock().await;
        let entered = Arc::new(AtomicBool::new(false));
        let flag = entered.clone();
        let handle = tokio::spawn(async move {
            let _guard = scrape_lock().await;
            flag.store(true, Ordering::SeqCst);
        });
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(!entered.load(Ordering::SeqCst));
        drop(_guard);
        handle.await.unwrap();
        assert!(entered.load(Ordering::SeqCst));
    }
}
