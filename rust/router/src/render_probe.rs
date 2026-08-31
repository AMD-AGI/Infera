//! Ask a worker what it would actually tokenize, and check we agree.
//!
//! kv-aware routing rests on an assumption this router cannot verify on its
//! own: that the prompt it renders is the prompt the engine renders, byte for
//! byte. When that stops being true nothing fails. The block hashes simply
//! never match, every lookup misses, the policy quietly degrades to load
//! balancing, and every health signal -- readiness, kv event flow, cache view
//! size -- stays green. We have found these by noticing a flat 0% hit rate days
//! later.
//!
//! sglang serves `/v1/tokenize`, and for a `messages` body it runs the real
//! `OpenAIServingChat._process_messages`. That makes it ground truth rather
//! than a second opinion: it sees the engine's own template, its own tokenizer,
//! and the server-side merges this router is never told about --
//! `--default-chat-template-kwargs` above all, which has broken kv-aware here
//! before and is invisible from the router by construction.
//!
//! This router carries the larger risk of the two: it re-implements Jinja
//! (minijinja plus hand-written shims for the Python string methods templates
//! call) and hand-ports the native encoders, so it can diverge in ways the
//! Python router -- which drives real `transformers` -- cannot.
//!
//! So on each discovery snapshot we render a few small probe bodies both ways
//! and compare token ids. The verdict is remembered per worker and exported on
//! `/metrics`; a worker is probed once and not re-probed while it stays in the
//! fleet. Best-effort throughout: a worker we cannot confirm still takes
//! traffic, it just cannot be trusted to hit cache -- and now we know that at
//! startup instead of at post-mortem.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::block_hasher::BlockHasher;
use crate::pool::Worker;

/// Deliberately tiny, and deliberately not the unit-test corpus: this runs
/// against production workers at registration, so it must cost the engine
/// almost nothing (`/v1/tokenize` never schedules a forward pass). Each body
/// earns its place by covering a divergence we have actually shipped.
fn probe_bodies() -> Vec<(&'static str, Value)> {
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]);
    vec![
        // The template preamble. GLM-5.3 opens with a "Reasoning Effort: High"
        // system line, so a router that gets the defaults wrong diverges in the
        // FIRST block of every prompt, and in everything chained after it.
        (
            "plain",
            json!({"messages": [{"role": "user", "content": "What is 2+2?"}]}),
        ),
        // An explicit reasoning_effort, which the engine forwards into template
        // scope and, for some templates, remaps onto a boolean toggle instead.
        (
            "effort",
            json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}),
        ),
        // Tools rendered from a full pydantic `Tool.model_dump()`, whose
        // materialised defaults the client never sent.
        (
            "tools",
            json!({"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": tools}),
        ),
        // An assistant turn carrying `tool_calls` with `arguments` in the
        // OpenAI string form. Templates disagree about parsing it; get it wrong
        // and every agentic conversation loses its cache from the second turn
        // on, which is nearly the whole workload.
        (
            "tool_call",
            json!({
                "messages": [
                    {"role": "user", "content": "weather in Paris?"},
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
                    }]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "18C, clear"}
                ],
                "tools": tools
            }),
        ),
    ]
}

/// A three-state verdict, and the third state matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// Every probe body matched; kv-aware is known good on this worker.
    Confirmed,
    /// At least one diverged; kv-aware is known broken on this worker.
    Diverged,
    /// We could not tell -- no such endpoint, unreachable, refused. Reporting
    /// that as a failure would alert on every engine that simply does not serve
    /// `/v1/tokenize`; reporting it as a pass would claim a guarantee we never
    /// obtained.
    Unknown,
}

impl Parity {
    fn gauge(self) -> i8 {
        match self {
            Parity::Confirmed => 1,
            Parity::Diverged => 0,
            Parity::Unknown => -1,
        }
    }
}

/// Per-worker verdicts, for `/metrics` and for "have we already asked this one".
#[derive(Default)]
pub struct ParityRegistry {
    state: Mutex<HashMap<String, (String, Parity)>>,
    /// Workers with a probe already running. A verdict is only recorded once
    /// all four bodies have round-tripped, which takes longer than the gap
    /// between discovery snapshots, so `state` alone does not answer "have we
    /// already asked this one" -- every snapshot would spawn another probe
    /// against a worker still answering the last one.
    pending: Mutex<HashSet<String>>,
}

impl ParityRegistry {
    /// Reserve this worker for a probe, or `false` if one is already running or
    /// finished. Reserving and probing are separate steps, so this must be
    /// called before the spawn, not inside it.
    pub fn claim(&self, worker_id: &str) -> bool {
        if self
            .state
            .lock()
            .expect("parity registry mutex poisoned")
            .contains_key(worker_id)
        {
            return false;
        }
        self.pending
            .lock()
            .expect("parity pending mutex poisoned")
            .insert(worker_id.to_string())
    }

    /// Store a finished probe's verdict -- unless the worker left the fleet
    /// while it was in flight, in which case the verdict is dropped rather than
    /// re-inserted behind `retain`'s back. A probe outlives its worker often
    /// enough to matter: it holds a 10s timeout per body against a worker that
    /// may be shutting down, which is exactly when it is slowest to answer.
    pub fn record(&self, worker_id: &str, model: &str, verdict: Parity) {
        let claimed = self
            .pending
            .lock()
            .expect("parity pending mutex poisoned")
            .remove(worker_id);
        if !claimed {
            tracing::debug!(
                worker = worker_id,
                "render probe: worker left the fleet mid-probe; verdict dropped"
            );
            return;
        }
        self.state
            .lock()
            .expect("parity registry mutex poisoned")
            .insert(worker_id.to_string(), (model.to_string(), verdict));
    }

    /// Forget workers that left the fleet. A gauge keyed by `worker_id` that
    /// outlives its worker keeps exporting a verdict -- possibly a `0` -- about
    /// something that is no longer running.
    pub fn retain<F: Fn(&str) -> bool>(&self, alive: F) {
        self.state
            .lock()
            .expect("parity registry mutex poisoned")
            .retain(|id, _| alive(id));
        self.pending
            .lock()
            .expect("parity pending mutex poisoned")
            .retain(|id| alive(id));
    }

    /// `(worker_id, model, gauge value)` for the metrics exposition.
    pub fn snapshot(&self) -> Vec<(String, String, i8)> {
        let mut out: Vec<_> = self
            .state
            .lock()
            .expect("parity registry mutex poisoned")
            .iter()
            .map(|(id, (model, v))| (id.clone(), model.clone(), v.gauge()))
            .collect();
        out.sort();
        out
    }
}

/// Compare our render against the worker's for every probe body.
pub async fn probe_worker(
    hasher: &BlockHasher,
    client: &reqwest::Client,
    worker: &Worker,
) -> (Parity, String) {
    let bodies = probe_bodies();
    let mut unknown: Vec<String> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();

    for (name, template) in &bodies {
        let mut body = template.clone();
        body["model"] = json!(worker.model_name);
        let Some(ours) = hasher.token_ids_for(&body) else {
            // Not a divergence: the router already knows it cannot reproduce
            // this body and routes it on load. Silent WRONGNESS is the thing
            // this probe exists to find.
            unknown.push(format!("{name}: router declined to render"));
            continue;
        };
        match engine_token_ids(client, &worker.url, &body).await {
            Some(theirs) if theirs != ours => mismatches.push(describe(name, &ours, &theirs)),
            Some(_) => {}
            None => unknown.push(format!("{name}: engine did not answer")),
        }
    }

    if !mismatches.is_empty() {
        return (Parity::Diverged, mismatches.join("; "));
    }
    if unknown.len() == bodies.len() {
        return (Parity::Unknown, unknown.join("; "));
    }
    if !unknown.is_empty() {
        return (
            Parity::Confirmed,
            format!("matched, with gaps: {}", unknown.join("; ")),
        );
    }
    (
        Parity::Confirmed,
        format!("matched on {} probe bodies", bodies.len()),
    )
}

/// `/v1/tokenize` for a chat body, or `None` if this worker cannot answer.
async fn engine_token_ids(
    client: &reqwest::Client,
    base_url: &str,
    body: &Value,
) -> Option<Vec<u32>> {
    let mut req = body.clone();
    // Mirrors what the chat path does with an already-templated string. For a
    // `messages` body sglang returns `_process_messages`' ids and ignores the
    // flag, but pinning it keeps the request honest against an engine that
    // routes the two the same way.
    req["add_special_tokens"] = json!(false);

    for path in ["/v1/tokenize", "/tokenize"] {
        let resp = match client
            .post(format!("{base_url}{path}"))
            .json(&req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(url = %base_url, path, err = %e, "render probe: unreachable");
                return None;
            }
        };
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            continue; // older engine, or a non-sglang one; try the alias
        }
        if !resp.status().is_success() {
            tracing::debug!(
                url = %base_url, path, status = %resp.status(),
                "render probe: tokenize endpoint refused"
            );
            return None;
        }
        let parsed: Value = resp.json().await.ok()?;
        let tokens = parsed.get("tokens")?.as_array()?;
        return tokens
            .iter()
            .map(|t| t.as_u64().map(|v| v as u32))
            .collect();
    }
    None
}

fn describe(name: &str, ours: &[u32], theirs: &[u32]) -> String {
    let at = ours
        .iter()
        .zip(theirs)
        .position(|(a, b)| a != b)
        .unwrap_or(ours.len().min(theirs.len()));
    let lo = at.saturating_sub(4);
    fn window(v: &[u32], lo: usize, hi: usize) -> &[u32] {
        &v[lo.min(v.len())..hi.min(v.len())]
    }
    format!(
        "{name}: diverges at token {at} of {} (router {:?} vs engine {:?})",
        theirs.len(),
        window(ours, lo, at + 4),
        window(theirs, lo, at + 4)
    )
}

/// Probe every worker we have not yet judged, in the background.
///
/// Called from the (synchronous) discovery-snapshot hook, so it must not block
/// and must not care whether the probes finish. Off a Tokio runtime it does
/// nothing rather than starting one behind the caller's back.
pub fn spawn_probes(
    hasher: Arc<BlockHasher>,
    registry: Arc<ParityRegistry>,
    workers: &[Arc<Worker>],
) {
    if !hasher.is_enabled() || tokio::runtime::Handle::try_current().is_err() {
        return;
    }
    let todo: Vec<Arc<Worker>> = workers
        .iter()
        // `/v1/tokenize` is sglang's. vLLM's equivalent differs in both path
        // and payload, so probing it blindly would only produce 404 noise --
        // and an `Unknown` we never asked an honest question to earn.
        .filter(|w| w.engine.eq_ignore_ascii_case("sglang"))
        .filter(|w| registry.claim(&w.worker_id))
        .cloned()
        .collect();
    if todo.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(err = %e, "render probe: no http client");
                return;
            }
        };
        for w in todo {
            let (verdict, detail) = probe_worker(&hasher, &client, &w).await;
            match verdict {
                Parity::Diverged => tracing::error!(
                    worker = %w.worker_id, model = %w.model_name, detail,
                    "kv-aware: this worker does NOT render prompts the way the router does, so \
                     cache lookups for it will always miss and it will be routed on load only. \
                     Most likely causes: the engine was started with \
                     --default-chat-template-kwargs (the router is never told), or its model \
                     directory has a different chat template than --tokenizer-path"
                ),
                Parity::Unknown => tracing::info!(
                    worker = %w.worker_id, model = %w.model_name, detail,
                    "kv-aware: could not verify prompt rendering against this worker"
                ),
                Parity::Confirmed => tracing::info!(
                    worker = %w.worker_id, model = %w.model_name, detail,
                    "kv-aware: render verified against this worker"
                ),
            }
            registry.record(&w.worker_id, &w.model_name, verdict);
        }
    });
}
