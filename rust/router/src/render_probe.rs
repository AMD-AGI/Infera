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

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::block_hasher::BlockHasher;
use crate::pool::Worker;
use crate::render_variant::{RenderVariant, VariantRegistry};

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
        // The live failure this file exists for: a `/v1/responses` body hashes
        // to nothing unless `responses_input` rebuilds the chat request. Chat
        // probes stay green through that. Tokenise the converted chat body --
        // `/v1/tokenize` runs `_process_messages`, not `_make_request`.
        (
            "responses",
            json!({"input": "What is 2+2?"}),
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
    /// worker_id -> the epoch of the probe currently reserved for it. Keyed by
    /// probe, not by worker: a bare set cannot tell one probe from the next, so
    /// a probe still in flight when its worker left and rejoined would consume
    /// the REPLACEMENT probe's reservation, write the dead instance's verdict,
    /// and send the fresh one down the "left the fleet mid-probe" path -- after
    /// which `claim` refuses forever and the gauge is stuck on a verdict about
    /// an engine that is gone.
    pending: Mutex<HashMap<String, u64>>,
    next_epoch: AtomicU64,
    /// model -> how many workers have ever been judged `Diverged`.
    ///
    /// The gauge cannot answer this: `retain` drops a worker's series the
    /// moment it leaves, so a fleet that rolls THROUGH broken replicas -- each
    /// one diverged, each one replaced before anyone looked -- leaves no trace
    /// at all. A counter survives the worker, which is the whole reason the
    /// Python side has one; this port exported the gauge and not the counter,
    /// so an alert written on it silently never fired on --router-backend rust.
    diverged: Mutex<HashMap<String, u64>>,
    /// Fingerprint of the process last probed under this worker_id. An in-place
    /// restart keeps the id and changes the kv endpoint; without this a
    /// Confirmed/Diverged latch would skip the replacement.
    identity: Mutex<HashMap<String, String>>,
}

impl ParityRegistry {
    /// Reserve this worker for a probe, returning the epoch to record under, or
    /// `None` if one is already running or its verdict already stands.
    /// Reserving and probing are separate steps, so this must be called before
    /// the spawn, not inside it -- which is also why every path that abandons a
    /// probe has to `release`.
    ///
    /// An `Unknown` verdict is not final: it means the probe could not reach
    /// the engine, and a worker that was merely slow to start would otherwise
    /// export -1 for the life of the router.
    ///
    /// `identity` is the process fingerprint (kv endpoint + page size). A
    /// change drops a settled verdict so a replacement on the same worker_id
    /// is probed again.
    pub fn claim(&self, worker_id: &str, model: &str) -> Option<u64> {
        self.claim_identity(worker_id, model, "").map(|(epoch, _)| epoch)
    }

    /// Second value is true when `worker_id` already had a different process
    /// fingerprint: the Confirmed/Diverged latch and recorded variant belong
    /// to the engine that just died.
    pub fn claim_identity(
        &self,
        worker_id: &str,
        model: &str,
        identity: &str,
    ) -> Option<(u64, bool)> {
        let process_replaced = {
            let mut ids = self
                .identity
                .lock()
                .expect("parity identity mutex poisoned");
            match ids.get(worker_id) {
                Some(prev) if prev == identity => false,
                Some(_) => {
                    ids.insert(worker_id.to_string(), identity.to_string());
                    true
                }
                None => {
                    ids.insert(worker_id.to_string(), identity.to_string());
                    false
                }
            }
        };
        if process_replaced {
            self.state
                .lock()
                .expect("parity registry mutex poisoned")
                .remove(worker_id);
            self.pending
                .lock()
                .expect("parity pending mutex poisoned")
                .remove(worker_id);
        }
        if let Some((m, v)) = self
            .state
            .lock()
            .expect("parity registry mutex poisoned")
            .get(worker_id)
        {
            if *v != Parity::Unknown && m == model {
                return None;
            }
        }
        let mut pending = self.pending.lock().expect("parity pending mutex poisoned");
        if pending.contains_key(worker_id) {
            return None;
        }
        let epoch = self.next_epoch.fetch_add(1, Ordering::Relaxed);
        pending.insert(worker_id.to_string(), epoch);
        Some((epoch, process_replaced))
    }

    /// Store a finished probe's verdict -- unless the worker left the fleet
    /// while it was in flight, or this probe's reservation has since been
    /// superseded, in which case the verdict is dropped rather than written
    /// behind `retain`'s back. A probe outlives its worker often enough to
    /// matter: it holds a 10s timeout per body against a worker that may be
    /// shutting down, which is exactly when it is slowest to answer.
    pub fn record(&self, worker_id: &str, epoch: u64, model: &str, verdict: Parity) {
        {
            let mut pending = self.pending.lock().expect("parity pending mutex poisoned");
            if pending.get(worker_id) != Some(&epoch) {
                tracing::debug!(
                    worker = worker_id,
                    "render probe: reservation gone (worker left, or superseded by a \
                     later probe); verdict dropped"
                );
                return;
            }
            pending.remove(worker_id);
        }
        if verdict == Parity::Diverged {
            *self
                .diverged
                .lock()
                .expect("parity diverged mutex poisoned")
                .entry(model.to_string())
                .or_insert(0) += 1;
        }
        self.state
            .lock()
            .expect("parity registry mutex poisoned")
            .insert(worker_id.to_string(), (model.to_string(), verdict));
    }

    /// `(model, count)` for the exposition. Never pruned by `retain`: outliving
    /// the worker is the point.
    pub fn diverged_totals(&self) -> Vec<(String, u64)> {
        let mut out: Vec<_> = self
            .diverged
            .lock()
            .expect("parity diverged mutex poisoned")
            .iter()
            .map(|(m, n)| (m.clone(), *n))
            .collect();
        out.sort();
        out
    }

    /// Hand a reservation back unused.
    ///
    /// Claiming happens in the caller and recording only inside the spawned
    /// task, so every path that abandons the probe in between -- a client that
    /// would not build, a panic, runtime shutdown -- must come through here.
    /// Without it those workers sit in `pending` forever: `claim` refuses them
    /// on every later snapshot, `state` never gains a row, and `/metrics`
    /// exports no series at all for them, which reads as "this policy does no
    /// parity checking" rather than "we never managed to ask".
    pub fn release(&self, worker_id: &str, epoch: u64) {
        let mut pending = self.pending.lock().expect("parity pending mutex poisoned");
        if pending.get(worker_id) == Some(&epoch) {
            pending.remove(worker_id);
        }
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
            .retain(|id, _| alive(id));
        self.identity
            .lock()
            .expect("parity identity mutex poisoned")
            .retain(|id, _| alive(id));
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

/// What the worker says it was launched with.
///
/// `--default-chat-template-kwargs` is merged into every request *before* the
/// template runs and is invisible from everywhere else the router looks, so
/// this endpoint is the only way to know a worker renders a different preamble
/// than we do. `None` means we could not ask -- older engine, non-sglang,
/// unreachable -- which is not the same as "it has none", so the caller keeps
/// its existing assumption rather than recording an empty variant.
pub async fn engine_render_variant(
    client: &reqwest::Client,
    base_url: &str,
) -> Option<RenderVariant> {
    for path in ["/get_server_info", "/v1/server_info"] {
        let resp = match client.get(format!("{base_url}{path}")).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(url = %base_url, path, err = %e, "render probe: server info unreachable");
                return None;
            }
        };
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            continue;
        }
        if !resp.status().is_success() {
            return None;
        }
        let parsed: Value = resp.json().await.ok()?;
        // sglang has served this both flat and nested under `server_args`.
        // `Value::get` answers PRESENCE, so an explicit `null`/`{}` still
        // arrives as `Some` and is recorded as the empty variant -- the engine
        // genuinely has no defaults, and that must win over a fleet flag which
        // does not apply to this worker.
        if let Some(field) = parsed
            .get("default_chat_template_kwargs")
            .or_else(|| parsed.pointer("/server_args/default_chat_template_kwargs"))
        {
            return Some(RenderVariant::from_default_chat_template_kwargs(Some(
                field,
            )));
        }
        // Answered, but does not carry the field at all -- an engine older than
        // the flag, a renamed key, a proxy that trims the payload. That is "we
        // could not ask", not "it has none". Returning the empty variant here
        // would have it RECORDED against this worker, and `for_worker` prefers
        // a recorded entry over the fleet default, so an operator who set
        // --kv-default-chat-template-kwargs to match their fleet would have it
        // silently discarded for every worker and every request.
        tracing::debug!(
            url = %base_url, path,
            "render probe: answered without default_chat_template_kwargs; keeping the \
             router's existing assumption for this worker"
        );
        return None;
    }
    None
}

/// Compare our render against the worker's for every probe body.
///
/// `variant` is the server-side template default this worker was found to
/// render with; the probe applies it exactly as the policy will, so a
/// `Confirmed` here means the policy's hashes are the worker's hashes -- and a
/// mistake in modelling the variant shows up as `Diverged` rather than as a
/// hit rate nobody is watching.
pub async fn probe_worker(
    hasher: &BlockHasher,
    client: &reqwest::Client,
    worker: &Worker,
    variant: &RenderVariant,
) -> (Parity, String) {
    let bodies = probe_bodies();
    let mut unknown: Vec<String> = Vec::new();
    let mut declined: Vec<String> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();

    for (name, template) in &bodies {
        let mut body = template.clone();
        body["model"] = json!(worker.model_name);
        // Normalise before applying the variant, in the engine's order:
        // `_make_request` first, `_process_messages` (which merges
        // --default-chat-template-kwargs) second. The other way round, a
        // Responses probe body silently loses the variant and the probe reports
        // a parity the live path does not have.
        let normalised = crate::responses_input::normalised(&body);
        let Some(ours) = hasher.token_ids_for(&variant.apply(&normalised)) else {
            // NOT folded in with "engine did not answer". The two look alike --
            // neither produced a comparison -- and mean opposite things. An
            // engine that cannot be asked is a limit on the probe. A body the
            // ROUTER declined is a kv-aware outage for that request shape, on
            // this worker, already decided: those requests hash to nothing and
            // route on load. Reporting it as a gap and then returning
            // `Confirmed` is how a DSv4 worker, whose two tool-carrying probe
            // bodies `render_dsv4` refuses and whose model ships no chat
            // template, exports the same green 1 as a worker that matched on
            // all four.
            declined.push(format!("{name}: router declined to render"));
            continue;
        };
        match engine_token_ids(client, &worker.url, &normalised).await {
            Some(theirs) if theirs != ours => mismatches.push(describe(name, &ours, &theirs)),
            Some(_) => {}
            None => unknown.push(format!("{name}: engine did not answer")),
        }
    }

    if !mismatches.is_empty() {
        return (Parity::Diverged, mismatches.join("; "));
    }
    if !declined.is_empty() && declined.len() < bodies.len() {
        // PARTIAL decline is the dangerous shape, and `Diverged` is the honest
        // verdict: the router demonstrably renders for this model, just not for
        // these request shapes, so those requests hash to nothing and route on
        // load while the rest look fine. Folded into `unknown` it returned
        // `Confirmed`, and under the documented "alert on 0" rule a DSv4 worker
        // -- whose two tool-carrying probe bodies `render_dsv4` refuses and
        // whose model ships no chat template -- read as verified-good.
        declined.extend(unknown);
        return (Parity::Diverged, declined.join("; "));
    }
    if !declined.is_empty() || !unknown.is_empty() {
        // Nothing rendered at all (a hasher with no tokenizer), or the engine
        // never answered. Neither is a statement about this worker's render,
        // so neither may claim one.
        declined.extend(unknown);
        return (Parity::Unknown, declined.join("; "));
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
    variants: Arc<VariantRegistry>,
    workers: &[Arc<Worker>],
) {
    if !hasher.is_enabled() || tokio::runtime::Handle::try_current().is_err() {
        return;
    }
    let todo: Vec<(Arc<Worker>, u64)> = workers
        .iter()
        // `/v1/tokenize` is sglang's. vLLM's equivalent differs in both path
        // and payload, so probing it blindly would only produce 404 noise --
        // and an `Unknown` we never asked an honest question to earn.
        .filter(|w| w.engine.eq_ignore_ascii_case("sglang"))
        .filter_map(|w| {
            let identity = format!(
                "{}|{:?}|{:?}|{:?}",
                w.kv_events_endpoint.as_deref().unwrap_or(""),
                w.kv_block_size,
                w.dp_size,
                w.dp_rank
            );
            registry
                .claim_identity(&w.worker_id, &w.model_name, &identity)
                .map(|(epoch, replaced)| {
                    if replaced {
                        variants.forget(&w.worker_id);
                    }
                    (Arc::clone(w), epoch)
                })
        })
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
                // Claimed in the caller, so give every reservation back or
                // these workers are never probed again -- and export no gauge
                // series at all, which reads as "no parity checking here"
                // rather than "we never managed to ask".
                tracing::debug!(err = %e, "render probe: no http client");
                for (w, epoch) in &todo {
                    registry.release(&w.worker_id, *epoch);
                }
                return;
            }
        };
        for (w, epoch) in todo {
            // Ask what it renders with BEFORE checking whether we agree, so the
            // parity verdict is about the render we will actually route on.
            let variant = if variants.per_worker_enabled() {
                match engine_render_variant(&client, &w.url).await {
                    Some(v) => {
                        variants.record(&w.worker_id, v.clone());
                        Arc::new(v)
                    }
                    None => variants.for_worker(&w.worker_id),
                }
            } else {
                variants.for_worker(&w.worker_id)
            };
            let (verdict, detail) = probe_worker(&hasher, &client, &w, &variant).await;
            match verdict {
                Parity::Diverged => tracing::error!(
                    worker = %w.worker_id, model = %w.model_name,
                    variant = %variant.label(), detail,
                    "kv-aware: this worker does NOT render prompts the way the router does, so \
                     cache lookups for it will always miss and it will be routed on load only. \
                     The `variant` field is the server-side --default-chat-template-kwargs the \
                     router read back and already accounted for, so the remaining cause is \
                     something it cannot read: most likely this worker's model directory has a \
                     different chat template than --kv-tokenizer-path, or it was started with \
                     --chat-template"
                ),
                Parity::Unknown => tracing::info!(
                    worker = %w.worker_id, model = %w.model_name, detail,
                    "kv-aware: could not verify prompt rendering against this worker"
                ),
                Parity::Confirmed => tracing::info!(
                    worker = %w.worker_id, model = %w.model_name,
                    variant = %variant.label(), detail,
                    "kv-aware: render verified against this worker"
                ),
            }
            registry.record(&w.worker_id, epoch, &w.model_name, verdict);
        }
    });
}
