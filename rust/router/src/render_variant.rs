//! The server-side template defaults a worker was launched with.
//!
//! `--default-chat-template-kwargs` is the one input to the engine's render
//! that the router is never told about. The client does not send it, discovery
//! does not carry it, and it is applied *before* the template runs -- so a
//! router that does not model it renders a different preamble than the worker,
//! for every request, forever. We have shipped exactly that: on
//! `infera-glm53-pd-1p1d-stable` role1 the engine held
//! `{"reasoning_effort": "high"}` and the router rendered `Max`, diverging at
//! token 8 of 13.
//!
//! Its signature is nastier than a flat zero hit rate. The merge is a
//! `setdefault`, so a request that *does* carry `reasoning_effort` agrees with
//! the engine and hits normally -- only the requests that omit it diverge. The
//! symptom is a hit rate that is merely lower than it should be, which is why
//! this one hid for as long as it did.
//!
//! A variant is not per worker, it is per launch configuration: every worker in
//! a role shares one. A fleet with a single configuration has a single variant,
//! renders once, and pays nothing for this machinery.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

use serde_json::{Map, Value};

/// The template-scope defaults one group of workers renders with.
///
/// `Default` is the empty variant: no server-side defaults, the body renders as
/// the client sent it. That is what every worker gets until something tells the
/// router otherwise, and it is exactly today's behaviour.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderVariant {
    kwargs: BTreeMap<String, Value>,
}

impl RenderVariant {
    /// From `--default-chat-template-kwargs` as the engine reports it. A
    /// non-object (or an absent value) is the empty variant -- sglang's own
    /// `server_args` validation rejects a non-dict, so there is nothing here to
    /// be strict about that the engine has not already refused.
    pub fn from_default_chat_template_kwargs(value: Option<&Value>) -> Self {
        let kwargs = value
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        RenderVariant { kwargs }
    }

    pub fn is_empty(&self) -> bool {
        self.kwargs.is_empty()
    }

    /// Stable id, so the policy can hash a request once per *variant* rather
    /// than once per worker. The empty variant is 0 by construction: a fleet
    /// where nothing is configured collapses to a single key and the hash cache
    /// behaves exactly as it did before variants existed.
    pub fn id(&self) -> u64 {
        if self.kwargs.is_empty() {
            return 0;
        }
        // BTreeMap iterates in key order, so this does not depend on how the
        // engine happened to serialise its dict.
        let mut buf: Vec<u8> = Vec::new();
        for (k, v) in &self.kwargs {
            buf.extend_from_slice(k.as_bytes());
            buf.push(0);
            buf.extend_from_slice(v.to_string().as_bytes());
            buf.push(0);
        }
        // Never collide with the empty variant, whatever the digest says.
        xxhash_rust::xxh3::xxh3_64(&buf) | 1
    }

    /// A one-line rendering for logs and metrics labels.
    pub fn label(&self) -> String {
        if self.kwargs.is_empty() {
            return "default".to_string();
        }
        let inner: Vec<String> = self
            .kwargs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        inner.join(",")
    }

    /// The body the engine will actually template, given this worker's defaults.
    ///
    /// Mirrors `serving_chat.py:1026-1032` exactly, and the two details that
    /// look incidental are the whole contract:
    ///
    ///   * `setdefault`, not `insert` -- an explicit `chat_template_kwargs` from
    ///     the client wins over the server default. Overwriting instead would
    ///     make the router diverge on precisely the requests that agree today.
    ///   * the merged `reasoning_effort` is promoted back onto the top-level
    ///     field when the client left it unset, because downstream (the
    ///     `effort_kwarg` remap, `extra_template_kwargs`) reads it from there.
    ///     This promotion happens BEFORE the low/medium/high handling, so
    ///     applying the variant late would silently skip it.
    ///
    /// Rewrites the body rather than patching the template context, so the
    /// native encoders (`encoding_k3`, `encoding_dsv4`) -- which model
    /// `chat_template_kwargs` themselves and never see that context -- get the
    /// merge too.
    ///
    /// Borrows unchanged for the empty variant, which is the common path.
    pub fn apply<'a>(&self, body: &'a Value) -> Cow<'a, Value> {
        if self.kwargs.is_empty() {
            return Cow::Borrowed(body);
        }
        let Some(obj) = body.as_object() else {
            return Cow::Borrowed(body);
        };
        let mut ctk: Map<String, Value> = obj
            .get("chat_template_kwargs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (k, v) in &self.kwargs {
            ctk.entry(k.clone()).or_insert_with(|| v.clone());
        }
        let mut out = obj.clone();
        if out.get("reasoning_effort").is_none_or(|v| v.is_null()) {
            if let Some(effort) = ctk.get("reasoning_effort") {
                if !effort.is_null() {
                    out.insert("reasoning_effort".into(), effort.clone());
                }
            }
        }
        out.insert("chat_template_kwargs".into(), Value::Object(ctk));
        Cow::Owned(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn variant(v: Value) -> RenderVariant {
        RenderVariant::from_default_chat_template_kwargs(Some(&v))
    }

    #[test]
    fn the_empty_variant_is_a_no_op_and_does_not_clone() {
        let v = RenderVariant::default();
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(matches!(v.apply(&body), Cow::Borrowed(_)));
        assert_eq!(v.id(), 0);
        assert_eq!(v.label(), "default");
    }

    #[test]
    fn a_server_default_reaches_both_the_kwargs_and_the_top_level_field() {
        // This is the divergence measured on role1: the engine renders the
        // preamble for "high" and, without this, the router renders the
        // template's own fallback.
        let v = variant(json!({"reasoning_effort": "high"}));
        let out = v.apply(&json!({"messages": []})).into_owned();
        assert_eq!(out["reasoning_effort"], json!("high"));
        assert_eq!(
            out["chat_template_kwargs"]["reasoning_effort"],
            json!("high")
        );
    }

    #[test]
    fn the_request_wins_over_the_server_default() {
        // `setdefault`. Getting this backwards would break the requests that
        // are correct today -- the ones that send an explicit effort.
        let v = variant(json!({"reasoning_effort": "high", "enable_thinking": true}));
        let out = v
            .apply(&json!({
                "messages": [],
                "chat_template_kwargs": {"reasoning_effort": "low"}
            }))
            .into_owned();
        assert_eq!(
            out["chat_template_kwargs"]["reasoning_effort"],
            json!("low")
        );
        assert_eq!(out["chat_template_kwargs"]["enable_thinking"], json!(true));
        assert_eq!(
            out["reasoning_effort"],
            json!("low"),
            "promotion takes the merged value, which the request won"
        );
    }

    #[test]
    fn an_explicit_top_level_effort_is_not_overwritten() {
        let v = variant(json!({"reasoning_effort": "high"}));
        let out = v
            .apply(&json!({"messages": [], "reasoning_effort": "low"}))
            .into_owned();
        assert_eq!(out["reasoning_effort"], json!("low"));
        assert_eq!(
            out["chat_template_kwargs"]["reasoning_effort"],
            json!("high"),
            "the engine leaves the merged dict alone; only the field is guarded"
        );
    }

    #[test]
    fn a_default_that_is_not_an_effort_does_not_invent_one() {
        let v = variant(json!({"enable_thinking": false}));
        let out = v.apply(&json!({"messages": []})).into_owned();
        assert!(out.get("reasoning_effort").is_none());
        assert_eq!(out["chat_template_kwargs"]["enable_thinking"], json!(false));
    }

    #[test]
    fn ids_are_stable_across_key_order_and_distinct_across_values() {
        let a = variant(json!({"reasoning_effort": "high", "enable_thinking": true}));
        let b = variant(json!({"enable_thinking": true, "reasoning_effort": "high"}));
        let c = variant(json!({"reasoning_effort": "low", "enable_thinking": true}));
        assert_eq!(
            a.id(),
            b.id(),
            "serialisation order is the engine's business"
        );
        assert_ne!(a.id(), c.id());
        assert_ne!(a.id(), 0, "only the empty variant is 0");
    }

    #[test]
    fn a_missing_or_non_object_report_is_the_empty_variant() {
        assert!(RenderVariant::from_default_chat_template_kwargs(None).is_empty());
        assert!(RenderVariant::from_default_chat_template_kwargs(Some(&Value::Null)).is_empty());
        assert!(
            RenderVariant::from_default_chat_template_kwargs(Some(&json!("high"))).is_empty(),
            "sglang rejects a non-dict itself; we do not guess at one"
        );
    }
}

/// Which variant each worker renders with.
///
/// Two tiers on purpose. `fleet` comes from `--kv-default-chat-template-kwargs`
/// and applies to everything; `per_worker` is what the worker itself reported
/// from `/get_server_info` and wins where we have it. The fallback direction
/// matters: a worker we have not been able to ask keeps rendering the way the
/// router rendered for it before any of this existed, so nothing regresses on
/// an engine without that endpoint.
///
/// Populated by the render probe, which already visits every worker once at
/// registration and already owns a claim/record/retain lifecycle for exactly
/// this shape of per-worker fact.
pub struct VariantRegistry {
    fleet: Arc<RenderVariant>,
    per_worker: RwLock<HashMap<String, Arc<RenderVariant>>>,
    /// Whether to ask workers at all. Off pins the fleet to `fleet`.
    enabled: bool,
}

impl Default for VariantRegistry {
    fn default() -> Self {
        VariantRegistry::new(RenderVariant::default(), true)
    }
}

impl VariantRegistry {
    pub fn new(fleet: RenderVariant, enabled: bool) -> Self {
        VariantRegistry {
            fleet: Arc::new(fleet),
            per_worker: RwLock::new(HashMap::new()),
            enabled,
        }
    }

    pub fn per_worker_enabled(&self) -> bool {
        self.enabled
    }

    pub fn fleet(&self) -> Arc<RenderVariant> {
        Arc::clone(&self.fleet)
    }

    /// This worker's variant: its own if we know it, the fleet default if not.
    pub fn for_worker(&self, worker_id: &str) -> Arc<RenderVariant> {
        if self.enabled {
            if let Some(v) = self
                .per_worker
                .read()
                .expect("variant registry rwlock poisoned")
                .get(worker_id)
            {
                return Arc::clone(v);
            }
        }
        Arc::clone(&self.fleet)
    }

    /// Record what a worker reported. Logs only when it changes the answer,
    /// because the interesting event is a fleet that is not uniform -- a fleet
    /// that is uniform should be silent.
    pub fn record(&self, worker_id: &str, variant: RenderVariant) {
        if variant != *self.fleet {
            tracing::warn!(
                worker = worker_id,
                variant = %variant.label(),
                router_default = %self.fleet.label(),
                "kv-aware: this worker renders with server-side template defaults that differ \
                 from the router's. Requests for it are now hashed its way; if \
                 --kv-per-worker-template-kwargs is off, or this router is older than the \
                 worker, every lookup for it misses instead"
            );
        }
        self.per_worker
            .write()
            .expect("variant registry rwlock poisoned")
            .insert(worker_id.to_string(), Arc::new(variant));
    }

    /// Forget workers that left the fleet.
    pub fn retain<F: Fn(&str) -> bool>(&self, alive: F) {
        self.per_worker
            .write()
            .expect("variant registry rwlock poisoned")
            .retain(|id, _| alive(id));
    }

    /// `(worker_id, label)` for logging the distribution. Sorted, so a diff
    /// between two snapshots is readable.
    pub fn snapshot(&self) -> Vec<(String, String)> {
        let mut out: Vec<_> = self
            .per_worker
            .read()
            .expect("variant registry rwlock poisoned")
            .iter()
            .map(|(id, v)| (id.clone(), v.label()))
            .collect();
        out.sort();
        out
    }

    /// How many distinct variants the fleet is running, per model. This is the
    /// number worth watching before trusting any of this: a fleet that reports
    /// 1 everywhere never needed the per-worker tier, and one that reports 2+
    /// is a fleet where a single fleet-wide flag could not have been right.
    pub fn distinct(&self) -> usize {
        let g = self
            .per_worker
            .read()
            .expect("variant registry rwlock poisoned");
        let mut ids: Vec<u64> = g.values().map(|v| v.id()).collect();
        ids.push(self.fleet.id());
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;
    use serde_json::json;

    fn variant(v: Value) -> RenderVariant {
        RenderVariant::from_default_chat_template_kwargs(Some(&v))
    }

    #[test]
    fn an_unasked_worker_gets_the_fleet_default() {
        let reg = VariantRegistry::new(variant(json!({"reasoning_effort": "high"})), true);
        assert_eq!(
            reg.for_worker("never-probed").label(),
            "reasoning_effort=\"high\""
        );
    }

    #[test]
    fn a_workers_own_report_wins_over_the_flag() {
        // The flag is a guess about the fleet; /get_server_info is the fleet.
        let reg = VariantRegistry::new(variant(json!({"reasoning_effort": "high"})), true);
        reg.record("w1", variant(json!({"reasoning_effort": "low"})));
        assert_eq!(reg.for_worker("w1").label(), "reasoning_effort=\"low\"");
        assert_eq!(reg.for_worker("w2").label(), "reasoning_effort=\"high\"");
    }

    #[test]
    fn the_kill_switch_pins_everything_to_the_flag() {
        let reg = VariantRegistry::new(variant(json!({"reasoning_effort": "high"})), false);
        reg.record("w1", variant(json!({"reasoning_effort": "low"})));
        assert_eq!(
            reg.for_worker("w1").label(),
            "reasoning_effort=\"high\"",
            "off means off, even for a worker that answered"
        );
    }

    #[test]
    fn a_departed_worker_is_forgotten() {
        let reg = VariantRegistry::default();
        reg.record("w1", variant(json!({"a": 1})));
        reg.record("w2", variant(json!({"b": 2})));
        reg.retain(|id| id == "w2");
        assert_eq!(reg.snapshot().len(), 1);
        assert_eq!(reg.for_worker("w1").id(), 0, "back to the fleet default");
    }

    #[test]
    fn distinct_counts_the_fleet_default_too() {
        let reg = VariantRegistry::default();
        assert_eq!(reg.distinct(), 1, "empty fleet default, nothing reported");
        reg.record("w1", RenderVariant::default());
        assert_eq!(reg.distinct(), 1, "a worker agreeing adds nothing");
        reg.record("w2", variant(json!({"reasoning_effort": "high"})));
        assert_eq!(
            reg.distinct(),
            2,
            "this is the fleet a single flag cannot serve"
        );
    }
}
