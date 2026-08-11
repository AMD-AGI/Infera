///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! KV-event ingestion over NATS.
//!
//! The Rust half of `infera.router.kv_event.nats_client`. Same router-side
//! bookkeeping as the ZMQ path -- decoding and view application are shared --
//! but one subscription to `infera.kv.events.>` replaces a SUB socket per
//! worker, and the worker id and DP rank are recovered from the subject.
//!
//! Two sources feed the views, and which one is authoritative matters:
//!
//! * **The JetStream stream** is the ordered, authoritative history. It is
//!   replayed from the beginning, not from now: a `BlockStored` names its
//!   `parent_block_hash`, and an event whose parent was never seen is dropped,
//!   so a router that joined after a worker had already cached prefixes would
//!   build an empty view from live deltas alone.
//! * **The KV bucket** is a cold-start shortcut holding each worker's current
//!   view. It only ever seeds a rank that has no incremental view yet, and an
//!   empty snapshot is never applied -- a relay that desynced can publish one,
//!   and applying it would wipe a good view and collapse cache hits to zero.

use std::sync::Arc;

use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use futures::StreamExt;

use crate::kv_event::KvEventClient;

pub const KV_EVENTS_SUBJECT_PREFIX: &str = "infera.kv.events";
pub const KV_EVENTS_STREAM: &str = "INFERA_KV_EVENTS";
pub const KV_VIEW_BUCKET: &str = "infera_kv_view";

/// `infera.kv.events.<token>.<rank>` -> (worker_id, rank).
pub fn parse_kv_subject(subject: &str) -> Option<(String, i64)> {
    let rest = subject
        .strip_prefix(KV_EVENTS_SUBJECT_PREFIX)?
        .strip_prefix('.')?;
    parse_token_rank(rest)
}

/// KV bucket key `<token>.<rank>` -> (worker_id, rank).
pub fn parse_kv_key(key: &str) -> Option<(String, i64)> {
    parse_token_rank(key)
}

fn parse_token_rank(s: &str) -> Option<(String, i64)> {
    let (token, rank) = s.rsplit_once('.')?;
    if token.is_empty() {
        return None;
    }
    let decoded = URL_SAFE_NO_PAD.decode(token).ok()?;
    Some((String::from_utf8(decoded).ok()?, rank.parse().ok()?))
}

/// Subscribe and keep feeding the client's views. Runs until the process ends.
pub async fn run(client: Arc<KvEventClient>, url: Option<&str>) -> Result<()> {
    let url = crate::nats_request::resolve_url(url);
    let nc = async_nats::ConnectOptions::new()
        .name("infera-router-kv")
        .retry_on_initial_connect()
        .connect(&url)
        .await
        .with_context(|| format!("connecting to NATS at {url}"))?;
    let js = async_nats::jetstream::new(nc);
    ensure_event_stream(&js).await?;

    // The bucket bootstrap runs alongside the replay rather than before it:
    // waiting would delay live deltas, and the seeding rule (only where there
    // is no incremental view) makes the order harmless.
    tokio::spawn({
        let client = client.clone();
        let js = js.clone();
        async move {
            if let Err(e) = watch_bucket(client, js).await {
                tracing::warn!("kv events (nats): bucket bootstrap ended: {e}");
            }
        }
    });

    tracing::info!("kv events (nats): subscribed {KV_EVENTS_SUBJECT_PREFIX}.> at {url}");
    consume_events(client, js).await
}

async fn ensure_event_stream(js: &async_nats::jetstream::Context) -> Result<()> {
    use async_nats::jetstream::stream::{Config, DiscardPolicy, RetentionPolicy, StorageType};
    let cfg = Config {
        name: KV_EVENTS_STREAM.to_string(),
        subjects: vec![format!("{KV_EVENTS_SUBJECT_PREFIX}.>")],
        retention: RetentionPolicy::Limits,
        storage: StorageType::File,
        discard: DiscardPolicy::Old,
        // Bounded so history cannot grow without limit; oldest goes first.
        max_bytes: 256 * 1024 * 1024,
        max_messages: 1_000_000,
        ..Default::default()
    };
    // The relay may have created it first; either way the config converges.
    if js.create_stream(cfg.clone()).await.is_err() {
        js.update_stream(&cfg)
            .await
            .context("creating or updating the KV event stream")?;
    }
    Ok(())
}

async fn consume_events(
    client: Arc<KvEventClient>,
    js: async_nats::jetstream::Context,
) -> Result<()> {
    use async_nats::jetstream::consumer::{pull, AckPolicy, DeliverPolicy};

    let stream = js
        .get_stream(KV_EVENTS_STREAM)
        .await
        .context("opening the KV event stream")?;
    // Ephemeral and per-process, so every router replica gets its own full
    // replay. DeliverPolicy::All rather than New is a correctness requirement,
    // not a preference -- see the module docs. Nothing is acked because the
    // stream is Limits-retained: acks would not free anything, and every
    // consumer is meant to see the whole history.
    let consumer = stream
        .create_consumer(pull::Config {
            deliver_policy: DeliverPolicy::All,
            ack_policy: AckPolicy::None,
            ..Default::default()
        })
        .await
        .context("creating the KV event consumer")?;

    let mut messages = consumer.messages().await.context("consuming KV events")?;
    while let Some(msg) = messages.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("kv events (nats): stream error: {e}");
                continue;
            }
        };
        let (worker_id, rank) = match parse_kv_subject(&msg.subject) {
            Some(p) => p,
            None => continue,
        };
        // Events for a worker the router does not track yet (or dropped) are
        // discarded; a late registration reconciles through the bucket.
        client.apply_encoded_batch(&worker_id, rank, &msg.payload);
    }
    Ok(())
}

/// A bucket value is the msgpack list of chained hashes the relay wrote.
fn decode_view(bytes: &[u8]) -> Option<Vec<u64>> {
    let mut cur = std::io::Cursor::new(bytes);
    let value = rmpv::decode::read_value(&mut cur).ok()?;
    let items = value.as_array()?;
    // Hashes are u64 but msgpack may encode them as signed; reinterpret rather
    // than dropping them, since the ZMQ path treats the two the same way.
    Some(
        items
            .iter()
            .filter_map(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)))
            .collect(),
    )
}

/// Watch the KV bucket: initial values bootstrap a cold start, later PUTs heal
/// drift as the relay rewrites views.
async fn watch_bucket(
    client: Arc<KvEventClient>,
    js: async_nats::jetstream::Context,
) -> Result<()> {
    let store = match js.get_key_value(KV_VIEW_BUCKET).await {
        Ok(s) => s,
        Err(_) => js
            .create_key_value(async_nats::jetstream::kv::Config {
                bucket: KV_VIEW_BUCKET.to_string(),
                ..Default::default()
            })
            .await
            .context("opening the KV view bucket")?,
    };
    // `watch_all` delivers only subsequent updates, which would leave a
    // cold-starting router waiting for the next time a relay happens to rewrite
    // a view -- the bootstrap would never arrive. Asking for history delivers
    // each key's current value first, then the updates.
    let mut watcher = store
        .watch_many_with_history([">"])
        .await
        .context("watching the KV view bucket")?;
    while let Some(entry) = watcher.next().await {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("kv events (nats): bucket watch error: {e}");
                continue;
            }
        };
        let (worker_id, rank) = match parse_kv_key(&entry.key) {
            Some(p) => p,
            None => continue,
        };
        if matches!(
            entry.operation,
            async_nats::jetstream::kv::Operation::Delete
                | async_nats::jetstream::kv::Operation::Purge
        ) {
            client.drop_rank_view(&worker_id, rank);
            continue;
        }
        match decode_view(&entry.value) {
            Some(snapshot) => client.seed_rank_view(&worker_id, rank, snapshot),
            None => continue,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(s: &str) -> String {
        URL_SAFE_NO_PAD.encode(s.as_bytes())
    }

    #[test]
    fn subjects_round_trip_the_python_encoding() {
        // The relay publishes these; disagreeing on the encoding means the
        // router silently indexes nothing while looking healthy.
        let subject = format!("{KV_EVENTS_SUBJECT_PREFIX}.{}.0", token("10.0.0.1:8080"));
        assert_eq!(
            parse_kv_subject(&subject),
            Some(("10.0.0.1:8080".to_string(), 0))
        );
        let subject = format!("{KV_EVENTS_SUBJECT_PREFIX}.{}.3", token("host:9000"));
        assert_eq!(
            parse_kv_subject(&subject),
            Some(("host:9000".to_string(), 3))
        );
    }

    #[test]
    fn the_names_are_the_ones_the_python_side_publishes_to() {
        // Literals produced by infera.kv.nats_bus. The two implementations
        // only meet on a live broker: subscribing to the wrong subject, or
        // opening a differently named stream, looks exactly like a fleet with
        // no cache activity.
        assert_eq!(KV_EVENTS_SUBJECT_PREFIX, "infera.kv.events");
        assert_eq!(KV_EVENTS_STREAM, "INFERA_KV_EVENTS");
        assert_eq!(KV_VIEW_BUCKET, "infera_kv_view");
        assert_eq!(
            parse_kv_subject("infera.kv.events.MTAuMC4wLjE6ODA4MA.0"),
            Some(("10.0.0.1:8080".to_string(), 0))
        );
        assert_eq!(
            parse_kv_subject("infera.kv.events.aG9zdDo5MDAw.3"),
            Some(("host:9000".to_string(), 3))
        );
        assert_eq!(
            parse_kv_key("MTAuMC4wLjE6ODA4MA.2"),
            Some(("10.0.0.1:8080".to_string(), 2))
        );
    }

    #[test]
    fn a_foreign_or_malformed_subject_is_ignored() {
        assert_eq!(parse_kv_subject("infera.req.abc"), None);
        assert_eq!(parse_kv_subject("infera.kv.events.notbase64!.0"), None);
        assert_eq!(
            parse_kv_subject(&format!("{KV_EVENTS_SUBJECT_PREFIX}.{}.x", token("w"))),
            None
        );
    }

    fn tracked_client(block_size: i64) -> Arc<KvEventClient> {
        let c = Arc::new(KvEventClient::nats_fed());
        let w: crate::pool::Worker = serde_json::from_value(serde_json::json!({
            "worker_id": "w1", "url": "http://w1", "kv_block_size": block_size,
        }))
        .unwrap();
        c.on_worker_added(&w);
        c
    }

    #[test]
    fn an_empty_bucket_snapshot_never_wipes_a_view() {
        // A relay that desynced can publish an empty view. Applying it would
        // clear a view built from the ordered stream, and cache hits would
        // collapse to zero with nothing in the logs to explain it.
        let c = tracked_client(16);
        c.seed_rank_view("w1", 0, vec![1, 2, 3]);
        assert_eq!(c.total_blocks("w1"), 3);

        c.seed_rank_view("w1", 0, vec![]);
        assert_eq!(c.total_blocks("w1"), 3, "an empty snapshot must be ignored");
    }

    #[test]
    fn the_bucket_does_not_overwrite_an_existing_view() {
        // The stream is authoritative and ordered; the bucket is only a
        // cold-start shortcut, so it seeds and never corrects.
        let c = tracked_client(16);
        c.seed_rank_view("w1", 0, vec![1, 2, 3]);
        c.seed_rank_view("w1", 0, vec![9]);
        assert_eq!(c.total_blocks("w1"), 3);
    }

    #[test]
    fn an_untracked_worker_is_not_seeded() {
        // Its block size is unknown, so a view would be built against a
        // geometry we cannot verify.
        let c = Arc::new(KvEventClient::nats_fed());
        c.seed_rank_view("ghost", 0, vec![1, 2]);
        assert_eq!(c.total_blocks("ghost"), 0);
    }

    #[test]
    fn a_worker_without_a_block_size_is_not_tracked() {
        // Defaulting to 1 would produce a view that never matches, which reads
        // as "no cache hits" rather than as the misconfiguration it is.
        let c = Arc::new(KvEventClient::nats_fed());
        let w: crate::pool::Worker =
            serde_json::from_value(serde_json::json!({"worker_id": "w1", "url": "http://w1"}))
                .unwrap();
        c.on_worker_added(&w);
        c.seed_rank_view("w1", 0, vec![1, 2]);
        assert_eq!(c.total_blocks("w1"), 0);
    }

    #[test]
    fn a_deleted_bucket_key_drops_that_rank() {
        let c = tracked_client(16);
        c.seed_rank_view("w1", 0, vec![1, 2, 3]);
        c.seed_rank_view("w1", 1, vec![4]);
        c.drop_rank_view("w1", 0);
        assert_eq!(c.total_blocks("w1"), 1, "only the named rank goes");
    }

    #[test]
    fn a_bucket_value_decodes_as_the_relay_wrote_it() {
        let mut buf = Vec::new();
        rmpv::encode::write_value(
            &mut buf,
            &rmpv::Value::Array(vec![
                rmpv::Value::from(1u64),
                rmpv::Value::from(u64::MAX),
                // msgpack may render a hash as signed; it must not be dropped.
                rmpv::Value::from(-1i64),
            ]),
        )
        .unwrap();
        assert_eq!(decode_view(&buf), Some(vec![1, u64::MAX, u64::MAX]));
    }

    #[test]
    fn bucket_keys_parse_the_same_way() {
        assert_eq!(
            parse_kv_key(&format!("{}.2", token("10.0.0.1:8080"))),
            Some(("10.0.0.1:8080".to_string(), 2))
        );
    }
}
