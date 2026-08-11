///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! NATS per-instance request transport (router side).
//!
//! The Rust half of `infera.common.nats_request`. The policy still *selects* a
//! worker exactly as it does for HTTP -- selection is transport-agnostic, so
//! kv-aware routing is unaffected -- and this replaces the `POST {worker.url}`
//! with a publish onto that specific worker's subject plus a reply inbox.
//!
//! The wire protocol is the Python one, byte for byte, so the worker-side
//! `NatsRequestServer` is unchanged and a Python and a Rust router can serve
//! the same fleet through the same broker:
//!
//!   request (router -> `infera.req.<token(worker_id)>`, reply=<inbox>):
//!       JSON {"path": str, "stream": bool, "headers": {..}|null, "body": {..}}
//!
//!   reply (worker -> <inbox>), framed by the `rs-type` header:
//!       data  : payload = raw response bytes (an SSE chunk, or a whole body)
//!       done  : payload = b"", header `rs-status` = HTTP status
//!       error : payload = utf-8 error text
//!
//! Per-instance subjects rather than a shared queue group: the router keeps
//! full control of which worker serves each request, which is the whole point
//! of kv-aware placement.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::Bytes;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use futures::StreamExt;

pub const REQUEST_SUBJECT_PREFIX: &str = "infera.req";
pub const CANCEL_SUBJECT_PREFIX: &str = "infera.cancel";

pub const HDR_TYPE: &str = "rs-type";
pub const HDR_STATUS: &str = "rs-status";
pub const TYPE_DATA: &str = "data";
pub const TYPE_DONE: &str = "done";
pub const TYPE_ERROR: &str = "error";

/// Cancels are best-effort and resent a few times, so a worker that is briefly
/// reconnecting when a client disconnects still hears about it.
const CANCEL_RESEND: usize = 3;
const CANCEL_RESEND_GAP: Duration = Duration::from_secs(1);

/// Reversible, subject-safe encoding of a worker id: `host:port` contains a
/// colon and dots, and dots are NATS subject separators.
pub fn token(worker_id: &str) -> String {
    URL_SAFE_NO_PAD.encode(worker_id.as_bytes())
}

pub fn request_subject(worker_id: &str) -> String {
    format!("{REQUEST_SUBJECT_PREFIX}.{}", token(worker_id))
}

pub fn cancel_subject(worker_id: &str) -> String {
    format!("{CANCEL_SUBJECT_PREFIX}.{}", token(worker_id))
}

/// `NATS_SERVER` mirrors dynamo's convention, so one broker serves both.
pub fn resolve_url(explicit: Option<&str>) -> String {
    explicit
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| std::env::var("NATS_SERVER").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "nats://127.0.0.1:4222".to_string())
}

/// Idempotently create the WorkQueue stream covering `infera.req.>`.
///
/// Memory-backed: requests are ephemeral, so losing them when the broker
/// restarts is acceptable -- the client gets an error and retries, which is
/// better than paying disk for traffic that is worthless a second later.
async fn ensure_request_stream(js: &async_nats::jetstream::Context) -> Result<()> {
    use async_nats::jetstream::stream::{Config, DiscardPolicy, RetentionPolicy, StorageType};
    let cfg = Config {
        name: REQUEST_STREAM.to_string(),
        subjects: vec![format!("{REQUEST_SUBJECT_PREFIX}.>")],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::Memory,
        discard: DiscardPolicy::Old,
        max_messages: 1_000_000,
        ..Default::default()
    };
    // The worker may have created it first; either way the config converges.
    if js.create_stream(cfg.clone()).await.is_err() {
        js.update_stream(&cfg)
            .await
            .context("creating or updating the request stream")?;
    }
    Ok(())
}

/// One frame of a reply stream.
#[derive(Debug)]
pub enum Frame {
    Data(Bytes),
    Done {
        status: u16,
    },
    Error {
        status: Option<u16>,
        message: String,
    },
}

/// JetStream stream backing the request path when admission throttling is on.
/// WorkQueue retention means a request leaves the stream once its worker acks
/// it, so a consumer's pending count is a live backlog gauge rather than a
/// running total.
pub const REQUEST_STREAM: &str = "INFERA_REQUESTS";

/// Reply inbox carried as a header in JetStream mode: a JS-delivered message's
/// `reply` field is the ack subject, not the publisher's inbox.
pub const HDR_INBOX: &str = "rs-inbox";

/// Short enough that a backlog reading is still meaningful, long enough that a
/// burst of requests does not issue a `consumer_info` round-trip each.
const PENDING_CACHE_TTL: Duration = Duration::from_millis(200);

pub struct NatsRequestClient {
    nc: async_nats::Client,
    /// Present only when throttling is on, which also switches publishes to
    /// JetStream so the backlog is measurable at all.
    js: Option<async_nats::jetstream::Context>,
    /// Per-worker in-NATS backlog ceiling. Zero disables the throttle.
    max_pending: usize,
    pending_cache: std::sync::Mutex<HashMap<String, (tokio::time::Instant, u64)>>,
    /// Bounds the wait for the *next* chunk, reset on every one, so a steadily
    /// streaming long generation never trips it -- only a stall does. This is
    /// deliberately not an overall deadline. Zero disables it.
    idle_timeout: Duration,
    /// Hard wall-clock cap on the whole request regardless of token flow, for
    /// runaway generations. Zero (the default) disables it.
    max_duration: Duration,
}

impl NatsRequestClient {
    pub async fn connect(
        url: Option<&str>,
        idle_timeout_s: f64,
        max_duration_s: f64,
        max_pending: usize,
    ) -> Result<Self> {
        let url = resolve_url(url);
        let nc = async_nats::ConnectOptions::new()
            .name("infera-router-req")
            // Retry forever: a broker restart must not take the router down
            // with it, matching max_reconnect_attempts=-1 on the Python side.
            .retry_on_initial_connect()
            .connect(&url)
            .await
            .with_context(|| format!("connecting to NATS at {url}"))?;
        let js = if max_pending > 0 {
            let js = async_nats::jetstream::new(nc.clone());
            ensure_request_stream(&js).await?;
            tracing::info!(
                "NATS request transport -> {url} (JetStream throttle on, \
                 max_pending={max_pending})"
            );
            Some(js)
        } else {
            // Not "connected": retry_on_initial_connect means this returns
            // before a link exists, so claiming one would be a lie whenever the
            // broker is down -- exactly when someone is reading the log.
            tracing::info!("NATS request transport -> {url}");
            None
        };
        Ok(Self {
            nc,
            js,
            max_pending,
            pending_cache: std::sync::Mutex::new(HashMap::new()),
            idle_timeout: Duration::from_secs_f64(idle_timeout_s.max(0.0)),
            max_duration: Duration::from_secs_f64(max_duration_s.max(0.0)),
        })
    }

    /// Whether this worker may be dispatched to.
    ///
    /// Fails open: throttling off, no consumer yet, or a backlog that cannot be
    /// read all admit. Refusing on a reading we could not take would turn a
    /// monitoring gap into an outage.
    pub async fn admit(&self, worker_id: &str) -> bool {
        let js = match &self.js {
            Some(js) if self.max_pending > 0 => js,
            _ => return true,
        };
        if let Some(cached) = self.cached_pending(worker_id) {
            return (cached as usize) < self.max_pending;
        }
        let backlog = match js.get_stream(REQUEST_STREAM).await {
            Ok(stream) => match stream.consumer_info(&token(worker_id)).await {
                // Both halves count: messages not yet delivered, and ones the
                // worker has taken but not acked -- which is where a request
                // being generated right now sits.
                Ok(info) => info.num_pending + info.num_ack_pending as u64,
                Err(_) => return true,
            },
            Err(_) => return true,
        };
        if let Ok(mut c) = self.pending_cache.lock() {
            c.insert(
                worker_id.to_string(),
                (tokio::time::Instant::now(), backlog),
            );
        }
        (backlog as usize) < self.max_pending
    }

    fn cached_pending(&self, worker_id: &str) -> Option<u64> {
        let cache = self.pending_cache.lock().ok()?;
        let (at, backlog) = cache.get(worker_id)?;
        (at.elapsed() < PENDING_CACHE_TTL).then_some(*backlog)
    }

    /// Publish a request and return the reply inbox subscription to read frames
    /// from. Split from `next_frame` so the caller owns the loop and can drop it
    /// on client disconnect, which is what triggers the cancel.
    pub async fn dispatch(&self, worker_id: &str, payload: &[u8]) -> Result<ReplyStream> {
        let inbox = self.nc.new_inbox();
        let sub = self
            .nc
            .subscribe(inbox.clone())
            .await
            .context("subscribing to the reply inbox")?;
        match &self.js {
            // JetStream delivery replaces `reply` with the ack subject, so the
            // inbox has to travel as a header instead. The reply itself still
            // comes back over the core subscription above.
            Some(js) => {
                let mut headers = async_nats::HeaderMap::new();
                headers.insert(HDR_INBOX, inbox.as_str());
                js.publish_with_headers(
                    request_subject(worker_id),
                    headers,
                    payload.to_vec().into(),
                )
                .await
                .context("publishing the request to JetStream")?;
            }
            None => {
                self.nc
                    .publish_with_reply(
                        request_subject(worker_id),
                        inbox.clone(),
                        payload.to_vec().into(),
                    )
                    .await
                    .context("publishing the request")?;
            }
        }
        // Without this the publish can sit in the client's write buffer while
        // the caller is already waiting on the reply.
        self.nc.flush().await.context("flushing the request")?;
        Ok(ReplyStream {
            sub,
            inbox,
            worker_id: worker_id.to_string(),
            nc: self.nc.clone(),
            idle_timeout: self.idle_timeout,
            deadline: if self.max_duration.is_zero() {
                None
            } else {
                Some(tokio::time::Instant::now() + self.max_duration)
            },
            max_duration: self.max_duration,
            finished: false,
        })
    }
}

/// The reply side of one request. Dropping it before the stream finished tells
/// the worker to abort, so it stops generating tokens nobody will read.
pub struct ReplyStream {
    sub: async_nats::Subscriber,
    inbox: String,
    worker_id: String,
    nc: async_nats::Client,
    idle_timeout: Duration,
    deadline: Option<tokio::time::Instant>,
    max_duration: Duration,
    finished: bool,
}

impl ReplyStream {
    /// Next frame, or a synthesised error frame when a deadline expires.
    /// Returns `None` only if the subscription ends without a terminal frame.
    pub async fn next(&mut self) -> Option<Frame> {
        if self.finished {
            return None;
        }
        // Budget is the smaller of the idle gap and whatever is left of the
        // total, so whichever deadline is nearer fires first.
        let idle = (!self.idle_timeout.is_zero()).then_some(self.idle_timeout);
        let total_left = self
            .deadline
            .map(|d| d.saturating_duration_since(tokio::time::Instant::now()));
        if let Some(left) = total_left {
            if left.is_zero() {
                return Some(self.expired_total());
            }
        }
        let budget = match (idle, total_left) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        let msg = match budget {
            Some(b) => match tokio::time::timeout(b, self.sub.next()).await {
                Ok(m) => m,
                Err(_) => {
                    self.finished = true;
                    return Some(
                        if self
                            .deadline
                            .is_some_and(|d| tokio::time::Instant::now() >= d)
                        {
                            self.expired_total_inner()
                        } else {
                            Frame::Error {
                                status: Some(504),
                                message: format!(
                                    "nats request stalled: no reply within {:.0}s (idle timeout)",
                                    self.idle_timeout.as_secs_f64()
                                ),
                            }
                        },
                    );
                }
            },
            None => self.sub.next().await,
        };

        let msg = match msg {
            Some(m) => m,
            None => {
                self.finished = true;
                return None;
            }
        };

        let hdrs = msg.headers.as_ref();
        let rtype = hdrs
            .and_then(|h| h.get(HDR_TYPE))
            .map(|v| v.as_str())
            .unwrap_or(TYPE_DATA);
        match rtype {
            TYPE_DONE => {
                self.finished = true;
                let status = hdrs
                    .and_then(|h| h.get(HDR_STATUS))
                    .and_then(|v| v.as_str().parse::<u16>().ok())
                    .unwrap_or(200);
                Some(Frame::Done { status })
            }
            TYPE_ERROR => {
                self.finished = true;
                Some(Frame::Error {
                    status: None,
                    message: String::from_utf8_lossy(&msg.payload).into_owned(),
                })
            }
            _ => Some(Frame::Data(msg.payload)),
        }
    }

    fn expired_total(&mut self) -> Frame {
        self.finished = true;
        self.expired_total_inner()
    }

    fn expired_total_inner(&self) -> Frame {
        Frame::Error {
            status: Some(504),
            message: format!(
                "nats request exceeded total timeout {:.0}s",
                self.max_duration.as_secs_f64()
            ),
        }
    }
}

impl Drop for ReplyStream {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Timed out, or the client hung up. Either way the worker is still
        // generating into an inbox nobody reads; tell it to stop. Keyed by the
        // inbox, which is unique per request.
        let (nc, subject, inbox) = (
            self.nc.clone(),
            cancel_subject(&self.worker_id),
            self.inbox.clone(),
        );
        tokio::spawn(async move {
            for i in 0..CANCEL_RESEND {
                let _ = nc.publish(subject.clone(), inbox.clone().into()).await;
                let _ = nc.flush().await;
                if i + 1 < CANCEL_RESEND {
                    tokio::time::sleep(CANCEL_RESEND_GAP).await;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subjects_match_the_python_encoding() {
        // Both routers have to derive the same subject from a worker id or the
        // worker is subscribed somewhere the router never publishes. Values
        // taken from the Python `_token` (base64url, padding stripped).
        assert_eq!(token("10.0.0.1:8080"), "MTAuMC4wLjE6ODA4MA");
        assert_eq!(
            request_subject("10.0.0.1:8080"),
            "infera.req.MTAuMC4wLjE6ODA4MA"
        );
        assert_eq!(
            cancel_subject("10.0.0.1:8080"),
            "infera.cancel.MTAuMC4wLjE6ODA4MA"
        );
    }

    #[test]
    fn tokens_match_python_for_the_awkward_worker_ids() {
        // Generated by the Python `_token` and pinned here, because the two
        // implementations only ever meet on a live broker: if they disagree the
        // worker is subscribed to a subject the router never publishes to, and
        // every request simply times out with nothing in either log to say why.
        for (id, want) in [
            ("10.0.0.1:8080", "MTAuMC4wLjE6ODA4MA"),
            ("host.example.com:8000", "aG9zdC5leGFtcGxlLmNvbTo4MDAw"),
            ("[::1]:8080", "Wzo6MV06ODA4MA"),
        ] {
            assert_eq!(token(id), want, "{id}");
        }
    }

    #[test]
    fn a_token_never_contains_a_subject_separator() {
        // base64url uses - and _ where standard base64 uses + and /, so a dot
        // can never appear and split the subject.
        for id in ["a:1", "host.example.com:8000", "[::1]:8080"] {
            assert!(!token(id).contains('.'), "{id}");
        }
    }

    #[test]
    fn the_url_falls_back_the_way_python_does() {
        std::env::remove_var("NATS_SERVER");
        assert_eq!(resolve_url(None), "nats://127.0.0.1:4222");
        assert_eq!(resolve_url(Some("nats://a:1")), "nats://a:1");
        std::env::set_var("NATS_SERVER", "nats://env:4222");
        assert_eq!(resolve_url(None), "nats://env:4222");
        // Explicit still wins over the environment.
        assert_eq!(resolve_url(Some("nats://a:1")), "nats://a:1");
        std::env::remove_var("NATS_SERVER");
    }
}
