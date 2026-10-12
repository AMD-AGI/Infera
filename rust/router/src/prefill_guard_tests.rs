use super::*;
use crate::config::PrefillGuardRelease;
use crate::policy::{Pick, Policy, RoundRobin};
use crate::pool::{Snapshot, Worker};
use futures::StreamExt;
use std::sync::Mutex;
use std::time::Instant;
use tokio::sync::Notify;

#[derive(Default)]
struct Counts {
    started: Mutex<Vec<String>>,
    finished: Mutex<Vec<String>>,
    changed: Notify,
}
impl Policy for Counts {
    fn pick(&self, _: &[Arc<Worker>], _: &Value, _: Role) -> Pick {
        unreachable!()
    }
    fn on_request_started(&self, key: &str, _: &[u64]) {
        self.started.lock().unwrap().push(key.into());
    }
    fn on_request_finished(&self, key: &str, _: &[u64]) {
        self.finished.lock().unwrap().push(key.into());
        self.changed.notify_one();
    }
}
fn guard(counts: Arc<Counts>) -> ActiveGuard {
    ActiveGuard::start(counts, vec![("p".into(), vec![1]), ("d".into(), vec![2])])
}
async fn finished(counts: &Counts, key: &str) {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let notified = counts.changed.notified();
            if counts.finished.lock().unwrap().iter().any(|x| x == key) {
                return;
            }
            notified.await;
        }
    })
    .await
    .unwrap();
}
async fn server(app: axum::Router) -> (String, JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), task)
}
fn target(url: String) -> RouteTarget {
    RouteTarget {
        worker: Arc::new(
            serde_json::from_value(serde_json::json!({
                "worker_id":"worker", "url":url, "model_name":"model"
            }))
            .unwrap(),
        ),
        dp_rank: None,
    }
}
fn state(mode: PrefillGuardRelease) -> AppState {
    AppState {
        pool: Arc::new(arc_swap::ArcSwap::from_pointee(Snapshot::build(vec![]))),
        policy: Arc::new(RoundRobin::new()),
        http: reqwest::Client::new(),
        started: Instant::now(),
        retries: 0,
        breaker: Arc::new(CircuitBreaker::default()),
        nats: None,
        pd_prefill_drain_timeout: Duration::from_secs(1),
        pd_prefill_guard_release: mode,
        stream_stall_warn: Default::default(),
    }
}

#[test]
fn splitting_transfers_the_entry_without_duplicate_accounting() {
    let counts = Arc::new(Counts::default());
    let mut d = guard(counts.clone());
    let p = d.take_first().unwrap();
    assert_eq!(*counts.started.lock().unwrap(), ["p", "d"]);
    assert!(counts.finished.lock().unwrap().is_empty());
    drop(p);
    assert_eq!(*counts.finished.lock().unwrap(), ["p"]);
    drop(d);
    assert_eq!(*counts.finished.lock().unwrap(), ["p", "d"]);
}

#[tokio::test]
async fn detached_prefill_holds_load_until_body_eof_on_success_and_error() {
    for status in [StatusCode::OK, StatusCode::INTERNAL_SERVER_ERROR] {
        let opened = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let (o, r) = (opened.clone(), release.clone());
        let app = axum::Router::new().route(
            "/",
            axum::routing::post(move || {
                let (o, r) = (o.clone(), r.clone());
                async move {
                    let body = futures::stream::once(async move {
                        o.notify_one();
                        Ok::<_, std::io::Error>(Bytes::from_static(b"first"))
                    })
                    .chain(futures::stream::once(async move {
                        r.notified().await;
                        Ok::<_, std::io::Error>(Bytes::from_static(b"last"))
                    }));
                    Response::builder()
                        .status(status)
                        .body(Body::from_stream(body))
                        .unwrap()
                }
            }),
        );
        let (url, server) = server(app).await;
        let counts = Arc::new(Counts::default());
        let mut d = guard(counts.clone());
        let p = d.take_first();
        let drain = spawn_prefill_drain(
            reqwest::Client::new(),
            Arc::new(CircuitBreaker::default()),
            "p".into(),
            url,
            Map::new(),
            None,
            p,
        );
        tokio::time::timeout(Duration::from_secs(3), opened.notified())
            .await
            .unwrap();
        assert!(counts.finished.lock().unwrap().is_empty());
        drop(d);
        assert_eq!(*counts.finished.lock().unwrap(), ["d"]);
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(3), drain)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*counts.finished.lock().unwrap(), ["d", "p"]);
        server.abort();
    }
}

#[tokio::test]
async fn aborting_a_stalled_prefill_drain_releases_its_guard() {
    let counts = Arc::new(Counts::default());
    let mut d = guard(counts.clone());
    let p = d.take_first();
    let task = spawn_prefill_drain(
        reqwest::Client::new(),
        Arc::new(CircuitBreaker::default()),
        "p".into(),
        "http://127.0.0.1:1".into(),
        Map::new(),
        None,
        p,
    );
    task.abort();
    let _ = task.await;
    assert_eq!(*counts.finished.lock().unwrap(), ["p"]);
    drop(d);
    assert_eq!(*counts.finished.lock().unwrap(), ["p", "d"]);
}

#[tokio::test]
async fn unary_completion_mode_releases_p_while_decode_headers_are_pending() {
    for mode in [PrefillGuardRelease::Decode, PrefillGuardRelease::Completion] {
        let p_done = Arc::new(Notify::new());
        let d_opened = Arc::new(Notify::new());
        let d_release = Arc::new(Notify::new());
        let done = p_done.clone();
        let (p_url, p_server) = server(axum::Router::new().route(
            "/",
            axum::routing::post(move || {
                let done = done.clone();
                async move {
                    done.notify_one();
                    "{}"
                }
            }),
        ))
        .await;
        let (opened, release) = (d_opened.clone(), d_release.clone());
        let (d_url, d_server) = server(axum::Router::new().route(
            "/",
            axum::routing::post(move || {
                let (opened, release) = (opened.clone(), release.clone());
                async move {
                    opened.notify_one();
                    release.notified().await;
                    "{}"
                }
            }),
        ))
        .await;
        let counts = Arc::new(Counts::default());
        let g = guard(counts.clone());
        let run = tokio::spawn(async move {
            unary_dual(
                &state(mode),
                &target(p_url.clone()),
                &target(d_url.clone()),
                p_url,
                d_url,
                Map::new(),
                Map::new(),
                g,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(3), p_done.notified())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(3), d_opened.notified())
            .await
            .unwrap();
        if mode == PrefillGuardRelease::Completion {
            finished(&counts, "p").await;
            assert_eq!(*counts.finished.lock().unwrap(), ["p"]);
        } else {
            assert!(counts.finished.lock().unwrap().is_empty());
        }
        d_release.notify_one();
        assert_eq!(run.await.unwrap().status(), StatusCode::OK);
        assert_eq!(*counts.finished.lock().unwrap(), ["p", "d"]);
        p_server.abort();
        d_server.abort();
    }
}

#[tokio::test]
#[ignore = "needs a broker: INFERA_TEST_NATS"]
async fn nats_prefill_holds_load_until_a_terminal_frame() {
    use crate::nats_request::{
        request_subject, HDR_STATUS, HDR_TYPE, TYPE_DATA, TYPE_DONE, TYPE_ERROR,
    };
    let url = std::env::var("INFERA_TEST_NATS").expect("set INFERA_TEST_NATS");
    let nc = async_nats::connect(&url).await.unwrap();
    for terminal in [TYPE_DONE, TYPE_ERROR] {
        let wid = format!("guard-test-{}-{terminal}", std::process::id());
        let mut requests = nc.subscribe(request_subject(&wid)).await.unwrap();
        nc.flush().await.unwrap();
        let client = Arc::new(
            NatsRequestClient::connect(Some(&url), 5.0, 0.0, 0)
                .await
                .unwrap(),
        );
        let counts = Arc::new(Counts::default());
        let mut d = guard(counts.clone());
        let task = spawn_prefill_drain_nats(
            client,
            Arc::new(CircuitBreaker::default()),
            wid,
            b"{}".to_vec(),
            d.take_first(),
        );
        let request = tokio::time::timeout(Duration::from_secs(3), requests.next())
            .await
            .unwrap()
            .unwrap();
        let inbox = request.reply.unwrap();
        let mut headers = async_nats::HeaderMap::new();
        headers.insert(HDR_TYPE, TYPE_DATA);
        nc.publish_with_headers(inbox.clone(), headers, Bytes::from_static(b"partial"))
            .await
            .unwrap();
        nc.flush().await.unwrap();
        assert!(counts.finished.lock().unwrap().is_empty());
        let mut headers = async_nats::HeaderMap::new();
        headers.insert(HDR_TYPE, terminal);
        headers.insert(
            HDR_STATUS,
            if terminal == TYPE_DONE { "200" } else { "500" },
        );
        nc.publish_with_headers(inbox, headers, Bytes::new())
            .await
            .unwrap();
        nc.flush().await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*counts.finished.lock().unwrap(), ["p"]);
        drop(d);
        assert_eq!(*counts.finished.lock().unwrap(), ["p", "d"]);
    }
}
