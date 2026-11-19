//! U24: does reusing pooled connections race the server's keep-alive close? CONCURRENCY tasks each POST to URL, then
//! sleep a random gap around GAP_MS (+-10%), for DURATION_S, through a reqwest client built like the Infera router's
//! upstream client (pool_max_idle_per_host 1024) with the given pool idle timeout. THREADS runtime workers (default 4)
//! and BUSY_MS of CPU work on the runtime after every response (default 0) stand in for a loaded router, which polls
//! a peer's close late. Prints one JSON line: requests, errors, and how many errors were "error sending request" (a
//! write to a connection the server was closing).
//!   u24-race URL POOL_IDLE_MS GAP_MS DURATION_S CONCURRENCY [THREADS] [BUSY_MS]
use rand::Rng;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let threads: usize = a.get(6).map_or(4, |s| s.parse().unwrap());
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .unwrap()
        .block_on(run(a, threads));
}

async fn run(a: Vec<String>, threads: usize) {
    let url = a[1].clone();
    let idle_ms: u64 = a[2].parse().unwrap();
    let gap_ms: f64 = a[3].parse().unwrap();
    let duration = Duration::from_secs(a[4].parse().unwrap());
    let conc: usize = a[5].parse().unwrap();
    let busy = Duration::from_millis(a.get(7).map_or(0, |s| s.parse().unwrap()));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(60))
        .pool_max_idle_per_host(1024)
        .pool_idle_timeout(Duration::from_millis(idle_ms))
        .build()
        .unwrap();
    let (reqs, errs, send_errs) = (
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );
    let end = Instant::now() + duration;
    let mut tasks = Vec::new();
    for _ in 0..conc {
        let (c, u) = (client.clone(), url.clone());
        let (reqs, errs, send_errs) = (reqs.clone(), errs.clone(), send_errs.clone());
        tasks.push(tokio::spawn(async move {
            while Instant::now() < end {
                reqs.fetch_add(1, Ordering::Relaxed);
                match c.post(&u).body("{\"prompt\":\"x\"}").send().await {
                    Ok(r) => {
                        let _ = r.bytes().await;
                    }
                    Err(e) => {
                        errs.fetch_add(1, Ordering::Relaxed);
                        if e.to_string().contains("error sending request") {
                            send_errs.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                let spin_until = Instant::now() + busy;
                while Instant::now() < spin_until {
                    std::hint::spin_loop();
                }
                let jitter: f64 = rand::thread_rng().gen_range(0.9..1.1);
                tokio::time::sleep(Duration::from_secs_f64(gap_ms * jitter / 1000.0)).await;
            }
        }));
    }
    for t in tasks {
        let _ = t.await;
    }
    println!(
        "{{\"pool_idle_ms\":{idle_ms},\"gap_ms\":{gap_ms},\"concurrency\":{conc},\"threads\":{threads},\"busy_ms\":{},\"requests\":{},\"errors\":{},\"error_sending_request\":{}}}",
        busy.as_millis(),
        reqs.load(Ordering::Relaxed),
        errs.load(Ordering::Relaxed),
        send_errs.load(Ordering::Relaxed)
    );
}
