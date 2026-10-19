// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Router-local in-flight input tokens, not physical allocator occupancy.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    #[default]
    Off,
    Shadow,
    On,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Demand {
    pub tokens: u128,
    pub requests: usize,
}
pub(crate) type Ledger = Arc<Mutex<HashMap<String, Demand>>>;

/// Text bytes per token assumed when a request's tokens are unavailable.
const ESTIMATED_BYTES_PER_TOKEN: usize = 4;

/// Input tokens estimated from a body's string bytes. Inline binary payloads
/// (`data:` URLs and `data` fields, such as base64 images or audio) are
/// skipped: their size says nothing about their tokens.
pub(crate) fn estimate_tokens(body: &Value) -> u64 {
    fn text_bytes(value: &Value) -> usize {
        match value {
            Value::String(s) if s.starts_with("data:") => 0,
            Value::String(s) => s.len(),
            Value::Array(items) => items.iter().map(text_bytes).sum(),
            Value::Object(fields) => fields
                .iter()
                .filter(|(key, _)| key.as_str() != "data")
                .map(|(_, value)| text_bytes(value))
                .sum(),
            _ => 0,
        }
    }
    text_bytes(body).div_ceil(ESTIMATED_BYTES_PER_TOKEN) as u64
}

/// Booked atomically with selection and released even if the pick is abandoned.
#[derive(Debug)]
pub struct Reservation {
    ledger: Ledger,
    key: String,
    tokens: u64,
}

impl Reservation {
    pub(crate) fn book(
        ledger: &Ledger,
        state: &mut HashMap<String, Demand>,
        key: String,
        tokens: u64,
    ) -> Self {
        let entry = state.entry(key.clone()).or_default();
        entry.tokens += u128::from(tokens);
        entry.requests += 1;
        crate::metrics::set_decode_demand_tokens(&key, Some(entry.tokens));
        Self {
            ledger: ledger.clone(),
            key,
            tokens,
        }
    }
}

// Never panics: drop can run while unwinding, where a second panic aborts the router.
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(entry) = state.get_mut(&self.key) else {
            tracing::warn!(route_key = %self.key, "decode demand released without a reservation");
            return;
        };
        entry.tokens = entry.tokens.saturating_sub(u128::from(self.tokens));
        entry.requests = entry.requests.saturating_sub(1);
        if entry.requests == 0 {
            state.remove(&self.key);
            crate::metrics::set_decode_demand_tokens(&self.key, None);
        } else {
            crate::metrics::set_decode_demand_tokens(&self.key, Some(entry.tokens));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_inputs_add_and_release_independently() {
        let ledger = Ledger::default();
        let (a, b) = {
            let mut state = ledger.lock().unwrap();
            let a = Reservation::book(&ledger, &mut state, "d#dp0".into(), 32000);
            let b = Reservation::book(&ledger, &mut state, "d#dp0".into(), 32000);
            assert_eq!(
                state["d#dp0"],
                Demand {
                    tokens: 64000,
                    requests: 2,
                }
            );
            (a, b)
        };
        drop(a);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].tokens, 32000);
        drop(b);
        assert!(ledger.lock().unwrap().is_empty());
    }

    #[test]
    fn large_reservations_do_not_overflow_or_lose_small_charges() {
        let ledger = Ledger::default();
        let (a, b, small) = {
            let mut state = ledger.lock().unwrap();
            (
                Reservation::book(&ledger, &mut state, "d".into(), u64::MAX),
                Reservation::book(&ledger, &mut state, "d".into(), u64::MAX),
                Reservation::book(&ledger, &mut state, "d".into(), 1),
            )
        };
        assert_eq!(
            ledger.lock().unwrap()["d"].tokens,
            u128::from(u64::MAX) * 2 + 1
        );
        drop((a, b));
        assert_eq!(ledger.lock().unwrap()["d"].tokens, 1);
        drop(small);
        assert!(ledger.lock().unwrap().is_empty());
    }

    #[test]
    fn release_tolerates_a_missing_entry_and_a_poisoned_ledger() {
        let ledger = Ledger::default();
        let (gone, kept) = {
            let mut state = ledger.lock().unwrap();
            (
                Reservation::book(&ledger, &mut state, "gone".into(), 5),
                Reservation::book(&ledger, &mut state, "kept".into(), 7),
            )
        };
        ledger.lock().unwrap().remove("gone");
        drop(gone);
        let poisoner = ledger.clone();
        let _ = std::thread::spawn(move || {
            let _held = poisoner.lock().unwrap();
            panic!("poison the ledger");
        })
        .join();
        assert!(ledger.is_poisoned());
        drop(kept);
        assert!(ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty());
    }

    #[test]
    fn booked_tokens_are_exported_until_released() {
        let series = "infera_router_decode_demand_tokens{target=\"metrics-test#dp1\"}";
        let exported = || {
            let text = String::from_utf8(crate::metrics::render().0).unwrap();
            text.lines()
                .find_map(|line| line.strip_prefix(series).map(str::trim).map(str::to_owned))
        };
        let ledger = Ledger::default();
        let reservation = {
            let mut state = ledger.lock().unwrap();
            Reservation::book(&ledger, &mut state, "metrics-test#dp1".into(), 5)
        };
        assert_eq!(exported().as_deref(), Some("5"));
        drop(reservation);
        assert_eq!(exported(), None);
    }

    #[test]
    fn estimates_count_text_bytes_and_skip_inline_data() {
        let body = serde_json::json!({"messages":[
            {"role":"user","content":[
                {"type":"text","text":"12345678"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},
                {"type":"input_audio","input_audio":{"data":"UklGRiQAAABXQVZF","format":"wav"}},
            ]},
        ]});
        // "user" + "text" + "12345678" + "image_url" + "input_audio" + "wav"
        // = 39 bytes -> 10 tokens.
        assert_eq!(estimate_tokens(&body), 10);
        assert_eq!(estimate_tokens(&serde_json::json!({})), 0);
    }
}
