// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Router-local in-flight input tokens, not physical allocator occupancy.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    pub unknown: usize,
}
pub(crate) type Ledger = Arc<Mutex<HashMap<String, Demand>>>;

/// Booked atomically with selection and released even if the pick is abandoned.
#[derive(Debug)]
pub struct Reservation {
    ledger: Ledger,
    key: String,
    tokens: Option<u64>,
}

impl Reservation {
    pub(crate) fn book(
        ledger: &Ledger,
        state: &mut HashMap<String, Demand>,
        key: String,
        tokens: Option<u64>,
    ) -> Self {
        let entry = state.entry(key.clone()).or_default();
        entry.tokens += u128::from(tokens.unwrap_or(0));
        entry.requests += 1;
        entry.unknown += usize::from(tokens.is_none());
        Self {
            ledger: ledger.clone(),
            key,
            tokens,
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().expect("decode demand ledger poisoned");
        let entry = state.get_mut(&self.key).expect("reserved decode target");
        entry.tokens -= u128::from(self.tokens.unwrap_or(0));
        entry.requests -= 1;
        entry.unknown -= usize::from(self.tokens.is_none());
        if entry.requests == 0 {
            state.remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_inputs_add_and_unknown_leases_release_independently() {
        let ledger = Ledger::default();
        let (a, b, unknown) = {
            let mut state = ledger.lock().unwrap();
            let a = Reservation::book(&ledger, &mut state, "d#dp0".into(), Some(32000));
            let b = Reservation::book(&ledger, &mut state, "d#dp0".into(), Some(32000));
            let unknown = Reservation::book(&ledger, &mut state, "d#dp0".into(), None);
            assert_eq!(
                state["d#dp0"],
                Demand {
                    tokens: 64000,
                    requests: 3,
                    unknown: 1
                }
            );
            (a, b, unknown)
        };
        drop(a);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].tokens, 32000);
        drop(unknown);
        assert_eq!(ledger.lock().unwrap()["d#dp0"].unknown, 0);
        drop(b);
        assert!(ledger.lock().unwrap().is_empty());
    }

    #[test]
    fn large_reservations_do_not_overflow_or_lose_small_charges() {
        let ledger = Ledger::default();
        let (a, b, small) = {
            let mut state = ledger.lock().unwrap();
            (
                Reservation::book(&ledger, &mut state, "d".into(), Some(u64::MAX)),
                Reservation::book(&ledger, &mut state, "d".into(), Some(u64::MAX)),
                Reservation::book(&ledger, &mut state, "d".into(), Some(1)),
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
}
