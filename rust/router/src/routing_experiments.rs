// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Opt-in R2/R3/R4 controls and per-request demand reservations.
//! Token demand is a router estimate, not physical allocator occupancy.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Off,
    Shadow,
    On,
}
impl Mode {
    fn parse(value: &str) -> anyhow::Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "shadow" => Ok(Self::Shadow),
            "on" => Ok(Self::On),
            _ => anyhow::bail!("expected off|shadow|on, got {value:?}"),
        }
    }
}
#[derive(Clone, Debug)]
pub struct Experiments {
    pub prefill_guard_completion: bool,
    pub decode: Mode,
    pub tiers: Mode,
    pub prefill: Mode,
    pub host_weight: f64,
}
impl Default for Experiments {
    fn default() -> Self {
        Self {
            prefill_guard_completion: false,
            decode: Mode::Off,
            tiers: Mode::Off,
            prefill: Mode::Off,
            host_weight: 0.0,
        }
    }
}
impl Experiments {
    pub fn from_env() -> anyhow::Result<Self> {
        let mode = |name| -> anyhow::Result<Mode> {
            Mode::parse(&std::env::var(name).unwrap_or_else(|_| "off".into()))
        };
        let prefill_guard_completion = match std::env::var("INFERA_PD_PREFILL_GUARD_RELEASE")
            .unwrap_or_else(|_| "decode".into())
            .as_str()
        {
            "decode" => false,
            "completion" => true,
            other => anyhow::bail!("invalid INFERA_PD_PREFILL_GUARD_RELEASE: {other}"),
        };
        let result = Self {
            prefill_guard_completion,
            decode: mode("INFERA_R2_DECODE_DEMAND")?,
            tiers: mode("INFERA_R3_CACHE_TIERS")?,
            prefill: mode("INFERA_R4_PREFILL_WORK")?,
            host_weight: std::env::var("INFERA_R3_HOST_WEIGHT")
                .unwrap_or_else(|_| "0".into())
                .parse()?,
        };
        if !result.host_weight.is_finite() || !(0.0..=1.0).contains(&result.host_weight) {
            anyhow::bail!("INFERA_R3_HOST_WEIGHT must be finite and in [0,1]");
        }
        if result.host_weight != 0.0 && result.tiers == Mode::Off {
            anyhow::bail!("host weight requires INFERA_R3_CACHE_TIERS=shadow|on");
        }
        Ok(result)
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Demand {
    pub tokens: f64,
    pub requests: usize,
    pub unknown: usize,
}
pub type Ledger = Arc<Mutex<HashMap<String, Demand>>>;

/// Reserved inside the selection lock; dropped on abandoned picks as well as
/// dispatched request completion. Never keyed by prefix, so equal prompts add.
#[derive(Debug)]
pub struct Reservation {
    ledger: Ledger,
    key: String,
    tokens: f64,
    unknown: bool,
}
impl Reservation {
    pub fn book_unknown(ledger: &Ledger, state: &mut HashMap<String, Demand>, key: String) -> Self {
        let mut reservation = Self::book(ledger, state, key.clone(), 0.0);
        state.get_mut(&key).expect("just inserted").unknown += 1;
        reservation.unknown = true;
        reservation
    }

    pub fn book(
        ledger: &Ledger,
        state: &mut HashMap<String, Demand>,
        key: String,
        tokens: f64,
    ) -> Self {
        let entry = state.entry(key.clone()).or_default();
        entry.tokens += tokens;
        entry.requests += 1;
        Self {
            ledger: ledger.clone(),
            key,
            tokens,
            unknown: false,
        }
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut state = self.ledger.lock().expect("demand ledger poisoned");
        if let Some(entry) = state.get_mut(&self.key) {
            entry.requests -= 1;
            if self.unknown {
                entry.unknown -= 1;
            }
            entry.tokens = (entry.tokens - self.tokens).max(0.0);
            if entry.requests == 0 {
                state.remove(&self.key);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identical_requests_are_additive_and_drop_releases_exactly_once() {
        let ledger = Ledger::default();
        let (a, b) = {
            let mut state = ledger.lock().unwrap();
            let a = Reservation::book(&ledger, &mut state, "d#0".into(), 32000.0);
            let b = Reservation::book(&ledger, &mut state, "d#0".into(), 32000.0);
            assert_eq!(state["d#0"].tokens, 64000.0);
            (a, b)
        };
        drop(a);
        assert_eq!(ledger.lock().unwrap()["d#0"].tokens, 32000.0);
        drop(b);
        assert!(ledger.lock().unwrap().is_empty());
    }
    #[test]
    fn modes_reject_typographical_errors() {
        assert!(Mode::parse("enabled").is_err());
        assert_eq!(Mode::parse("shadow").unwrap(), Mode::Shadow);
    }
}
