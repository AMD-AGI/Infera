// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Per-rank, per-tier prefix residency. Independent from the legacy KV view.
use crate::hasher::{hash_chunk, ROUTER_SEED};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Device,
    Host,
    Unknown,
}
impl Tier {
    pub fn from_medium(medium: Option<&str>) -> Self {
        match medium {
            None | Some("GPU") => Self::Device,
            Some("CPU_PINNED" | "CPU" | "CPU_TIER1") => Self::Host,
            _ => Self::Unknown,
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct TierStats {
    pub device_stores: u64,
    pub host_stores: u64,
    pub unknown_events: u64,
    pub rejected_stores: u64,
}
#[derive(Default)]
pub struct TierIndex {
    pub stats: TierStats,
    device: HashMap<u64, usize>,
    host: HashMap<u64, usize>,
    mapping: HashMap<u64, Residency>,
}
#[derive(Clone, Copy)]
struct Residency {
    chain: u64,
    device: bool,
    host: bool,
}

impl TierIndex {
    pub fn store(
        &mut self,
        tier: Tier,
        hashes: &[u64],
        parent: Option<u64>,
        tokens: &[u32],
        bs: usize,
    ) {
        if tier == Tier::Unknown {
            self.stats.unknown_events += 1;
            return;
        }
        if bs == 0 || hashes.len().checked_mul(bs) != Some(tokens.len()) {
            self.stats.rejected_stores += 1;
            return;
        }
        let mut chain = match parent {
            None => ROUTER_SEED,
            Some(p) => match self.mapping.get(&p) {
                Some(entry) => entry.chain,
                None => {
                    self.stats.rejected_stores += 1;
                    return;
                }
            },
        };
        match tier {
            Tier::Device => self.stats.device_stores += 1,
            Tier::Host => self.stats.host_stores += 1,
            _ => {}
        }
        for (engine, chunk) in hashes.iter().zip(tokens.chunks_exact(bs)) {
            chain = hash_chunk(chain, chunk);
            if self
                .mapping
                .get(engine)
                .is_some_and(|entry| entry.chain != chain)
            {
                // An engine hash changed identity; no old residency is reliable.
                self.mapping.clear();
                self.device.clear();
                self.host.clear();
                self.stats.rejected_stores += 1;
                return;
            }
            let entry = self.mapping.entry(*engine).or_insert(Residency {
                chain,
                device: false,
                host: false,
            });
            let (present, index) = match tier {
                Tier::Device => (&mut entry.device, &mut self.device),
                Tier::Host => (&mut entry.host, &mut self.host),
                Tier::Unknown => unreachable!(),
            };
            if !*present {
                *index.entry(chain).or_default() += 1;
                *present = true;
            }
        }
    }

    pub fn remove(&mut self, tier: Tier, hashes: &[u64]) {
        if tier == Tier::Unknown {
            self.stats.unknown_events += 1;
            return;
        }
        for engine in hashes {
            let Some(entry) = self.mapping.get_mut(engine) else {
                continue;
            };
            let (present, index) = match tier {
                Tier::Device => (&mut entry.device, &mut self.device),
                Tier::Host => (&mut entry.host, &mut self.host),
                Tier::Unknown => unreachable!(),
            };
            if *present {
                let count = index
                    .get_mut(&entry.chain)
                    .expect("resident block has a count");
                *count -= 1;
                if *count == 0 {
                    index.remove(&entry.chain);
                }
                *present = false;
            }
            if !entry.device && !entry.host {
                self.mapping.remove(engine);
            }
        }
    }
    /// Device prefix first, then contiguous host extension; never double-count.
    pub fn hits(&self, query: &[u64]) -> (usize, usize) {
        let gpu = query
            .iter()
            .take_while(|h| self.device.contains_key(h))
            .count();
        let host = query[gpu..]
            .iter()
            .take_while(|h| self.host.contains_key(h))
            .count();
        (gpu, host)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eviction_is_tier_local_and_overlap_is_not_double_counted() {
        let mut index = TierIndex::default();
        let tokens = [1, 2, 3, 4];
        let query = crate::hasher::hash_request(&tokens, 2);
        index.store(Tier::Device, &[11, 12], None, &tokens, 2);
        index.store(Tier::Host, &[11, 12], None, &tokens, 2);
        assert_eq!(index.hits(&query), (2, 0));
        index.remove(Tier::Device, &[11, 12]);
        assert_eq!(index.hits(&query), (0, 2));
        index.store(Tier::Device, &[11], None, &tokens[..2], 2);
        assert_eq!(index.hits(&query), (1, 1));
        index.remove(Tier::Host, &[11]);
        assert_eq!(index.hits(&query), (1, 1));
        index.remove(Tier::Device, &[11]);
        assert_eq!(index.hits(&query), (0, 0));
    }
    #[test]
    fn unknown_tier_and_missing_parent_do_not_create_hits() {
        let mut index = TierIndex::default();
        index.store(Tier::Unknown, &[1], None, &[7], 1);
        index.store(Tier::Host, &[2], Some(999), &[8], 1);
        assert!(index.mapping.is_empty());
    }
    #[test]
    fn deleting_one_engine_alias_preserves_the_other() {
        let mut index = TierIndex::default();
        let query = crate::hasher::hash_request(&[7], 1);
        for _ in 0..2 {
            index.store(Tier::Device, &[11], None, &[7], 1);
        }
        index.store(Tier::Device, &[12], None, &[7], 1);
        index.store(Tier::Host, &[11], None, &[7], 1);
        index.remove(Tier::Device, &[11, 11]);
        assert_eq!(index.hits(&query), (1, 0));
        index.remove(Tier::Device, &[12]);
        assert_eq!(index.hits(&query), (0, 1));
        index.remove(Tier::Host, &[11]);
        assert_eq!(index.hits(&query), (0, 0));
        assert!(index.mapping.is_empty());
    }

    #[test]
    fn reused_engine_hash_invalidates_old_residency() {
        let mut index = TierIndex::default();
        index.store(Tier::Device, &[11], None, &[7], 1);
        index.store(Tier::Host, &[11], None, &[8], 1);
        assert_eq!(index.hits(&crate::hasher::hash_request(&[7], 1)), (0, 0));
        assert_eq!(index.stats.rejected_stores, 1);
    }
}
