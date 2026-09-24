// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Per-rank, per-tier prefix residency. Independent from the legacy KV view.
use crate::hasher::{hash_chunk, ROUTER_SEED};
use std::collections::{HashMap, HashSet};

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
    device: HashSet<u64>,
    host: HashSet<u64>,
    // Engine hash -> router chain; kept while either tier retains the block.
    mapping: HashMap<u64, u64>,
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
                Some(h) => *h,
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
            self.mapping.insert(*engine, chain);
            match tier {
                Tier::Device => {
                    self.device.insert(chain);
                }
                Tier::Host => {
                    self.host.insert(chain);
                }
                _ => {}
            }
        }
    }
    pub fn remove(&mut self, tier: Tier, hashes: &[u64]) {
        if tier == Tier::Unknown {
            self.stats.unknown_events += 1;
            return;
        }
        for engine in hashes {
            if let Some(&chain) = self.mapping.get(engine) {
                match tier {
                    Tier::Device => {
                        self.device.remove(&chain);
                    }
                    Tier::Host => {
                        self.host.remove(&chain);
                    }
                    _ => continue,
                }
                if !self.device.contains(&chain) && !self.host.contains(&chain) {
                    self.mapping.remove(engine);
                }
            }
        }
    }
    /// Device prefix first, then contiguous host extension; never double-count.
    pub fn hits(&self, query: &[u64]) -> (usize, usize) {
        let gpu = query.iter().take_while(|h| self.device.contains(h)).count();
        let host = query[gpu..]
            .iter()
            .take_while(|h| self.host.contains(h))
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
}
