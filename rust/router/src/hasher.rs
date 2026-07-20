///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Chained XXH3-64 hashing for KV-aware routing — the Rust twin of
//! `infera.router.kv_event.hasher`.
//!
//! Both the query side (tokenized prompt) and the event side (the kv-event
//! subscriber processing `BlockStored`) feed tokens through the SAME chain, so
//! the hashes match byte-for-byte with the Python router.

use xxhash_rust::xxh3::xxh3_64;

/// Seed for the first block in a chain (matches Python `ROUTER_SEED = 0`).
pub const ROUTER_SEED: u64 = 0;

/// XXH3-64 over `parent (8 LE bytes) || token_ids (4 LE bytes each)`.
pub fn hash_chunk(parent: u64, token_ids: &[u32]) -> u64 {
    let mut buf = Vec::with_capacity(8 + 4 * token_ids.len());
    buf.extend_from_slice(&parent.to_le_bytes());
    for &t in token_ids {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    xxh3_64(&buf)
}

/// Chained block hashes for a token sequence; trailing partial block dropped.
pub fn hash_request(token_ids: &[u32], block_size: usize) -> Vec<u64> {
    if block_size == 0 {
        return Vec::new();
    }
    let mut parent = ROUTER_SEED;
    let n_full = token_ids.len() / block_size;
    let mut out = Vec::with_capacity(n_full);
    for i in 0..n_full {
        let chunk = &token_ids[i * block_size..(i + 1) * block_size];
        parent = hash_chunk(parent, chunk);
        out.push(parent);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values captured from the Python implementation:
    //   from infera.router.kv_event.hasher import hash_chunk, hash_request
    //   hex(hash_chunk(0, [1, 2, 3, 4]))            -> 0x341839abbe30be0b
    //   [hex(x) for x in hash_request([1..10], 4)]  -> [..30be0b, ..b5ca]
    #[test]
    fn hash_chunk_matches_python_golden() {
        assert_eq!(hash_chunk(0, &[1, 2, 3, 4]), 0x341839abbe30be0b);
    }

    #[test]
    fn hash_request_chains_and_drops_partial() {
        // 10 tokens, block_size 4 => 2 full blocks, last 2 tokens dropped.
        let h = hash_request(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0], 0x341839abbe30be0b);
        assert_eq!(h[1], 0x37b6de0623bdb5ca);
        // block 2 chains off block 1 (same as the event side's hash_chunk).
        assert_eq!(h[1], hash_chunk(h[0], &[5, 6, 7, 8]));
    }

    #[test]
    fn block_size_zero_is_empty() {
        assert!(hash_request(&[1, 2, 3], 0).is_empty());
    }
}
