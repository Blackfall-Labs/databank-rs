use std::collections::HashMap;
use ternary_signal::Signal;

use crate::entry::BankEntry;
use crate::similarity::{sparse_cosine_similarity, QueryResult};
use crate::types::EntryId;

/// Vector similarity index for fast recall.
///
/// For 0.1 this is a brute-force linear scan. Future versions may add
/// IVF (inverted file index) or LSH (locality-sensitive hashing) for
/// sub-linear query time on large banks.
pub trait VectorIndex: Send + Sync {
    /// Record a new entry in the index.
    fn insert(&mut self, id: EntryId, vector: &[Signal]);

    /// Remove an entry from the index.
    fn remove(&mut self, id: EntryId);

    /// Query the index for the top_k most similar entries to the query vector.
    fn query(
        &self,
        query: &[Signal],
        entries: &HashMap<EntryId, BankEntry>,
        top_k: usize,
    ) -> Vec<QueryResult>;

    /// Rebuild the index from scratch (e.g. after loading from disk).
    fn rebuild(&mut self, entries: &HashMap<EntryId, BankEntry>);
}

/// Brute-force linear scan index. O(n) per query.
///
/// Sufficient for 0.1 where banks hold up to ~4096 entries.
/// At 64-dimensional vectors with integer arithmetic, a full scan
/// of 4096 entries takes <1ms on modern hardware.
#[derive(Debug, Default)]
pub struct BruteForceIndex;

impl VectorIndex for BruteForceIndex {
    fn insert(&mut self, _id: EntryId, _vector: &[Signal]) {
        // No-op: brute force scans the entry map directly.
    }

    fn remove(&mut self, _id: EntryId) {
        // No-op: brute force scans the entry map directly.
    }

    fn query(
        &self,
        query: &[Signal],
        entries: &HashMap<EntryId, BankEntry>,
        top_k: usize,
    ) -> Vec<QueryResult> {
        if top_k == 0 || entries.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<QueryResult> = entries
            .iter()
            .map(|(&id, entry)| QueryResult {
                entry_id: id,
                score: sparse_cosine_similarity(query, &entry.vector),
            })
            .collect();

        // Sort descending by score
        results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        results.truncate(top_k);
        results
    }

    fn rebuild(&mut self, _entries: &HashMap<EntryId, BankEntry>) {
        // No-op: brute force doesn't maintain state.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BankId, Temperature};

    fn sig(polarity: i8, magnitude: u8) -> Signal {
        Signal::new(polarity, magnitude)
    }

    fn make_entry(id: u64, vector: Vec<Signal>) -> (EntryId, BankEntry) {
        let eid = EntryId::from_raw(id);
        let entry = BankEntry::new(eid, vector, BankId::from_raw(1), Temperature::Hot, 0);
        (eid, entry)
    }

    #[test]
    fn brute_force_returns_top_k() {
        let mut entries = HashMap::new();
        // Entry 1: strongly matches query
        let (id1, e1) = make_entry(1, vec![sig(1, 200), sig(1, 100)]);
        entries.insert(id1, e1);
        // Entry 2: weakly matches
        let (id2, e2) = make_entry(2, vec![sig(1, 50), sig(1, 200)]);
        entries.insert(id2, e2);
        // Entry 3: opposite
        let (id3, e3) = make_entry(3, vec![sig(-1, 200), sig(-1, 100)]);
        entries.insert(id3, e3);

        let index = BruteForceIndex;
        let query = vec![sig(1, 200), sig(1, 100)];
        let results = index.query(&query, &entries, 2);

        assert_eq!(results.len(), 2);
        // Best match should be entry 1 (identical)
        assert_eq!(results[0].entry_id, id1);
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn brute_force_empty_returns_empty() {
        let entries = HashMap::new();
        let index = BruteForceIndex;
        let query = vec![sig(1, 100)];
        assert!(index.query(&query, &entries, 5).is_empty());
    }

    #[test]
    fn brute_force_top_k_zero_returns_empty() {
        let mut entries = HashMap::new();
        let (id, entry) = make_entry(1, vec![sig(1, 100)]);
        entries.insert(id, entry);

        let index = BruteForceIndex;
        let query = vec![sig(1, 100)];
        assert!(index.query(&query, &entries, 0).is_empty());
    }
}
