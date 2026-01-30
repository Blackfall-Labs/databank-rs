//! Inverted File (IVF) Index for Sub-Linear Similarity Search
//!
//! Partitions the vector space into k clusters. Each entry is assigned to
//! its nearest centroid. Queries search only the `nprobe` nearest clusters
//! instead of all entries, giving ~k/nprobe speedup.
//!
//! For 0.2, centroids are initialized by random sampling from entries.
//! Full k-means iterative refinement is deferred to 0.3.

use std::collections::HashMap;
use ternary_signal::Signal;

use crate::entry::BankEntry;
use crate::index::VectorIndex;
use crate::similarity::{sparse_cosine_similarity, QueryResult};
use crate::types::EntryId;

/// Inverted File Index — partitions vector space into clusters for
/// sub-linear approximate nearest neighbor search.
pub struct IvfIndex {
    /// k centroids stored as signed i32 vectors (polarity * magnitude).
    centroids: Vec<Vec<i32>>,
    /// Per-centroid list of assigned entry IDs.
    assignments: Vec<Vec<EntryId>>,
    /// Number of clusters to search per query.
    nprobe: usize,
    /// Number of centroids.
    k: usize,
}

impl IvfIndex {
    /// Create a new IVF index with the given parameters.
    ///
    /// - `k`: number of centroids (typically sqrt(n))
    /// - `nprobe`: number of clusters to search per query (default: 4)
    pub fn new(k: usize, nprobe: usize) -> Self {
        Self {
            centroids: Vec::new(),
            assignments: Vec::new(),
            nprobe: nprobe.max(1),
            k: k.max(1),
        }
    }

    /// Find the nearest centroid index for a given vector.
    fn nearest_centroid(&self, vector: &[Signal]) -> usize {
        if self.centroids.is_empty() {
            return 0;
        }
        let i32_vec = signals_to_i32_vec(vector);
        let mut best_idx = 0;
        let mut best_score = i64::MIN;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let score = dot_i32(&i32_vec, centroid);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Find the `nprobe` nearest centroid indices for a query.
    fn nearest_centroids(&self, query: &[Signal]) -> Vec<usize> {
        if self.centroids.is_empty() {
            return Vec::new();
        }
        let i32_vec = signals_to_i32_vec(query);
        let mut scored: Vec<(usize, i64)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, dot_i32(&i32_vec, c)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scored
            .iter()
            .take(self.nprobe.min(scored.len()))
            .map(|&(i, _)| i)
            .collect()
    }

    /// Initialize centroids by sampling from existing entries.
    fn initialize_centroids(&mut self, entries: &HashMap<EntryId, BankEntry>) {
        if entries.is_empty() {
            self.centroids.clear();
            self.assignments.clear();
            return;
        }

        let k = self.k.min(entries.len());
        let entry_list: Vec<&BankEntry> = entries.values().collect();

        // Deterministic spacing: pick every (n/k)th entry
        let step = entry_list.len() / k;
        self.centroids = (0..k)
            .map(|i| {
                let idx = (i * step).min(entry_list.len() - 1);
                signals_to_i32_vec(&entry_list[idx].vector)
            })
            .collect();

        self.assignments = vec![Vec::new(); k];
    }

    /// Assign all entries to their nearest centroid.
    fn assign_all(&mut self, entries: &HashMap<EntryId, BankEntry>) {
        for bucket in &mut self.assignments {
            bucket.clear();
        }
        if self.centroids.is_empty() {
            return;
        }
        for (&id, entry) in entries {
            let ci = self.nearest_centroid(&entry.vector);
            if ci < self.assignments.len() {
                self.assignments[ci].push(id);
            }
        }
    }
}

impl VectorIndex for IvfIndex {
    fn insert(&mut self, id: EntryId, vector: &[Signal]) {
        if self.centroids.is_empty() {
            // No centroids yet — can't assign. Will rebuild on next query.
            return;
        }
        let ci = self.nearest_centroid_from_i32(&signals_to_i32_vec(vector));
        if ci < self.assignments.len() {
            self.assignments[ci].push(id);
        }
    }

    fn remove(&mut self, id: EntryId) {
        for bucket in &mut self.assignments {
            bucket.retain(|&eid| eid != id);
        }
    }

    fn query(
        &self,
        query: &[Signal],
        entries: &HashMap<EntryId, BankEntry>,
        top_k: usize,
    ) -> Vec<QueryResult> {
        if top_k == 0 || entries.is_empty() || self.centroids.is_empty() {
            // Fallback to brute force if no centroids
            return brute_force_query(query, entries, top_k);
        }

        let probe_indices = self.nearest_centroids(query);
        let mut results: Vec<QueryResult> = Vec::new();

        for ci in &probe_indices {
            if *ci >= self.assignments.len() {
                continue;
            }
            for &id in &self.assignments[*ci] {
                if let Some(entry) = entries.get(&id) {
                    let score = sparse_cosine_similarity(query, &entry.vector);
                    results.push(QueryResult {
                        entry_id: id,
                        score,
                    });
                }
            }
        }

        results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        results.truncate(top_k);
        results
    }

    fn rebuild(&mut self, entries: &HashMap<EntryId, BankEntry>) {
        self.initialize_centroids(entries);
        self.assign_all(entries);
    }
}

impl IvfIndex {
    /// Rebuild with k-means clustering.
    ///
    /// Iteratively refines centroids by:
    /// 1. Assign each entry to nearest centroid
    /// 2. Recompute centroids as mean of assigned entries
    /// 3. Repeat until convergence or max_iterations
    ///
    /// Uses integer arithmetic only (ASTRO_004 compliant).
    pub fn rebuild_kmeans(
        &mut self,
        entries: &HashMap<EntryId, BankEntry>,
        max_iterations: usize,
    ) {
        if entries.is_empty() {
            self.centroids.clear();
            self.assignments.clear();
            return;
        }

        // Initialize centroids with deterministic spacing (same as before)
        self.initialize_centroids(entries);
        if self.centroids.is_empty() {
            return;
        }

        let width = self.centroids[0].len();
        let k = self.centroids.len();
        let entry_vecs: Vec<(EntryId, Vec<i32>)> = entries.iter()
            .map(|(&id, e)| (id, signals_to_i32_vec(&e.vector)))
            .collect();

        for _iter in 0..max_iterations {
            // Step 1: Assign entries to nearest centroid
            let mut new_assignments: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (i, (_id, vec)) in entry_vecs.iter().enumerate() {
                let ci = self.nearest_centroid_from_i32(vec);
                new_assignments[ci].push(i);
            }

            // Step 2: Recompute centroids as mean of assigned entries
            let mut changed = false;
            for ci in 0..k {
                if new_assignments[ci].is_empty() {
                    continue; // keep old centroid if no assignments
                }
                let n = new_assignments[ci].len() as i64;
                let mut new_centroid = vec![0i64; width];
                for &ei in &new_assignments[ci] {
                    for (j, &v) in entry_vecs[ei].1.iter().enumerate() {
                        if j < width {
                            new_centroid[j] += v as i64;
                        }
                    }
                }
                let updated: Vec<i32> = new_centroid.iter()
                    .map(|&v| (v / n) as i32)
                    .collect();

                if updated != self.centroids[ci] {
                    changed = true;
                    self.centroids[ci] = updated;
                }
            }

            if !changed {
                break; // converged
            }
        }

        // Final assignment pass
        self.assign_all(entries);
    }

    /// Internal: nearest centroid from pre-converted i32 vector.
    fn nearest_centroid_from_i32(&self, i32_vec: &[i32]) -> usize {
        let mut best_idx = 0;
        let mut best_score = i64::MIN;
        for (i, centroid) in self.centroids.iter().enumerate() {
            let score = dot_i32(i32_vec, centroid);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert Signal vector to i32 vector (polarity * magnitude).
fn signals_to_i32_vec(signals: &[Signal]) -> Vec<i32> {
    signals
        .iter()
        .map(|s| s.polarity as i32 * s.magnitude as i32)
        .collect()
}

/// Dot product of two i32 vectors (integer only).
fn dot_i32(a: &[i32], b: &[i32]) -> i64 {
    let len = a.len().min(b.len());
    let mut sum: i64 = 0;
    for i in 0..len {
        sum += a[i] as i64 * b[i] as i64;
    }
    sum
}

/// Brute-force fallback when IVF has no centroids.
fn brute_force_query(
    query: &[Signal],
    entries: &HashMap<EntryId, BankEntry>,
    top_k: usize,
) -> Vec<QueryResult> {
    let mut results: Vec<QueryResult> = entries
        .iter()
        .map(|(&id, entry)| QueryResult {
            entry_id: id,
            score: sparse_cosine_similarity(query, &entry.vector),
        })
        .collect();
    results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
    results.truncate(top_k);
    results
}

/// Index type selector for BankConfig.
#[derive(Debug, Clone)]
pub enum IndexType {
    /// Linear scan of all entries. O(n) per query.
    BruteForce,
    /// Inverted file index. O(n/k * nprobe) per query.
    Ivf { k: usize, nprobe: usize },
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::BruteForce
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
    fn ivf_rebuild_and_query() {
        let mut entries = HashMap::new();
        // Create 16 entries with distinct patterns
        for i in 0u64..16 {
            let v = vec![
                sig(1, (i * 10 + 10).min(255) as u8),
                sig(1, (i * 5 + 5).min(255) as u8),
                sig(if i < 8 { 1 } else { -1 }, 100),
                sig(1, 50),
            ];
            let (id, e) = make_entry(i + 1, v);
            entries.insert(id, e);
        }

        let mut index = IvfIndex::new(4, 2);
        index.rebuild(&entries);

        assert_eq!(index.centroids.len(), 4);
        let total_assigned: usize = index.assignments.iter().map(|b| b.len()).sum();
        assert_eq!(total_assigned, 16);

        // Query with a vector similar to entry 1
        let query = vec![sig(1, 10), sig(1, 5), sig(1, 100), sig(1, 50)];
        let results = index.query(&query, &entries, 3);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // Top result should have positive score (same direction)
        assert!(results[0].score > 0);
    }

    #[test]
    fn ivf_insert_and_remove() {
        let mut entries = HashMap::new();
        for i in 0u64..8 {
            let v = vec![sig(1, (i * 30 + 10).min(255) as u8), sig(1, 100)];
            let (id, e) = make_entry(i + 1, v);
            entries.insert(id, e);
        }

        let mut index = IvfIndex::new(2, 2);
        index.rebuild(&entries);

        let total_before: usize = index.assignments.iter().map(|b| b.len()).sum();
        assert_eq!(total_before, 8);

        // Insert a new entry
        let new_id = EntryId::from_raw(100);
        let new_vec = vec![sig(1, 200), sig(1, 200)];
        index.insert(new_id, &new_vec);
        let total_after: usize = index.assignments.iter().map(|b| b.len()).sum();
        assert_eq!(total_after, 9);

        // Remove it
        index.remove(new_id);
        let total_removed: usize = index.assignments.iter().map(|b| b.len()).sum();
        assert_eq!(total_removed, 8);
    }

    #[test]
    fn ivf_empty_entries_fallback() {
        let entries = HashMap::new();
        let index = IvfIndex::new(4, 2);
        let query = vec![sig(1, 100)];
        let results = index.query(&query, &entries, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn ivf_accuracy_vs_brute_force() {
        // Generate entries and compare IVF results vs brute force
        let mut entries = HashMap::new();
        for i in 0u64..32 {
            let v = vec![
                sig(1, ((i * 7 + 3) % 255 + 1) as u8),
                sig(if i % 3 == 0 { -1 } else { 1 }, ((i * 11 + 7) % 255 + 1) as u8),
                sig(1, ((i * 13 + 11) % 255 + 1) as u8),
                sig(if i % 5 == 0 { -1 } else { 1 }, ((i * 17 + 13) % 255 + 1) as u8),
            ];
            let (id, e) = make_entry(i + 1, v);
            entries.insert(id, e);
        }

        // Brute force baseline
        let query = vec![sig(1, 100), sig(1, 150), sig(1, 200), sig(1, 50)];
        let bf_results = brute_force_query(&query, &entries, 5);

        // IVF with full probe (nprobe = k) should match brute force
        let mut index = IvfIndex::new(4, 4); // nprobe = k, searches all clusters
        index.rebuild(&entries);
        let ivf_results = index.query(&query, &entries, 5);

        // Both should find the same top result
        assert!(!bf_results.is_empty());
        assert!(!ivf_results.is_empty());
        assert_eq!(bf_results[0].entry_id, ivf_results[0].entry_id);
    }

    #[test]
    fn ivf_k_larger_than_entries() {
        let mut entries = HashMap::new();
        let (id, e) = make_entry(1, vec![sig(1, 100), sig(1, 200)]);
        entries.insert(id, e);

        let mut index = IvfIndex::new(100, 4); // k=100 but only 1 entry
        index.rebuild(&entries);
        assert_eq!(index.centroids.len(), 1); // clamped to entries.len()

        let query = vec![sig(1, 100), sig(1, 200)];
        let results = index.query(&query, &entries, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn kmeans_convergence() {
        let mut entries = HashMap::new();
        // Two clear clusters: positive and negative
        for i in 0u64..16 {
            let pol = if i < 8 { 1 } else { -1 };
            let v = vec![
                sig(pol, 200),
                sig(pol, 150),
                sig(pol, 100),
                sig(1, (i * 10 + 10).min(255) as u8), // slight variation
            ];
            let (id, e) = make_entry(i + 1, v);
            entries.insert(id, e);
        }

        let mut index = IvfIndex::new(2, 2);
        index.rebuild_kmeans(&entries, 20);

        assert_eq!(index.centroids.len(), 2);
        let total: usize = index.assignments.iter().map(|b| b.len()).sum();
        assert_eq!(total, 16);

        // Each cluster should have roughly 8 entries
        for bucket in &index.assignments {
            assert!(bucket.len() >= 4, "cluster too small: {}", bucket.len());
        }

        // Query accuracy: search for positive cluster
        let query = vec![sig(1, 200), sig(1, 150), sig(1, 100), sig(1, 50)];
        let results = index.query(&query, &entries, 5);
        assert!(!results.is_empty());
        assert!(results[0].score > 200, "top result should be strongly positive");
    }

    #[test]
    fn kmeans_vs_brute_force_accuracy() {
        let mut entries = HashMap::new();
        for i in 0u64..32 {
            let v = vec![
                sig(1, ((i * 7 + 3) % 255 + 1) as u8),
                sig(if i % 3 == 0 { -1 } else { 1 }, ((i * 11 + 7) % 255 + 1) as u8),
                sig(1, ((i * 13 + 11) % 255 + 1) as u8),
                sig(if i % 5 == 0 { -1 } else { 1 }, ((i * 17 + 13) % 255 + 1) as u8),
            ];
            let (id, e) = make_entry(i + 1, v);
            entries.insert(id, e);
        }

        let query = vec![sig(1, 100), sig(1, 150), sig(1, 200), sig(1, 50)];
        let bf_results = brute_force_query(&query, &entries, 5);

        // K-means with full probe should match
        let mut index = IvfIndex::new(4, 4);
        index.rebuild_kmeans(&entries, 15);
        let km_results = index.query(&query, &entries, 5);

        assert!(!bf_results.is_empty());
        assert!(!km_results.is_empty());
        assert_eq!(bf_results[0].entry_id, km_results[0].entry_id);
    }

    #[test]
    fn dot_i32_correctness() {
        assert_eq!(dot_i32(&[1, 2, 3], &[4, 5, 6]), 32);
        assert_eq!(dot_i32(&[100, -200], &[-100, 200]), -50000);
        assert_eq!(dot_i32(&[], &[]), 0);
    }
}
