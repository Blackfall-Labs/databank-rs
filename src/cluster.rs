use std::collections::{HashMap, VecDeque};
use std::path::Path;
use ternary_signal::Signal;

use crate::bank::DataBank;
use crate::codec;
use crate::error::{DataBankError, Result};
use crate::journal::{self, JournalReader, JournalWriter};
use crate::similarity::QueryResult;
use crate::types::*;

/// Result of a cross-bank query.
#[derive(Debug, Clone)]
pub struct ClusterQueryResult {
    pub bank_id: BankId,
    pub bank_name: String,
    pub entry_id: EntryId,
    pub score: i32,
    pub normalized_score: i32,
}

/// Multi-bank manager — the brain's distributed representational memory.
///
/// Each region owns one or more banks in the cluster. The cluster provides
/// cross-bank operations (linking, traversal) and batch persistence.
///
/// Banks are indexed by both BankId and name for flexible lookup.
pub struct BankCluster {
    banks: HashMap<BankId, DataBank>,
    name_index: HashMap<String, BankId>,
    journal_writer: Option<JournalWriter>,
}

impl BankCluster {
    /// Create an empty cluster (no journal).
    pub fn new() -> Self {
        Self {
            banks: HashMap::new(),
            name_index: HashMap::new(),
            journal_writer: None,
        }
    }

    /// Create an empty cluster with a journal writer for crash recovery.
    pub fn with_journal(journal_path: &Path) -> Result<Self> {
        let writer = JournalWriter::open(journal_path)?;
        Ok(Self {
            banks: HashMap::new(),
            name_index: HashMap::new(),
            journal_writer: Some(writer),
        })
    }

    /// Get a reference to a bank by ID.
    pub fn get(&self, id: BankId) -> Option<&DataBank> {
        self.banks.get(&id)
    }

    /// Get a mutable reference to a bank by ID.
    pub fn get_mut(&mut self, id: BankId) -> Option<&mut DataBank> {
        self.banks.get_mut(&id)
    }

    /// Get a reference to a bank by name (e.g. "temporal.semantic").
    pub fn get_by_name(&self, name: &str) -> Option<&DataBank> {
        self.name_index.get(name).and_then(|id| self.banks.get(id))
    }

    /// Get a mutable reference to a bank by name.
    pub fn get_by_name_mut(&mut self, name: &str) -> Option<&mut DataBank> {
        self.name_index
            .get(name)
            .copied()
            .and_then(|id| self.banks.get_mut(&id))
    }

    /// Get an existing bank or create a new one if it doesn't exist.
    pub fn get_or_create(
        &mut self,
        id: BankId,
        name: String,
        config: BankConfig,
    ) -> &mut DataBank {
        if !self.banks.contains_key(&id) {
            let bank = DataBank::new(id, name.clone(), config);
            self.banks.insert(id, bank);
            self.name_index.insert(name, id);
        }
        self.banks.get_mut(&id).unwrap()
    }

    /// Add a bank to the cluster.
    pub fn add(&mut self, bank: DataBank) {
        let id = bank.id;
        let name = bank.name.clone();
        self.banks.insert(id, bank);
        self.name_index.insert(name, id);
    }

    /// Remove a bank from the cluster.
    pub fn remove(&mut self, id: BankId) -> Option<DataBank> {
        if let Some(bank) = self.banks.remove(&id) {
            self.name_index.remove(&bank.name);
            Some(bank)
        } else {
            None
        }
    }

    /// Create a cross-bank edge from one entry to another.
    ///
    /// The edge is added to the source entry. The reverse index on the
    /// target bank is NOT updated here (the target bank may not exist in
    /// this cluster if it's on a different host).
    pub fn link(
        &mut self,
        from: BankRef,
        to: BankRef,
        edge_type: EdgeType,
        weight: u8,
        tick: u64,
    ) -> Result<()> {
        let source_bank = self
            .banks
            .get_mut(&from.bank)
            .ok_or(DataBankError::BankNotFound { id: from.bank })?;

        let edge = Edge {
            edge_type,
            target: to,
            weight,
            created_tick: tick,
        };

        source_bank.add_edge(from.entry, edge)
    }

    /// Traverse edges from a starting entry, following edges of the given type.
    ///
    /// Returns all reachable BankRefs up to the given depth (BFS).
    /// Only follows edges that exist in banks within THIS cluster.
    pub fn traverse(
        &self,
        start: BankRef,
        edge_type: EdgeType,
        depth: usize,
    ) -> Vec<BankRef> {
        if depth == 0 {
            return Vec::new();
        }

        let mut visited: Vec<BankRef> = Vec::new();
        let mut queue: VecDeque<(BankRef, usize)> = VecDeque::new();
        queue.push_back((start, 0));

        while let Some((current, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            let Some(bank) = self.banks.get(&current.bank) else {
                continue;
            };

            for edge in bank.edges_from(current.entry) {
                if edge.edge_type == edge_type && !visited.contains(&edge.target) {
                    visited.push(edge.target);
                    queue.push_back((edge.target, current_depth + 1));
                }
            }
        }

        visited
    }

    /// Query across ALL banks in the cluster.
    ///
    /// Takes per-bank query vectors (banks may have different widths).
    /// Returns top_k results globally with z-score normalization.
    pub fn query_all(
        &self,
        query_per_bank: &HashMap<BankId, Vec<Signal>>,
        top_k: usize,
    ) -> Vec<ClusterQueryResult> {
        let mut all_results: Vec<ClusterQueryResult> = Vec::new();

        for (&bank_id, bank) in &self.banks {
            let query = match query_per_bank.get(&bank_id) {
                Some(q) => q,
                None => continue,
            };

            let results = bank.query_sparse(query, top_k);
            if results.is_empty() {
                continue;
            }

            // Compute mean and stddev for z-score normalization
            let (mean, stddev) = z_score_params(&results);

            for r in &results {
                let normalized = if stddev > 0 {
                    ((r.score as i64 - mean as i64) * 256 / stddev as i64) as i32
                } else {
                    0
                };

                all_results.push(ClusterQueryResult {
                    bank_id,
                    bank_name: bank.name.clone(),
                    entry_id: r.entry_id,
                    score: r.score,
                    normalized_score: normalized,
                });
            }
        }

        all_results.sort_by(|a, b| b.normalized_score.cmp(&a.normalized_score));
        all_results.truncate(top_k);
        all_results
    }

    /// Query a subset of banks by name prefix.
    ///
    /// E.g., "temporal." queries all banks whose names start with "temporal.".
    /// Uses the same query vector for all matching banks (assumes same width).
    pub fn query_by_prefix(
        &self,
        prefix: &str,
        query: &[Signal],
        top_k: usize,
    ) -> Vec<ClusterQueryResult> {
        let mut query_map = HashMap::new();
        for (name, &id) in &self.name_index {
            if name.starts_with(prefix) {
                query_map.insert(id, query.to_vec());
            }
        }
        self.query_all(&query_map, top_k)
    }

    /// Flush all dirty banks that have exceeded their persistence threshold.
    ///
    /// Each bank is saved atomically (temp + rename) to the given directory.
    /// Returns the number of banks flushed.
    pub fn flush_dirty(&mut self, dir: &Path, current_tick: u64) -> Result<usize> {
        let mut flushed = 0;

        let ids_to_flush: Vec<BankId> = self
            .banks
            .iter()
            .filter(|(_, bank)| bank.should_persist(current_tick))
            .map(|(&id, _)| id)
            .collect();

        for id in ids_to_flush {
            if let Some(bank) = self.banks.get(&id) {
                let path = dir.join(format!("{}.bank", bank.name));
                codec::save_atomic(bank, &path)?;
            }
            if let Some(bank) = self.banks.get_mut(&id) {
                bank.mark_persisted(current_tick);
            }
            flushed += 1;
        }

        Ok(flushed)
    }

    /// Load all `.bank` files from a directory into the cluster.
    pub fn load_all(dir: &Path) -> Result<Self> {
        let mut cluster = Self::new();

        if !dir.exists() {
            return Ok(cluster);
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("bank") {
                match codec::load(&path) {
                    Ok(bank) => {
                        log::info!("loaded bank '{}' ({} entries)", bank.name, bank.len());
                        cluster.add(bank);
                    }
                    Err(e) => {
                        log::error!("failed to load {:?}: {}", path, e);
                        return Err(e);
                    }
                }
            }
        }

        Ok(cluster)
    }

    /// Get all bank IDs in the cluster.
    pub fn bank_ids(&self) -> Vec<BankId> {
        self.banks.keys().copied().collect()
    }

    /// Get all bank names in the cluster.
    pub fn bank_names(&self) -> Vec<&str> {
        self.name_index.keys().map(|s| s.as_str()).collect()
    }

    /// Number of banks in the cluster.
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// Whether the cluster has no banks.
    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    /// Record a mutation to the journal (if one is configured).
    pub fn journal_mutation(&mut self, entry: crate::journal::JournalEntry) -> Result<()> {
        if let Some(ref mut writer) = self.journal_writer {
            writer.append(&entry)?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Load cluster from directory with journal replay.
    ///
    /// 1. Load all `.bank` files
    /// 2. Find and replay `.journal` file if it exists
    /// 3. Truncate journal after successful replay
    pub fn load_with_journal(dir: &Path) -> Result<Self> {
        let mut cluster = Self::load_all(dir)?;

        let journal_path = dir.join("databank.journal");
        if journal_path.exists() {
            let entries = JournalReader::read_all(&journal_path)?;
            if !entries.is_empty() {
                let count = JournalReader::replay(&entries, &mut cluster)?;
                log::info!("replayed {} journal entries from {:?}", count, journal_path);
            }
            journal::truncate_journal(&journal_path)?;
        }

        // Open a fresh journal for ongoing mutations
        let writer = JournalWriter::open(&journal_path)?;
        cluster.journal_writer = Some(writer);

        Ok(cluster)
    }

    /// Flush dirty banks AND truncate journal.
    ///
    /// After a full snapshot, the journal is no longer needed because all
    /// mutations are captured in the `.bank` files.
    pub fn flush_dirty_with_journal(
        &mut self,
        dir: &Path,
        current_tick: u64,
    ) -> Result<usize> {
        let flushed = self.flush_dirty(dir, current_tick)?;

        if flushed > 0 {
            let journal_path = dir.join("databank.journal");
            journal::truncate_journal(&journal_path)?;
        }

        Ok(flushed)
    }
}

/// Compute mean and standard deviation of query result scores (integer arithmetic).
fn z_score_params(results: &[QueryResult]) -> (i32, i32) {
    if results.is_empty() {
        return (0, 0);
    }
    let n = results.len() as i64;
    let sum: i64 = results.iter().map(|r| r.score as i64).sum();
    let mean = (sum / n) as i32;

    if n < 2 {
        return (mean, 1); // avoid division by zero; stddev=1 for single result
    }

    let variance: i64 = results.iter()
        .map(|r| {
            let diff = r.score as i64 - mean as i64;
            diff * diff
        })
        .sum::<i64>() / (n - 1);

    let stddev = isqrt_i64(variance) as i32;
    (mean, stddev.max(1)) // clamp to 1 to avoid division by zero
}

/// Integer square root (same algorithm as similarity.rs).
fn isqrt_i64(n: i64) -> i64 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    let mut x = 1i64 << (((64 - n.leading_zeros()) + 1) / 2);
    for _ in 0..8 {
        let next = (x + n / x) / 2;
        if next >= x { break; }
        x = next;
    }
    x
}

impl Default for BankCluster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ternary_signal::Signal;

    fn make_config(width: u16) -> BankConfig {
        BankConfig {
            vector_width: width,
            max_entries: 100,
            max_edges_per_entry: 8,
            persist_after_mutations: 1, // flush after every mutation for testing
            persist_after_ticks: 0,
            ..BankConfig::default()
        }
    }

    fn make_vector(width: u16) -> Vec<Signal> {
        (0..width)
            .map(|i| Signal::new(1, (i % 255) as u8 + 1))
            .collect()
    }

    #[test]
    fn create_and_lookup() {
        let mut cluster = BankCluster::new();
        let id = BankId::from_raw(1);
        cluster.get_or_create(id, "temporal.semantic".into(), make_config(64));

        assert!(cluster.get(id).is_some());
        assert!(cluster.get_by_name("temporal.semantic").is_some());
        assert!(cluster.get_by_name("nonexistent").is_none());
        assert_eq!(cluster.len(), 1);
    }

    #[test]
    fn remove_bank() {
        let mut cluster = BankCluster::new();
        let id = BankId::from_raw(1);
        cluster.get_or_create(id, "test".into(), make_config(32));
        assert_eq!(cluster.len(), 1);

        let removed = cluster.remove(id);
        assert!(removed.is_some());
        assert_eq!(cluster.len(), 0);
        assert!(cluster.get_by_name("test").is_none());
    }

    #[test]
    fn cross_bank_linking() {
        let mut cluster = BankCluster::new();
        let id_a = BankId::from_raw(1);
        let id_b = BankId::from_raw(2);

        let bank_a = cluster.get_or_create(id_a, "bank_a".into(), make_config(4));
        let entry_a = bank_a
            .insert(make_vector(4), Temperature::Hot, 0)
            .unwrap();

        let bank_b = cluster.get_or_create(id_b, "bank_b".into(), make_config(4));
        let entry_b = bank_b
            .insert(make_vector(4), Temperature::Hot, 0)
            .unwrap();

        let from = BankRef {
            bank: id_a,
            entry: entry_a,
        };
        let to = BankRef {
            bank: id_b,
            entry: entry_b,
        };

        cluster
            .link(from, to, EdgeType::SoundsLike, 200, 0)
            .unwrap();

        // Verify edge exists
        let edges = cluster.get(id_a).unwrap().edges_from(entry_a);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, EdgeType::SoundsLike);
        assert_eq!(edges[0].target, to);
    }

    #[test]
    fn traverse_follows_edges() {
        let mut cluster = BankCluster::new();
        let id_a = BankId::from_raw(1);
        let id_b = BankId::from_raw(2);
        let id_c = BankId::from_raw(3);

        let bank_a = cluster.get_or_create(id_a, "a".into(), make_config(4));
        let ea = bank_a.insert(make_vector(4), Temperature::Hot, 0).unwrap();

        let bank_b = cluster.get_or_create(id_b, "b".into(), make_config(4));
        let eb = bank_b.insert(make_vector(4), Temperature::Hot, 0).unwrap();

        let bank_c = cluster.get_or_create(id_c, "c".into(), make_config(4));
        let ec = bank_c.insert(make_vector(4), Temperature::Hot, 0).unwrap();

        // a → b → c (chain of RelatedTo edges)
        let ref_a = BankRef { bank: id_a, entry: ea };
        let ref_b = BankRef { bank: id_b, entry: eb };
        let ref_c = BankRef { bank: id_c, entry: ec };

        cluster.link(ref_a, ref_b, EdgeType::RelatedTo, 200, 0).unwrap();
        cluster.link(ref_b, ref_c, EdgeType::RelatedTo, 150, 0).unwrap();

        // Depth 1: should find b
        let d1 = cluster.traverse(ref_a, EdgeType::RelatedTo, 1);
        assert_eq!(d1.len(), 1);
        assert_eq!(d1[0], ref_b);

        // Depth 2: should find b and c
        let d2 = cluster.traverse(ref_a, EdgeType::RelatedTo, 2);
        assert_eq!(d2.len(), 2);
        assert!(d2.contains(&ref_b));
        assert!(d2.contains(&ref_c));

        // Depth 0: nothing
        let d0 = cluster.traverse(ref_a, EdgeType::RelatedTo, 0);
        assert!(d0.is_empty());

        // Wrong edge type: nothing
        let wrong = cluster.traverse(ref_a, EdgeType::LooksLike, 2);
        assert!(wrong.is_empty());
    }

    #[test]
    fn flush_and_load_round_trip() {
        let mut cluster = BankCluster::new();
        let id = BankId::from_raw(1);
        let bank = cluster.get_or_create(id, "test.round.trip".into(), make_config(4));
        bank.insert(make_vector(4), Temperature::Hot, 0).unwrap();
        bank.insert(make_vector(4), Temperature::Warm, 0).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let flushed = cluster.flush_dirty(dir.path(), 100).unwrap();
        assert_eq!(flushed, 1);

        // Load back
        let loaded = BankCluster::load_all(dir.path()).unwrap();
        assert_eq!(loaded.len(), 1);
        let loaded_bank = loaded.get_by_name("test.round.trip").unwrap();
        assert_eq!(loaded_bank.len(), 2);
        assert_eq!(loaded_bank.id, id);
    }

    #[test]
    fn load_all_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let cluster = BankCluster::load_all(dir.path()).unwrap();
        assert_eq!(cluster.len(), 0);
    }

    #[test]
    fn query_all_cross_bank() {
        let mut cluster = BankCluster::new();
        let id_a = BankId::from_raw(1);
        let id_b = BankId::from_raw(2);

        let bank_a = cluster.get_or_create(id_a, "temporal.semantic".into(), make_config(4));
        bank_a.insert(make_vector(4), Temperature::Hot, 0).unwrap();

        let bank_b = cluster.get_or_create(id_b, "temporal.auditory".into(), make_config(4));
        bank_b.insert(make_vector(4), Temperature::Hot, 0).unwrap();

        let mut queries = HashMap::new();
        queries.insert(id_a, make_vector(4));
        queries.insert(id_b, make_vector(4));

        let results = cluster.query_all(&queries, 5);
        assert_eq!(results.len(), 2);
        // Both should have high scores (identical vectors)
        for r in &results {
            assert!(r.score > 200, "expected high score, got {}", r.score);
        }
    }

    #[test]
    fn query_by_prefix_filters() {
        let mut cluster = BankCluster::new();
        let id_a = BankId::from_raw(1);
        let id_b = BankId::from_raw(2);
        let id_c = BankId::from_raw(3);

        cluster.get_or_create(id_a, "temporal.semantic".into(), make_config(4))
            .insert(make_vector(4), Temperature::Hot, 0).unwrap();
        cluster.get_or_create(id_b, "temporal.auditory".into(), make_config(4))
            .insert(make_vector(4), Temperature::Hot, 0).unwrap();
        cluster.get_or_create(id_c, "occipital.v4".into(), make_config(4))
            .insert(make_vector(4), Temperature::Hot, 0).unwrap();

        // Query only temporal.* banks
        let results = cluster.query_by_prefix("temporal.", &make_vector(4), 5);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.bank_name.starts_with("temporal."));
        }
    }

    #[test]
    fn load_all_nonexistent_dir() {
        let cluster = BankCluster::load_all(Path::new("/nonexistent/path/that/does/not/exist"));
        assert!(cluster.is_ok());
        assert_eq!(cluster.unwrap().len(), 0);
    }
}
