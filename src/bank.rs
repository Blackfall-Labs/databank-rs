use std::collections::HashMap;
use ternary_signal::Signal;

use crate::entry::BankEntry;
use crate::error::{DataBankError, Result};
use crate::index::{BruteForceIndex, VectorIndex};
use crate::similarity::QueryResult;
use crate::types::{BankConfig, BankId, BankRef, Edge, EdgeType, EntryId, Temperature};

/// A single databank — one region's representational memory.
///
/// Each brain region owns one or more DataBanks, each storing signal-vector
/// fragments of distributed concepts. Entries are fixed-width signal vectors
/// with typed edges to entries in other banks.
///
/// The bank manages its own persistence cadence (mutations + ticks since
/// last flush). The kernel calls `should_persist()` each tick and flushes
/// dirty banks that exceed their threshold.
pub struct DataBank {
    /// Bank identity (temporally sortable, region-tagged).
    pub id: BankId,
    /// Per-region configuration.
    config: BankConfig,
    /// Human-readable name for debugging (e.g. "temporal.semantic").
    pub name: String,
    /// All entries, indexed by EntryId.
    entries: HashMap<EntryId, BankEntry>,
    /// Next sequence counter for EntryId generation.
    next_seq: u32,
    /// Vector similarity index (brute-force for 0.1).
    vector_index: BruteForceIndex,
    /// Reverse edge index: "who points to me?"
    reverse_edges: HashMap<EntryId, Vec<(BankRef, EdgeType)>>,
    /// Mutations since last persistence flush.
    mutations_since_persist: u32,
    /// Tick of last persistence flush.
    last_persist_tick: u64,
    /// Whether the bank has unsaved changes.
    dirty: bool,
}

impl DataBank {
    /// Create a new empty bank with the given identity and configuration.
    pub fn new(id: BankId, name: String, config: BankConfig) -> Self {
        Self {
            id,
            config,
            name,
            entries: HashMap::new(),
            next_seq: 0,
            vector_index: BruteForceIndex,
            reverse_edges: HashMap::new(),
            mutations_since_persist: 0,
            last_persist_tick: 0,
            dirty: false,
        }
    }

    /// Insert a new entry into the bank.
    ///
    /// The vector must match the bank's configured `vector_width`.
    /// If the bank is at capacity, the lowest-scoring entry is evicted first.
    pub fn insert(
        &mut self,
        vector: Vec<Signal>,
        temperature: Temperature,
        tick: u64,
    ) -> Result<EntryId> {
        // Validate vector width
        if vector.len() != self.config.vector_width as usize {
            return Err(DataBankError::VectorWidthMismatch {
                expected: self.config.vector_width,
                got: vector.len() as u16,
            });
        }

        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries as usize {
            self.evict_lowest(tick);
        }

        // Still full after eviction? (shouldn't happen, but be safe)
        if self.entries.len() >= self.config.max_entries as usize {
            return Err(DataBankError::BankFull {
                capacity: self.config.max_entries,
            });
        }

        let id = EntryId::new(self.next_seq);
        self.next_seq = self.next_seq.wrapping_add(1);

        let entry = BankEntry::new(id, vector.clone(), self.id, temperature, tick);
        self.vector_index.insert(id, &vector);
        self.entries.insert(id, entry);

        self.mark_mutated();
        Ok(id)
    }

    /// Get a reference to an entry by ID.
    pub fn get(&self, id: EntryId) -> Option<&BankEntry> {
        self.entries.get(&id)
    }

    /// Get a mutable reference to an entry by ID.
    pub fn get_mut(&mut self, id: EntryId) -> Option<&mut BankEntry> {
        self.entries.get_mut(&id)
    }

    /// Remove an entry by ID, returning it if it existed.
    pub fn remove(&mut self, id: EntryId) -> Option<BankEntry> {
        if let Some(entry) = self.entries.remove(&id) {
            self.vector_index.remove(id);
            self.reverse_edges.remove(&id);
            self.mark_mutated();
            Some(entry)
        } else {
            None
        }
    }

    /// Query the bank for entries most similar to the given vector.
    ///
    /// Uses sparse cosine similarity — only non-zero query dimensions
    /// participate. This IS pattern completion: a partial cue activates
    /// the full stored patterns that best match.
    pub fn query_sparse(&self, query: &[Signal], top_k: usize) -> Vec<QueryResult> {
        self.vector_index.query(query, &self.entries, top_k)
    }

    /// Add a directed edge from one entry to another.
    pub fn add_edge(&mut self, from: EntryId, edge: Edge) -> Result<()> {
        let max = self.config.max_edges_per_entry;
        let entry = self
            .entries
            .get_mut(&from)
            .ok_or(DataBankError::EntryNotFound { id: from })?;
        entry.add_edge(edge, max)?;

        // Update reverse index: the target now has a back-pointer
        self.reverse_edges
            .entry(edge.target.entry)
            .or_default()
            .push((
                BankRef {
                    bank: self.id,
                    entry: from,
                },
                edge.edge_type,
            ));

        self.mark_mutated();
        Ok(())
    }

    /// Get edges from a specific entry.
    pub fn edges_from(&self, id: EntryId) -> &[Edge] {
        self.entries
            .get(&id)
            .map(|e| e.edges.as_slice())
            .unwrap_or(&[])
    }

    /// Get reverse edges pointing to an entry in this bank.
    pub fn reverse_edges(&self, id: EntryId) -> &[(BankRef, EdgeType)] {
        self.reverse_edges
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Evict the entry with the lowest eviction score.
    fn evict_lowest(&mut self, current_tick: u64) {
        let lowest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.eviction_score(current_tick))
            .map(|(&id, _)| id);

        if let Some(id) = lowest {
            self.entries.remove(&id);
            self.vector_index.remove(id);
            self.reverse_edges.remove(&id);
            log::debug!("evicted entry {:?} from bank {:?}", id, self.id);
        }
    }

    /// Check whether the bank should be flushed to disk.
    pub fn should_persist(&self, current_tick: u64) -> bool {
        if !self.dirty {
            return false;
        }
        let ticks_since = current_tick.saturating_sub(self.last_persist_tick);
        self.config
            .should_persist(self.mutations_since_persist, ticks_since)
    }

    /// Mark the bank as persisted, resetting mutation counters.
    pub fn mark_persisted(&mut self, tick: u64) {
        self.mutations_since_persist = 0;
        self.last_persist_tick = tick;
        self.dirty = false;
    }

    /// Whether the bank has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Number of entries in the bank.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the bank has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the bank's configuration.
    pub fn config(&self) -> &BankConfig {
        &self.config
    }

    /// Get an iterator over all entries.
    pub fn entries(&self) -> impl Iterator<Item = (&EntryId, &BankEntry)> {
        self.entries.iter()
    }

    /// Get the next sequence counter (for codec restore).
    pub(crate) fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Get the mutations since persist (for codec restore).
    pub(crate) fn mutations_since_persist(&self) -> u32 {
        self.mutations_since_persist
    }

    /// Get the last persist tick (for codec restore).
    pub(crate) fn last_persist_tick(&self) -> u64 {
        self.last_persist_tick
    }

    /// Get the reverse edges map (for codec).
    #[allow(dead_code)]
    pub(crate) fn reverse_edges_map(&self) -> &HashMap<EntryId, Vec<(BankRef, EdgeType)>> {
        &self.reverse_edges
    }

    /// Restore bank state from decoded fields (used by codec).
    pub(crate) fn restore(
        id: BankId,
        name: String,
        config: BankConfig,
        entries: HashMap<EntryId, BankEntry>,
        reverse_edges: HashMap<EntryId, Vec<(BankRef, EdgeType)>>,
        next_seq: u32,
        mutations_since_persist: u32,
        last_persist_tick: u64,
    ) -> Self {
        let mut bank = Self {
            id,
            config,
            name,
            entries,
            next_seq,
            vector_index: BruteForceIndex,
            reverse_edges,
            mutations_since_persist,
            last_persist_tick,
            dirty: false,
        };
        bank.vector_index.rebuild(&bank.entries);
        bank
    }

    /// Promote an entry's temperature. Returns Ok(true) if promoted.
    pub fn promote_entry(&mut self, id: EntryId) -> Result<bool> {
        let entry = self.entries.get_mut(&id)
            .ok_or(DataBankError::EntryNotFound { id })?;
        let promoted = entry.promote();
        if promoted {
            self.mark_mutated();
        }
        Ok(promoted)
    }

    /// Demote an entry's temperature. Returns Ok(true) if demoted.
    pub fn demote_entry(&mut self, id: EntryId) -> Result<bool> {
        let entry = self.entries.get_mut(&id)
            .ok_or(DataBankError::EntryNotFound { id })?;
        let demoted = entry.demote();
        if demoted {
            self.mark_mutated();
        }
        Ok(demoted)
    }

    /// Batch promote all eligible entries. Returns count promoted.
    pub fn consolidation_pass(
        &mut self,
        current_tick: u64,
        min_accesses: u32,
        min_age_ticks: u64,
    ) -> usize {
        let eligible: Vec<EntryId> = self.entries.iter()
            .filter(|(_, e)| e.promotion_eligible(current_tick, min_accesses, min_age_ticks))
            .map(|(&id, _)| id)
            .collect();
        let mut count = 0;
        for id in eligible {
            if let Some(entry) = self.entries.get_mut(&id) {
                if entry.promote() {
                    count += 1;
                }
            }
        }
        if count > 0 {
            self.mark_mutated();
        }
        count
    }

    /// Batch demote entries below confidence threshold. Returns count demoted.
    pub fn demotion_pass(&mut self, confidence_threshold: u8) -> usize {
        let eligible: Vec<EntryId> = self.entries.iter()
            .filter(|(_, e)| e.demotion_eligible(confidence_threshold))
            .map(|(&id, _)| id)
            .collect();
        let mut count = 0;
        for id in eligible {
            if let Some(entry) = self.entries.get_mut(&id) {
                if entry.demote() {
                    count += 1;
                }
            }
        }
        if count > 0 {
            self.mark_mutated();
        }
        count
    }

    /// Evict lowest-scoring entries. Returns count evicted.
    pub fn evict_n(&mut self, count: usize, current_tick: u64) -> usize {
        let mut scored: Vec<(EntryId, i64)> = self.entries.iter()
            .map(|(&id, e)| (id, e.eviction_score(current_tick)))
            .collect();
        scored.sort_by_key(|&(_, score)| score);
        let to_evict = scored.iter().take(count).map(|&(id, _)| id).collect::<Vec<_>>();
        let mut evicted = 0;
        for id in to_evict {
            if self.entries.remove(&id).is_some() {
                self.vector_index.remove(id);
                self.reverse_edges.remove(&id);
                evicted += 1;
            }
        }
        if evicted > 0 {
            self.mark_mutated();
        }
        evicted
    }

    /// Compact internal data structures after mass eviction.
    pub fn compact(&mut self) {
        self.vector_index.rebuild(&self.entries);
        // Clean up reverse edges pointing to removed entries
        let valid_ids: std::collections::HashSet<EntryId> = self.entries.keys().copied().collect();
        self.reverse_edges.retain(|id, _| valid_ids.contains(id));
    }

    fn mark_mutated(&mut self) {
        self.mutations_since_persist = self.mutations_since_persist.saturating_add(1);
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(width: u16) -> BankConfig {
        BankConfig {
            vector_width: width,
            max_entries: 10,
            max_edges_per_entry: 4,
            ..BankConfig::default()
        }
    }

    fn make_vector(width: u16) -> Vec<Signal> {
        (0..width)
            .map(|i| Signal::new(1, (i % 255) as u8 + 1))
            .collect()
    }

    fn make_bank() -> DataBank {
        let id = BankId::from_raw(1);
        DataBank::new(id, "test.bank".into(), make_config(8))
    }

    #[test]
    fn insert_and_get() {
        let mut bank = make_bank();
        let v = make_vector(8);
        let entry_id = bank.insert(v.clone(), Temperature::Hot, 0).unwrap();
        let entry = bank.get(entry_id).unwrap();
        assert_eq!(entry.vector, v);
        assert_eq!(bank.len(), 1);
    }

    #[test]
    fn insert_wrong_width_fails() {
        let mut bank = make_bank();
        let v = make_vector(16); // wrong width
        let result = bank.insert(v, Temperature::Hot, 0);
        assert!(result.is_err());
    }

    #[test]
    fn remove_entry() {
        let mut bank = make_bank();
        let entry_id = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        assert_eq!(bank.len(), 1);
        let removed = bank.remove(entry_id);
        assert!(removed.is_some());
        assert_eq!(bank.len(), 0);
        assert!(bank.get(entry_id).is_none());
    }

    #[test]
    fn eviction_on_capacity() {
        let mut bank = make_bank(); // max_entries = 10
        for i in 0..10 {
            bank.insert(make_vector(8), Temperature::Hot, i).unwrap();
        }
        assert_eq!(bank.len(), 10);
        // 11th insert triggers eviction
        let result = bank.insert(make_vector(8), Temperature::Hot, 100);
        assert!(result.is_ok());
        assert_eq!(bank.len(), 10); // still 10 after eviction + insert
    }

    #[test]
    fn query_sparse_returns_results() {
        let mut bank = make_bank();
        let v = make_vector(8);
        bank.insert(v.clone(), Temperature::Hot, 0).unwrap();

        let results = bank.query_sparse(&v, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 200); // should be near-identical match
    }

    #[test]
    fn add_edge_and_retrieve() {
        let mut bank = make_bank();
        let id1 = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();

        let target = BankRef {
            bank: BankId::from_raw(2),
            entry: EntryId::from_raw(999),
        };
        let edge = Edge {
            edge_type: EdgeType::RelatedTo,
            target,
            weight: 200,
            created_tick: 0,
        };
        bank.add_edge(id1, edge).unwrap();

        let edges = bank.edges_from(id1);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, EdgeType::RelatedTo);
        assert_eq!(edges[0].weight, 200);
    }

    #[test]
    fn dirty_tracking() {
        let mut bank = make_bank();
        assert!(!bank.is_dirty());
        bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        assert!(bank.is_dirty());
        bank.mark_persisted(10);
        assert!(!bank.is_dirty());
    }

    #[test]
    fn promote_and_demote_entry() {
        let mut bank = make_bank();
        let id = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        assert!(bank.promote_entry(id).unwrap());
        assert_eq!(bank.get(id).unwrap().temperature, Temperature::Warm);
        assert!(bank.demote_entry(id).unwrap());
        assert_eq!(bank.get(id).unwrap().temperature, Temperature::Hot);
    }

    #[test]
    fn consolidation_pass_promotes_eligible() {
        let mut bank = make_bank();
        let id1 = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        let id2 = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        // Give id1 enough accesses
        for _ in 0..5 {
            bank.get_mut(id1).unwrap().touch(50);
        }
        // id2 stays at 0 accesses
        let promoted = bank.consolidation_pass(200, 5, 100);
        assert_eq!(promoted, 1);
        assert_eq!(bank.get(id1).unwrap().temperature, Temperature::Warm);
        assert_eq!(bank.get(id2).unwrap().temperature, Temperature::Hot);
    }

    #[test]
    fn demotion_pass_demotes_low_confidence() {
        let mut bank = make_bank();
        let id = bank.insert(make_vector(8), Temperature::Warm, 0).unwrap();
        bank.get_mut(id).unwrap().confidence = 30;
        let demoted = bank.demotion_pass(50);
        assert_eq!(demoted, 1);
        assert_eq!(bank.get(id).unwrap().temperature, Temperature::Hot);
    }

    #[test]
    fn evict_n_removes_lowest() {
        let mut bank = make_bank();
        // Insert 5 entries at different ticks for different recency
        let mut ids = Vec::new();
        for i in 0..5 {
            ids.push(bank.insert(make_vector(8), Temperature::Hot, i as u64).unwrap());
        }
        assert_eq!(bank.len(), 5);
        let evicted = bank.evict_n(2, 100);
        assert_eq!(evicted, 2);
        assert_eq!(bank.len(), 3);
    }

    #[test]
    fn compact_rebuilds_index() {
        let mut bank = make_bank();
        let id1 = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        let _id2 = bank.insert(make_vector(8), Temperature::Hot, 0).unwrap();
        bank.remove(id1);
        bank.compact();
        assert_eq!(bank.len(), 1);
        // Query should still work after compact
        let results = bank.query_sparse(&make_vector(8), 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn should_persist_logic() {
        let mut bank = make_bank();
        assert!(!bank.should_persist(0));
        // Insert enough to trigger mutation threshold
        for i in 0..100 {
            bank.insert(make_vector(8), Temperature::Hot, i)
                .unwrap_or_else(|_| EntryId::from_raw(0));
        }
        assert!(bank.should_persist(0));
    }
}
