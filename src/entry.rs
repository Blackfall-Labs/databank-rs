use serde::{Deserialize, Serialize};
use ternary_signal::Signal;

use crate::error::{DataBankError, Result};
use crate::types::{BankId, BankRef, Edge, EntryId, Temperature};

/// A single entry in a databank — one fragment of a distributed concept.
///
/// Each entry stores a signal vector (the representational pattern), typed
/// edges to related entries (possibly in other banks), and lifecycle metadata.
/// The vector width is fixed per bank and validated on insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankEntry {
    /// Unique entry identifier (temporally sortable).
    pub id: EntryId,
    /// The representational signal vector. Fixed width per bank.
    pub vector: Vec<Signal>,
    /// Typed, weighted edges to other entries (cross-bank allowed).
    pub edges: Vec<Edge>,
    /// Which bank originally created this entry.
    pub origin: BankId,
    /// Thermogram-compatible temperature lifecycle state.
    pub temperature: Temperature,
    /// Tick when this entry was created.
    pub created_tick: u64,
    /// Tick when this entry was last accessed (read or touched).
    pub last_accessed_tick: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u32,
    /// Confidence score (0-255). Higher = more reliable.
    pub confidence: u8,
    /// Human-readable label for debugging/introspection. Optional.
    pub debug_tag: Option<String>,
    /// CRC32 checksum of the vector data for integrity verification.
    pub checksum: u32,
}

impl BankEntry {
    /// Create a new entry with the given vector and metadata.
    ///
    /// The checksum is computed automatically from the vector data.
    pub fn new(
        id: EntryId,
        vector: Vec<Signal>,
        origin: BankId,
        temperature: Temperature,
        tick: u64,
    ) -> Self {
        let checksum = compute_vector_checksum(&vector);
        Self {
            id,
            vector,
            edges: Vec::new(),
            origin,
            temperature,
            created_tick: tick,
            last_accessed_tick: tick,
            access_count: 0,
            confidence: 128, // neutral default
            debug_tag: None,
            checksum,
        }
    }

    /// Record an access: increment count and update last-accessed tick.
    pub fn touch(&mut self, tick: u64) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_accessed_tick = tick;
    }

    /// Add a directed edge from this entry to another.
    ///
    /// Returns an error if the entry already has `max` edges.
    pub fn add_edge(&mut self, edge: Edge, max: u16) -> Result<()> {
        if self.edges.len() >= max as usize {
            return Err(DataBankError::EdgeLimitReached { max });
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Remove all edges pointing to a specific target.
    pub fn remove_edges_to(&mut self, target: BankRef) {
        self.edges.retain(|e| e.target != target);
    }

    /// Compute a hybrid eviction score. Lower = more evictable.
    ///
    /// Formula combines temperature (Cold entries are valuable), access
    /// frequency, confidence, and recency. The kernel calls this when the
    /// bank is at capacity and needs to make room.
    pub fn eviction_score(&self, current_tick: u64) -> i64 {
        let temperature_weight: i64 = match self.temperature {
            Temperature::Hot => 10,
            Temperature::Warm => 50,
            Temperature::Cool => 200,
            Temperature::Cold => 1000,
        };

        let recency = if current_tick > self.last_accessed_tick {
            let age = current_tick - self.last_accessed_tick;
            // Newer entries get higher scores; cap at 500
            500i64.saturating_sub(age.min(500) as i64)
        } else {
            500
        };

        let access = (self.access_count as i64).min(500);
        let conf = self.confidence as i64;

        temperature_weight + recency + access + conf
    }

    /// Promote temperature one step: Hot->Warm, Warm->Cool, Cool->Cold.
    /// Returns true if promoted, false if already Cold.
    pub fn promote(&mut self) -> bool {
        match self.temperature {
            Temperature::Hot => { self.temperature = Temperature::Warm; true }
            Temperature::Warm => { self.temperature = Temperature::Cool; true }
            Temperature::Cool => { self.temperature = Temperature::Cold; true }
            Temperature::Cold => false,
        }
    }

    /// Demote temperature one step: Cold->Cool, Cool->Warm, Warm->Hot.
    /// Returns true if demoted, false if already Hot.
    pub fn demote(&mut self) -> bool {
        match self.temperature {
            Temperature::Cold => { self.temperature = Temperature::Cool; true }
            Temperature::Cool => { self.temperature = Temperature::Warm; true }
            Temperature::Warm => { self.temperature = Temperature::Hot; true }
            Temperature::Hot => false,
        }
    }

    /// Check if this entry qualifies for promotion based on access patterns.
    /// Criteria: access_count >= threshold AND age >= min_age ticks.
    pub fn promotion_eligible(&self, current_tick: u64, min_accesses: u32, min_age_ticks: u64) -> bool {
        if self.temperature == Temperature::Cold {
            return false; // already at max
        }
        let age = current_tick.saturating_sub(self.created_tick);
        self.access_count >= min_accesses && age >= min_age_ticks
    }

    /// Check if this entry should be demoted (confidence below threshold).
    pub fn demotion_eligible(&self, confidence_threshold: u8) -> bool {
        if self.temperature == Temperature::Hot {
            return false; // already at min
        }
        self.confidence < confidence_threshold
    }

    /// Recompute the CRC32 checksum from the current vector data.
    pub fn compute_checksum(&self) -> u32 {
        compute_vector_checksum(&self.vector)
    }

    /// Verify that the stored checksum matches the vector data.
    pub fn validate(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

/// Compute CRC32 checksum over raw signal bytes (polarity + magnitude).
fn compute_vector_checksum(vector: &[Signal]) -> u32 {
    crc32fast_compute(vector)
}

/// CRC32 over signal vector bytes: [polarity, magnitude, polarity, magnitude, ...]
fn crc32fast_compute(signals: &[Signal]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for s in signals {
        crc = crc32_update(crc, s.polarity as u8);
        crc = crc32_update(crc, s.magnitude);
    }
    crc ^ 0xFFFF_FFFF
}

/// CRC32 single-byte update (IEEE polynomial, same as crc32fast).
fn crc32_update(crc: u32, byte: u8) -> u32 {
    let mut c = crc ^ (byte as u32);
    for _ in 0..8 {
        if c & 1 != 0 {
            c = (c >> 1) ^ 0xEDB8_8320;
        } else {
            c >>= 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BankId;

    fn make_entry(width: usize, tick: u64) -> BankEntry {
        let vector: Vec<Signal> = (0..width)
            .map(|i| Signal::new(1, (i % 255) as u8 + 1))
            .collect();
        let bank_id = BankId::from_raw(1);
        BankEntry::new(EntryId::new(0), vector, bank_id, Temperature::Hot, tick)
    }

    #[test]
    fn new_entry_has_valid_checksum() {
        let entry = make_entry(64, 0);
        assert!(entry.validate());
    }

    #[test]
    fn touch_increments_access() {
        let mut entry = make_entry(32, 0);
        assert_eq!(entry.access_count, 0);
        entry.touch(10);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_accessed_tick, 10);
        entry.touch(20);
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn add_edge_respects_limit() {
        let mut entry = make_entry(32, 0);
        let edge = Edge {
            edge_type: crate::types::EdgeType::RelatedTo,
            target: BankRef {
                bank: BankId::from_raw(2),
                entry: EntryId::from_raw(100),
            },
            weight: 200,
            created_tick: 0,
        };
        // Add up to limit
        for _ in 0..3 {
            entry.add_edge(edge, 3).unwrap();
        }
        // One more should fail
        let result = entry.add_edge(edge, 3);
        assert!(result.is_err());
    }

    #[test]
    fn remove_edges_to_target() {
        let mut entry = make_entry(32, 0);
        let target = BankRef {
            bank: BankId::from_raw(2),
            entry: EntryId::from_raw(100),
        };
        let other = BankRef {
            bank: BankId::from_raw(3),
            entry: EntryId::from_raw(200),
        };
        entry
            .add_edge(
                Edge {
                    edge_type: crate::types::EdgeType::IsA,
                    target,
                    weight: 100,
                    created_tick: 0,
                },
                32,
            )
            .unwrap();
        entry
            .add_edge(
                Edge {
                    edge_type: crate::types::EdgeType::HasA,
                    target: other,
                    weight: 50,
                    created_tick: 0,
                },
                32,
            )
            .unwrap();
        assert_eq!(entry.edges.len(), 2);
        entry.remove_edges_to(target);
        assert_eq!(entry.edges.len(), 1);
        assert_eq!(entry.edges[0].target, other);
    }

    #[test]
    fn eviction_score_cold_higher_than_hot() {
        let mut hot = make_entry(32, 0);
        hot.temperature = Temperature::Hot;
        let mut cold = make_entry(32, 0);
        cold.temperature = Temperature::Cold;
        // Cold entries should have higher score (harder to evict)
        assert!(cold.eviction_score(100) > hot.eviction_score(100));
    }

    #[test]
    fn eviction_score_recent_higher() {
        let mut recent = make_entry(32, 0);
        recent.last_accessed_tick = 90;
        let mut old = make_entry(32, 0);
        old.last_accessed_tick = 0;
        assert!(recent.eviction_score(100) > old.eviction_score(100));
    }

    #[test]
    fn promote_hot_to_cold() {
        let mut entry = make_entry(32, 0);
        assert_eq!(entry.temperature, Temperature::Hot);
        assert!(entry.promote());
        assert_eq!(entry.temperature, Temperature::Warm);
        assert!(entry.promote());
        assert_eq!(entry.temperature, Temperature::Cool);
        assert!(entry.promote());
        assert_eq!(entry.temperature, Temperature::Cold);
        assert!(!entry.promote()); // already Cold
    }

    #[test]
    fn demote_cold_to_hot() {
        let mut entry = make_entry(32, 0);
        entry.temperature = Temperature::Cold;
        assert!(entry.demote());
        assert_eq!(entry.temperature, Temperature::Cool);
        assert!(entry.demote());
        assert_eq!(entry.temperature, Temperature::Warm);
        assert!(entry.demote());
        assert_eq!(entry.temperature, Temperature::Hot);
        assert!(!entry.demote()); // already Hot
    }

    #[test]
    fn promotion_eligibility() {
        let mut entry = make_entry(32, 0);
        entry.access_count = 10;
        // Not old enough
        assert!(!entry.promotion_eligible(5, 10, 100));
        // Old enough and accessed enough
        assert!(entry.promotion_eligible(200, 10, 100));
        // Not accessed enough
        assert!(!entry.promotion_eligible(200, 20, 100));
        // Already Cold
        entry.temperature = Temperature::Cold;
        assert!(!entry.promotion_eligible(200, 10, 100));
    }

    #[test]
    fn demotion_eligibility() {
        let mut entry = make_entry(32, 0);
        entry.temperature = Temperature::Warm;
        entry.confidence = 50;
        assert!(entry.demotion_eligible(100)); // below threshold
        assert!(!entry.demotion_eligible(30)); // above threshold
        // Already Hot
        entry.temperature = Temperature::Hot;
        assert!(!entry.demotion_eligible(255));
    }

    #[test]
    fn checksum_detects_corruption() {
        let mut entry = make_entry(32, 0);
        assert!(entry.validate());
        // Corrupt one signal
        entry.vector[0] = Signal::new(-1, 255);
        assert!(!entry.validate());
    }
}
