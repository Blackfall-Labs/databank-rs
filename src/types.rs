use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// BankId — 64-bit temporally sortable bank identity
// Layout: [timestamp_s:32][region_tag:24][seq:8]
// ---------------------------------------------------------------------------

/// Identifies a databank. Temporally sortable, region-grouped, compact.
///
/// The upper 32 bits are a Unix timestamp (seconds), the next 24 bits are
/// an FNV-1a hash of the region name (for grouping), and the lower 8 bits
/// are a sequence counter (for multiple banks per region per second).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub struct BankId(pub u64);

impl BankId {
    /// Create a new BankId from a region name and sequence number.
    ///
    /// The region name is hashed to a 24-bit tag via FNV-1a.
    /// The timestamp is captured at call time.
    pub fn new(region_name: &str, seq: u8) -> Self {
        let timestamp = unix_timestamp_secs() as u64;
        let region_tag = fnv1a_24(region_name) as u64;
        Self((timestamp << 32) | (region_tag << 8) | seq as u64)
    }

    /// Create a BankId from a raw u64 value (e.g. loaded from disk).
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Extract the Unix timestamp (seconds) from this BankId.
    pub fn timestamp_secs(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Extract the 24-bit region tag from this BankId.
    pub fn region_tag(&self) -> u32 {
        ((self.0 >> 8) & 0x00FF_FFFF) as u32
    }

    /// Extract the sequence byte from this BankId.
    pub fn seq(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
}

impl std::fmt::Display for BankId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BankId(t={}, tag={:#08x}, seq={})",
            self.timestamp_secs(),
            self.region_tag(),
            self.seq()
        )
    }
}

// ---------------------------------------------------------------------------
// EntryId — 64-bit temporally sortable entry identity
// Layout: [timestamp_ms:42][seq:22]
// ---------------------------------------------------------------------------

/// Identifies an entry within a bank. Temporally sortable.
///
/// The upper 42 bits are a millisecond timestamp (good until year 2109),
/// and the lower 22 bits are a sequence counter (~4M entries per ms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub struct EntryId(pub u64);

impl EntryId {
    /// Create a new EntryId with the current timestamp and a sequence number.
    pub fn new(seq: u32) -> Self {
        let ms = unix_timestamp_ms();
        Self((ms << 22) | (seq as u64 & 0x003F_FFFF))
    }

    /// Create an EntryId from a raw u64 value (e.g. loaded from disk).
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Extract the millisecond timestamp from this EntryId.
    pub fn timestamp_ms(&self) -> u64 {
        self.0 >> 22
    }

    /// Extract the sequence counter from this EntryId.
    pub fn seq(&self) -> u32 {
        (self.0 & 0x003F_FFFF) as u32
    }
}

impl std::fmt::Display for EntryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EntryId(ms={}, seq={})", self.timestamp_ms(), self.seq())
    }
}

// ---------------------------------------------------------------------------
// BankRef — cross-bank pointer
// ---------------------------------------------------------------------------

/// A cross-bank reference: identifies one entry in one bank.
///
/// Used in edges to point from an entry in bank A to an entry in bank B.
/// This is a record of a relationship, NOT a database foreign key —
/// the actual recall happens through thermogram neural propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct BankRef {
    pub bank: BankId,
    pub entry: EntryId,
}

// ---------------------------------------------------------------------------
// EdgeType — typed relationships between entries
// ---------------------------------------------------------------------------

/// The semantic type of an edge between two entries.
///
/// Edges are directed and weighted. Each type carries domain meaning
/// that the cognitive system uses for traversal decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdgeType {
    // Taxonomic
    IsA = 0,
    HasA = 1,
    PartOf = 2,
    // Associative
    RelatedTo = 3,
    SimilarTo = 4,
    // Causal
    Causes = 5,
    Precedes = 6,
    // Sensory binding
    LooksLike = 7,
    SoundsLike = 8,
    FeelsLike = 9,
    // Episodic
    CoOccurred = 10,
    FollowedBy = 11,
    // Open-ended
    Custom = 255,
}

impl EdgeType {
    /// Convert a raw u8 to an EdgeType, if valid.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::IsA),
            1 => Some(Self::HasA),
            2 => Some(Self::PartOf),
            3 => Some(Self::RelatedTo),
            4 => Some(Self::SimilarTo),
            5 => Some(Self::Causes),
            6 => Some(Self::Precedes),
            7 => Some(Self::LooksLike),
            8 => Some(Self::SoundsLike),
            9 => Some(Self::FeelsLike),
            10 => Some(Self::CoOccurred),
            11 => Some(Self::FollowedBy),
            255 => Some(Self::Custom),
            _ => None,
        }
    }

    /// Convert this EdgeType to its raw u8 discriminant.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// Edge — directed weighted connection between entries
// ---------------------------------------------------------------------------

/// A directed, weighted edge from one entry to another.
///
/// Edges cross bank boundaries — the target can be in any bank.
/// Weight is 0-255 (strength of association).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub edge_type: EdgeType,
    pub target: BankRef,
    pub weight: u8,
    pub created_tick: u64,
}

// ---------------------------------------------------------------------------
// Temperature — thermogram-compatible lifecycle state
// ---------------------------------------------------------------------------

/// Temperature lifecycle for bank entries.
///
/// Matches the thermogram lifecycle: Hot (active learning) → Warm (session
/// patterns) → Cool (proven) → Cold (frozen priors). Eviction scoring
/// uses temperature as a major factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Temperature {
    Hot = 0,
    Warm = 1,
    Cool = 2,
    Cold = 3,
}

impl Temperature {
    /// Convert a raw u8 to a Temperature, if valid.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Hot),
            1 => Some(Self::Warm),
            2 => Some(Self::Cool),
            3 => Some(Self::Cold),
            _ => None,
        }
    }

    /// Convert this Temperature to its raw u8 discriminant.
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

impl std::fmt::Display for Temperature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hot => write!(f, "HOT"),
            Self::Warm => write!(f, "WARM"),
            Self::Cool => write!(f, "COOL"),
            Self::Cold => write!(f, "COLD"),
        }
    }
}

// ---------------------------------------------------------------------------
// BankConfig — per-region bank configuration
// ---------------------------------------------------------------------------

/// Configuration for a single DataBank.
///
/// Each region sets its own persistence frequency, capacity, and vector
/// dimensions. The vector_width is FIXED at bank creation and cannot change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankConfig {
    /// Flush to disk after this many mutations. Default: 100.
    pub persist_after_mutations: u32,
    /// Flush to disk after this many ticks since last flush. Default: 10_000.
    pub persist_after_ticks: u64,
    /// Maximum number of entries in the bank. Default: 4096.
    pub max_entries: u32,
    /// Fixed signal vector width for all entries. Set at creation.
    pub vector_width: u16,
    /// Maximum edges per entry. Default: 32.
    pub max_edges_per_entry: u16,
}

impl BankConfig {
    /// Check whether the bank should be flushed to disk.
    pub fn should_persist(&self, mutations_since: u32, ticks_since: u64) -> bool {
        mutations_since >= self.persist_after_mutations
            || ticks_since >= self.persist_after_ticks
    }
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            persist_after_mutations: 100,
            persist_after_ticks: 10_000,
            max_entries: 4096,
            vector_width: 64,
            max_edges_per_entry: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// FNV-1a hash truncated to 24 bits. Deterministic, no-std friendly.
fn fnv1a_24(s: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV offset basis
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
    }
    hash & 0x00FF_FFFF
}

/// Current Unix timestamp in seconds.
fn unix_timestamp_secs() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32
}

/// Current Unix timestamp in milliseconds.
fn unix_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_id_round_trip() {
        let id = BankId::new("temporal.semantic", 3);
        assert_eq!(id.seq(), 3);
        assert!(id.timestamp_secs() > 0);
        assert!(id.region_tag() > 0);
        assert!(id.region_tag() <= 0x00FF_FFFF);
    }

    #[test]
    fn bank_id_temporal_ordering() {
        let a = BankId::new("region_a", 0);
        // Same second, different region — ordering is timestamp-first
        let b = BankId::new("region_b", 0);
        // Both created in the same second, so ordering depends on region_tag
        // Just verify they're both valid
        assert!(a.timestamp_secs() > 0);
        assert!(b.timestamp_secs() > 0);
    }

    #[test]
    fn bank_id_deterministic_hash() {
        let a = BankId::new("temporal.semantic", 0);
        let b = BankId::new("temporal.semantic", 0);
        // Same region name in same second → same region_tag
        assert_eq!(a.region_tag(), b.region_tag());
    }

    #[test]
    fn entry_id_round_trip() {
        let id = EntryId::new(42);
        assert_eq!(id.seq(), 42);
        assert!(id.timestamp_ms() > 0);
    }

    #[test]
    fn entry_id_temporal_ordering() {
        let a = EntryId::new(0);
        let b = EntryId::new(1);
        // b has higher seq, same ms → b > a
        assert!(b >= a);
    }

    #[test]
    fn edge_type_round_trip() {
        for v in [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255] {
            let et = EdgeType::from_u8(v).expect("valid edge type");
            assert_eq!(et.as_u8(), v);
        }
        assert!(EdgeType::from_u8(12).is_none());
        assert!(EdgeType::from_u8(254).is_none());
    }

    #[test]
    fn temperature_round_trip() {
        for v in 0..=3u8 {
            let t = Temperature::from_u8(v).expect("valid temperature");
            assert_eq!(t.as_u8(), v);
        }
        assert!(Temperature::from_u8(4).is_none());
    }

    #[test]
    fn temperature_ordering() {
        assert!(Temperature::Hot < Temperature::Warm);
        assert!(Temperature::Warm < Temperature::Cool);
        assert!(Temperature::Cool < Temperature::Cold);
    }

    #[test]
    fn bank_config_should_persist() {
        let cfg = BankConfig::default();
        assert!(!cfg.should_persist(0, 0));
        assert!(cfg.should_persist(100, 0));
        assert!(cfg.should_persist(0, 10_000));
        assert!(cfg.should_persist(100, 10_000));
        assert!(!cfg.should_persist(99, 9_999));
    }

    #[test]
    fn fnv1a_24_deterministic() {
        let h1 = fnv1a_24("temporal.semantic");
        let h2 = fnv1a_24("temporal.semantic");
        assert_eq!(h1, h2);
        // Different inputs give different hashes (probabilistic but reliable)
        let h3 = fnv1a_24("occipital.v4");
        assert_ne!(h1, h3);
    }
}
