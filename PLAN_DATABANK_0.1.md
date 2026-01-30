# PLAN: databank-rs 0.1.0 — Distributed Representational Memory

**Date:** 2026-01-29
**Crate:** `databank-rs` (`E:\repos\blackfall-labs\databank-rs`)
**Status:** Draft — awaiting operator approval

---

## What This Is

A jar. What does the brain know about a jar?

- **Occipital bank:** cylindrical shape, glass material, transparent, has a lid
- **Temporal-auditory bank:** the phoneme sequence /dʒɑːr/, rhymes with "car", "star"
- **Temporal-semantic bank:** it's a container, holds things, made of glass or ceramic, fragile, has a mouth
- **Parietal-spatial bank:** typical size ~15cm tall, found in kitchens, on shelves
- **Frontal-motor bank:** grip with whole hand, twist lid counter-clockwise to open
- **Frontal-ofc bank:** associated with preservation, grandma's kitchen, comfort

None of these fragments alone IS the concept "jar." The concept IS the distributed activation across all of them, bound together by thermogram connections that fire cross-bank retrieval. Activate "jar" in any one bank, and pattern completion cascades through the thermogram wiring to retrieve the full distributed representation.

**Thermograms** store the wiring (which neurons connect to which, with what weight). **Databanks** store the content those connections point to (the actual representational fragments). Thermograms are the connectome. Databanks are the engrams.

This replaces SQLite (CartridgeMux) from v2. No SQL. No tables. No schemas. Just signal patterns with typed edges and attributes, persisted as binary `.bank` files.

---

## The Concept of a Concept

A concept is not an entry in a database. A concept is a **resonance pattern** across distributed banks.

```
                         "jar"
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │ occipital  │ │temporal│ │ frontal   │
        │   .bank    │ │ .bank │ │   .bank   │
        ├────────────┤ ├───────┤ ├───────────┤
        │ entry #47  │ │ #203  │ │ #89       │
        │ [shape:    │ │[phon: │ │[motor:    │
        │  cylinder, │ │ dʒɑːr,│ │ grip,     │
        │  glass,    │ │ IS_A: │ │ twist_lid,│
        │  transp.]  │ │ noun] │ │ careful]  │
        └─────┬──────┘ └───┬───┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    thermogram wiring
                    (cross-bank edges
                     fire retrieval)
```

**Encoding (hippocampus):** When the brain encounters a new jar, hippocampal firmware binds the distributed fragments. It writes entries to each relevant bank and creates cross-bank edges linking them. The thermogram connections between regions ensure that future activation of any fragment cascades to all others.

**Recall (pattern completion):** Firmware in one region yields a similarity query against its local bank. The results activate that region's neurons. Thermogram connections propagate activation to other regions. Those regions query THEIR banks. The full concept assembles across the workspace.

**Evolution:** When a new region forms (via `arch.GROW` + `arch.WIRE`), it gets its own bank. New sensory modalities, new specializations, new functional clusters — each gets storage for its own representational fragments. The bank system grows organically with the brain.

---

## Architecture

### Core Types

```rust
/// 64-bit temporally unique bank identifier (see Q4 for full spec).
/// Layout: [timestamp_s: 32][region_tag: 24][seq: 8]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub struct BankId(pub u64);

/// 64-bit temporally unique entry identifier (see Q4 for full spec).
/// Layout: [timestamp_ms: 42][seq: 22]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub struct EntryId(pub u64);

/// Cross-bank reference. 16 bytes total. Compact enough for 32 edges
/// per entry at only 512 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct BankRef {
    pub bank: BankId,    // 8 bytes
    pub entry: EntryId,  // 8 bytes
}

/// Typed relationship between entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdgeType {
    // Taxonomic
    IsA       = 0x01,  // jar IS_A container
    HasA      = 0x02,  // jar HAS_A lid
    PartOf    = 0x03,  // lid PART_OF jar

    // Associative
    RelatedTo = 0x10,  // jar RELATED_TO kitchen
    SimilarTo = 0x11,  // jar SIMILAR_TO bottle
    OppositeOf= 0x12,  // open OPPOSITE_OF closed

    // Causal
    Causes    = 0x20,  // drop CAUSES break
    Precedes  = 0x21,  // grip PRECEDES twist
    Enables   = 0x22,  // open ENABLES pour

    // Sensory binding
    LooksLike = 0x30,  // cross-bank: semantic → visual fragment
    SoundsLike= 0x31,  // cross-bank: semantic → auditory fragment
    FeelsLike = 0x32,  // cross-bank: semantic → motor/tactile fragment

    // Episodic
    CoOccurred= 0x40,  // appeared together in an episode
    FollowedBy= 0x41,  // temporal sequence in episode

    // Modality-specific (extension range)
    Custom    = 0xFF,  // for future/region-specific edge types
}

/// An edge to another entry (local or cross-bank).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub edge_type: EdgeType,
    pub target: BankRef,         // can point to same bank or different bank
    pub weight: u8,              // 0-255 strength
    pub created_tick: u64,
}

/// Temperature lifecycle — mirrors thermogram temperatures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Temperature {
    Hot  = 0,   // just encoded, unproven, high plasticity
    Warm = 1,   // repeated access, moderate stability
    Cool = 2,   // proven useful, low plasticity
    Cold = 3,   // consolidated, frozen — only offline mutation
}
```

### BankEntry

The fundamental unit of stored knowledge. Each entry is a fragment of a distributed concept.

```rust
/// A single representational fragment stored in a bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankEntry {
    /// Unique ID within this bank.
    pub id: EntryId,

    /// The activation pattern — the actual representation.
    /// This is what similarity search matches against.
    /// Signal vectors: each Signal is 2 bytes (polarity:i8 + magnitude:u8).
    pub vector: Vec<Signal>,

    /// Typed edges to other entries (local or cross-bank).
    pub edges: Vec<Edge>,

    /// Which bank (region) created this entry.
    pub origin: BankId,

    /// Lifecycle temperature.
    pub temperature: Temperature,

    /// Tick when this entry was created.
    pub created_tick: u64,

    /// Tick when this entry was last accessed (for LRU/decay).
    pub last_accessed_tick: u64,

    /// How many times this entry has been accessed (for importance).
    pub access_count: u32,

    /// Confidence in this entry's representation (0-255).
    /// Increases with successful recall, decreases with contradictory input.
    pub confidence: u8,

    /// Optional human-readable tag (for debugging/introspection, not computation).
    /// The brain does NOT use this for anything. It's for us.
    pub debug_tag: Option<String>,

    /// CRC32 over vector + edges (integrity check).
    pub checksum: u32,
}
```

**Size estimate:** A typical entry with a 64-signal vector, 5 edges, and minimal metadata:
- Vector: 64 × 2 bytes = 128 bytes
- Edges: 5 × ~20 bytes = 100 bytes
- Metadata: ~50 bytes
- **Total: ~280 bytes per entry**
- A bank with 10,000 entries ≈ 2.8MB in memory. Manageable.

### DataBank

One bank per region/modality. In-memory, periodically persisted.

```rust
/// A single databank — one region's representational memory.
pub struct DataBank {
    /// Bank identity.
    pub id: BankId,

    /// Per-region configuration (persistence frequency, capacity, vector width).
    config: BankConfig,

    /// Human-readable name, stored once (for debug/introspection).
    pub name: String,

    /// All entries, indexed by EntryId.
    entries: HashMap<EntryId, BankEntry>,

    /// Next entry sequence counter (for EntryId generation).
    next_seq: u32,

    /// Vector similarity index for fast recall.
    /// Rebuilt on load, updated incrementally on insert.
    vector_index: VectorIndex,

    /// Reverse edge index: "who points to me?"
    /// Enables backward traversal for pattern completion.
    reverse_edges: HashMap<EntryId, Vec<(BankRef, EdgeType)>>,

    /// Mutation counter since last persist.
    mutations_since_persist: u32,

    /// Tick of last persistence.
    last_persist_tick: u64,

    /// Whether the bank has unsaved changes.
    dirty: bool,
}
```

### BankCluster

The brain's complete distributed memory — all banks together.

```rust
/// All databanks in the system, indexed by BankId.
/// Each region registers its bank(s) here.
pub struct BankCluster {
    banks: HashMap<BankId, DataBank>,

    /// Base directory for .bank file persistence.
    persist_dir: PathBuf,
}

impl BankCluster {
    /// Create or get a bank. New regions call this to get storage.
    pub fn get_or_create(&mut self, id: BankId, config: BankConfig) -> &mut DataBank;

    /// Cross-bank similarity query: search across ALL banks.
    pub fn query_all(&self, vector: &[Signal], top_k: usize) -> Vec<(BankRef, f32)>;

    /// Cross-bank edge traversal: follow edges across bank boundaries.
    pub fn traverse(&self, start: BankRef, edge_type: EdgeType, depth: u8) -> Vec<BankRef>;

    /// Persist all dirty banks.
    pub fn flush_all(&self) -> Result<()>;

    /// Load all banks from persist directory.
    pub fn load_all(dir: &Path) -> Result<Self>;
}
```

---

## Vector Similarity Index

Fast recall requires an index over entry vectors. Options for 0.1:

### Brute-Force (0.1.0)

For banks under ~50,000 entries, brute-force cosine similarity over Signal vectors is sufficient:

```rust
/// Cosine similarity between two Signal vectors.
/// Uses integer arithmetic only (ASTRO_004 compliant).
/// Returns similarity as i32 scaled ×256 (higher = more similar).
pub fn cosine_similarity(a: &[Signal], b: &[Signal]) -> i32 {
    // dot = Σ (a.polarity * a.magnitude * b.polarity * b.magnitude)
    // norm_a = Σ (a.magnitude²)
    // norm_b = Σ (b.magnitude²)
    // sim = dot * 256 / sqrt(norm_a * norm_b)
    // Integer sqrt via Newton's method
}
```

**Performance:** 50,000 entries × 64 signals × ~4 ops = ~12.8M ops. At 100M ops/sec interpreter throughput: **~128ms**. Acceptable for non-real-time recall. Real-time recall (during cognitive tick) limits to ~5,000 entries per bank at <13ms.

### IVF or LSH (0.2+)

For larger banks, add inverted file index or locality-sensitive hashing. Deferred — 0.1 proves the concept with brute-force.

---

## Binary Format: .bank v1

```
HEADER (32 bytes)
  Offset  Size  Field
  0x00    4     Magic: "BANK" (0x42414E4B)
  0x04    2     Format version: u16 LE (1)
  0x06    2     Flags: u16 LE
  0x08    4     Entry count: u32 LE
  0x0C    4     Edge table offset: u32 LE
  0x10    4     Vector index offset: u32 LE (0 if no index stored)
  0x14    8     Checksum: xxhash64 over payload
  0x1C    4     Reserved

BANK METADATA (32 bytes)
  [BankId: u64 LE]                    // 8 bytes — compact temporal ID
  [Vector width: u16 LE]              // 2 bytes
  [Max entries: u32 LE]               // 4 bytes
  [Next entry ID: u64 LE]             // 8 bytes
  [Name length: u16 LE]               // 2 bytes — human-readable name (stored once)
  [Name: UTF-8 bytes]                 // variable — "occipital.v4", for debugging
  [Padding to 8-byte alignment]

ENTRY TABLE (variable per entry)
  [EntryId: u64 LE]
  [Vector length: u16 LE]
  [Vector: N × 2 bytes (Signal: polarity:i8, magnitude:u8)]
  [Edge count: u16 LE]
  [Edges: per-edge encoding below]
  [Temperature: u8]
  [Confidence: u8]
  [Created tick: u64 LE]
  [Last accessed tick: u64 LE]
  [Access count: u32 LE]
  [Origin BankId: u64 LE]
  [Debug tag length: u16 LE (0 if None)]
  [Debug tag: UTF-8 bytes (if present)]
  [Entry checksum: u32 LE (CRC32 over this entry's bytes)]

EDGE ENCODING (per edge, 26 bytes fixed)
  [EdgeType: u8]
  [Target BankId: u64 LE]
  [Target EntryId: u64 LE]
  [Weight: u8]
  [Created tick: u64 LE]
```

**Persistence model:**
- In-memory is authoritative. `.bank` files are snapshots.
- Persist every N mutations (configurable, default 100) or on explicit flush.
- Atomic writes: encode to buffer → temp file → fsync → rename. No partial writes.
- On load: read `.bank` file → rebuild vector index → ready.
- Delta journal (0.2+): append mutations between full persists for crash recovery.

---

## DomainOp Integration (Ternsig Extension)

New extension: **bank** (0x000B) — or DomainOps yielded by existing extensions.

For 0.1, databank-rs is a library crate. Ternsig integration (DomainOps, extension opcodes) comes in 0.2 when the data model is proven. In 0.1, the host (Astromind v3 kernel) calls databank-rs directly from Rust when fulfilling existing DomainOps.

**Future DomainOps (0.2 scope):**

```rust
/// Query bank by vector similarity.
BankQuery { target: Register, bank_id: u8, top_k: u8 },

/// Write entry to bank.
BankWrite { source: Register, bank_id: u8 },

/// Add edge between entries.
BankLink { src_entry: Register, dst_ref: Register, edge_type: u8 },

/// Traverse edges from entry.
BankTraverse { target: Register, start: Register, edge_type: u8, depth: u8 },

/// Delete entry from bank.
BankDelete { bank_id: u8, entry_id: Register },

/// Flush bank to disk.
BankFlush { bank_id: u8 },
```

---

## Crate Structure

```
databank-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs           // Public API re-exports
│   ├── types.rs         // BankId, EntryId, BankRef, EdgeType, Edge, Temperature
│   ├── entry.rs         // BankEntry struct, construction, validation
│   ├── bank.rs          // DataBank: insert, query, traverse, delete, eviction
│   ├── cluster.rs       // BankCluster: multi-bank management, cross-bank ops
│   ├── similarity.rs    // Vector similarity (cosine, dot product, integer-only)
│   ├── codec.rs         // Binary .bank format: encode/decode
│   ├── error.rs         // Error types
│   └── index.rs         // VectorIndex trait + brute-force impl
├── tests/
│   ├── entry_tests.rs
│   ├── bank_tests.rs
│   ├── cluster_tests.rs
│   ├── codec_tests.rs
│   └── similarity_tests.rs
```

### Cargo.toml

```toml
[package]
name = "databank-rs"
version = "0.1.0"
edition = "2021"
authors = ["Magnus Trent <magnus@blackfall.dev>"]
license = "MIT OR Apache-2.0"
description = "Distributed representational memory banks with vector similarity, graph edges, and binary persistence"
repository = "https://github.com/blackfall-labs/databank-rs"
keywords = ["memory", "distributed", "vector", "graph", "neural"]
categories = ["data-structures", "science"]

[dependencies]
# Signal type (canonical)
ternary-signal = "0.1"

# Checksums
crc32fast = "1.4"
xxhash-rust = { version = "0.8", features = ["xxh3"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
thiserror = "2.0"

# Logging
log = "0.4"

[dev-dependencies]
tempfile = "3.0"
```

**Dependencies are minimal.** No SQLite. No heavy indexing libraries. Signal vectors, checksums, serialization, errors. A pacemaker can use this.

---

## Implementation Phases

### Phase 1: Core Types + Entry

1. `types.rs` — BankId, EntryId, BankRef, EdgeType, Edge, Temperature
2. `entry.rs` — BankEntry struct, builder, validation, checksum computation
3. `error.rs` — Error enum
4. Unit tests for entry construction and validation

### Phase 2: Single Bank Operations

1. `bank.rs` — DataBank struct, insert, get, delete, update
2. `similarity.rs` — Integer-only cosine similarity on Signal vectors
3. `index.rs` — VectorIndex trait + BruteForceIndex
4. Bank query (similarity search), edge traversal (local)
5. Eviction policy (temperature-based + LRU hybrid)
6. Unit tests for all bank operations

### Phase 3: Binary Persistence

1. `codec.rs` — .bank v1 binary format encode/decode
2. Atomic writes (temp file → fsync → rename)
3. Load from .bank file, rebuild vector index
4. CRC32 per entry + xxhash64 over full file
5. Round-trip tests: create → persist → load → verify identical

### Phase 4: Multi-Bank Cluster

1. `cluster.rs` — BankCluster: multi-bank management
2. Cross-bank edge traversal
3. Cross-bank similarity query
4. Persist-all / load-all
5. Integration tests: distributed concept (jar example)

### Phase 5: Consolidation Lifecycle

1. Temperature transitions: Hot → Warm → Cool → Cold
2. Access-count-based promotion (frequently accessed entries warm up)
3. Decay-based demotion (unused entries cool down)
4. Eviction of cold + low-confidence entries when at capacity
5. Sleep-cycle hook: bulk temperature transitions, pruning pass

---

## The Jar Test

The definitive integration test for 0.1:

```rust
#[test]
fn test_distributed_concept_jar() {
    let mut cluster = BankCluster::new(temp_dir());

    // Create banks for different regions (BankId is u64 temporal ID)
    let visual = BankId::new("occipital.v4", 0);
    let semantic = BankId::new("temporal.semantic", 0);
    let motor = BankId::new("frontal.motor", 0);
    let auditory = BankId::new("temporal.auditory", 0);

    // Encode visual fragment: cylindrical, transparent, glass
    let visual_entry = cluster.get_or_create(visual, default_config())
        .insert(BankEntry::new(
            signal_vector(&[/* shape features */]),
            Temperature::Hot,
            0, // tick
        ));

    // Encode semantic fragment: container, holds things, fragile
    let semantic_entry = cluster.get_or_create(semantic, default_config())
        .insert(BankEntry::new(
            signal_vector(&[/* semantic features */]),
            Temperature::Hot,
            0,
        ));

    // Encode motor fragment: grip, twist lid
    let motor_entry = cluster.get_or_create(motor, default_config())
        .insert(BankEntry::new(
            signal_vector(&[/* motor features */]),
            Temperature::Hot,
            0,
        ));

    // Encode auditory fragment: /dʒɑːr/ phonemes
    let auditory_entry = cluster.get_or_create(auditory, default_config())
        .insert(BankEntry::new(
            signal_vector(&[/* auditory features */]),
            Temperature::Hot,
            0,
        ));

    // Create cross-bank edges binding them all together
    cluster.link(
        BankRef { bank: semantic, entry: semantic_entry },
        BankRef { bank: visual, entry: visual_entry },
        EdgeType::LooksLike, 200, 0,
    );
    cluster.link(
        BankRef { bank: semantic, entry: semantic_entry },
        BankRef { bank: motor, entry: motor_entry },
        EdgeType::FeelsLike, 180, 0,
    );
    cluster.link(
        BankRef { bank: semantic, entry: semantic_entry },
        BankRef { bank: auditory, entry: auditory_entry },
        EdgeType::SoundsLike, 220, 0,
    );

    // Pattern completion: query visual bank with "cylindrical shape"
    // → should find the visual fragment
    let visual_results = cluster.bank(&visual).unwrap()
        .query_similar(&signal_vector(&[/* cylinder-like */]), 5);
    assert!(!visual_results.is_empty());

    // Follow cross-bank edges from visual → semantic
    let related = cluster.traverse(
        BankRef { bank: visual, entry: visual_entry },
        EdgeType::LooksLike,
        1, // depth
    );
    // Should find the semantic entry (reverse edge: "what looks like this?")

    // Persist all banks
    cluster.flush_all().unwrap();

    // Reload from disk — concept survives crash
    let reloaded = BankCluster::load_all(temp_dir()).unwrap();
    assert_eq!(reloaded.bank(&visual).unwrap().entry_count(), 1);
    assert_eq!(reloaded.bank(&semantic).unwrap().entry_count(), 1);
    assert_eq!(reloaded.bank(&motor).unwrap().entry_count(), 1);
    assert_eq!(reloaded.bank(&auditory).unwrap().entry_count(), 1);

    // Cross-bank edges survive persistence
    let edges = reloaded.bank(&semantic).unwrap()
        .get(semantic_entry).unwrap()
        .edges.len();
    assert_eq!(edges, 3); // LooksLike, FeelsLike, SoundsLike
}
```

---

## Relationship to Existing Crates

| Crate | What It Stores | Mutability | Format |
|-------|---------------|-----------|--------|
| **engram-rs** | Constitutional identity, immutable archives | Read-only | .eng |
| **thermogram-rs** | Connectome weights, temperature lifecycle | Mutable (delta chain) | .thermo |
| **databank-rs** | Representational content, distributed concepts | Mutable (in-memory + periodic persist) | .bank |
| **dataspool-rs** | Append-only audit log, telemetry | Append-only | .spool |
| **cartridge-rs** | Brain identity container (directory structure) | Directory layout | directory |
| **datacard-rs** | Generic binary card format | Read/write | .card |

**Thermograms + Databanks = Complete Brain State:**
- Thermograms: HOW neurons connect (weights, topology)
- Databanks: WHAT those connections retrieve (representational content)
- Together they replace SQLite entirely. No SQL. No tables. Just signal patterns, edges, and connectome wiring.

---

## Resolved Design Decisions

### Q1: Vector Width — Fixed Per Bank (CONFIRMED)

Each bank declares its vector dimension at creation. All entries within a bank share the same width. Different banks can differ.

- Occipital V4 bank: 128 signals (rich visual features)
- Temporal semantic bank: 64 signals (abstract concepts)
- Frontal motor bank: 32 signals (action primitives)

Biologically correct — a cortical column has a fixed number of neurons. Cache-friendly, SIMD-able similarity.

```rust
pub struct DataBank {
    pub id: BankId,
    pub vector_width: u16,  // declared at creation, immutable
    // ...
}
```

Inserts with wrong vector width are rejected with an error.

### Q2: Edge Capacity — Fixed Cap, Default 32 (CONFIRMED)

Max 32 edges per entry (configurable per bank). Under pressure, lowest-weight edges pruned first. Biologically: limited dendritic capacity forces prioritization.

```rust
pub struct BankConfig {
    pub max_edges_per_entry: u16,  // default 32
    // ...
}
```

### Q3: Eviction Policy — Hybrid Scoring (CONFIRMED)

Score = f(temperature, access_count, confidence, age). Lowest score evicted first.

```rust
fn eviction_score(entry: &BankEntry, current_tick: u64) -> u32 {
    let temp_weight: u32 = match entry.temperature {
        Temperature::Hot  => 0,     // hot entries evict first (unproven)
        Temperature::Warm => 64,
        Temperature::Cool => 192,
        Temperature::Cold => 255,   // cold entries resist eviction
    };
    let access_weight = (entry.access_count.min(255)) as u32;
    let confidence_weight = entry.confidence as u32;
    let recency = ((current_tick - entry.last_accessed_tick).min(65535)) as u32;
    let recency_weight = 255u32.saturating_sub(recency / 256);

    // Higher score = more important = evicted LAST
    temp_weight + access_weight + confidence_weight + recency_weight
}
```

The brain forgets things that were never important, not just things that are old.

### Q4: BankId — Compact Temporal ID (RESOLVED)

Not a string. A **64-bit temporally sortable ID** — simplified UUIDv7 semantics in 8 bytes.

```rust
/// 64-bit temporally unique bank identifier.
///
/// Layout:
///   [timestamp_s: 32 bits][region_tag: 24 bits][seq: 8 bits]
///
/// - timestamp_s: Unix seconds (u32). Good for ~136 years from epoch.
/// - region_tag: 24-bit hash of region name. Provides human-traceable
///   grouping without storing variable-length strings in every reference.
/// - seq: 8-bit sequence counter. 256 banks per second per region.
///   Sufficient — region formation is rare (sleep cycles, not milliseconds).
///
/// Properties:
/// - Temporally sortable: newer banks sort after older ones
/// - Region-grouped: banks from the same region cluster in sort order
///   (within the same second)
/// - Collision-resistant: 24-bit region hash x 8-bit seq = 16M combinations
///   per second. Brain will never create that many regions.
/// - Compact: 8 bytes. Fits in a register. Cheap to copy, hash, compare.
/// - No strings in hot path: the string name is stored ONCE in bank metadata,
///   never in cross-bank references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct BankId(pub u64);

impl BankId {
    /// Create a new BankId from current time + region name + sequence.
    pub fn new(region_name: &str, seq: u8) -> Self {
        let timestamp = unix_timestamp_secs() as u64;
        let region_tag = fnv1a_24(region_name) as u64;
        Self((timestamp << 32) | (region_tag << 8) | seq as u64)
    }

    /// Extract creation timestamp (seconds since epoch).
    pub fn timestamp_secs(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Extract region tag (24-bit hash).
    pub fn region_tag(&self) -> u32 {
        ((self.0 >> 8) & 0x00FF_FFFF) as u32
    }

    /// Extract sequence number.
    pub fn seq(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
}
```

**EntryId follows the same pattern** but at finer granularity:

```rust
/// 64-bit temporally unique entry identifier.
///
/// Layout:
///   [timestamp_ms: 42 bits][seq: 22 bits]
///
/// - timestamp_ms: Milliseconds since epoch. 42 bits = ~139 years.
/// - seq: 22-bit sequence counter. 4M entries per millisecond.
///   Entries are created during cognitive ticks (100ms+), so this is
///   absurdly generous. The real value is collision avoidance.
///
/// Temporally sortable: iterating entries by ID gives chronological order.
/// This is free — no secondary index needed for "most recent entries."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct EntryId(pub u64);

impl EntryId {
    pub fn new(seq: u32) -> Self {
        let timestamp = unix_timestamp_ms() as u64;
        Self((timestamp << 22) | (seq as u64 & 0x003F_FFFF))
    }

    pub fn timestamp_ms(&self) -> u64 {
        self.0 >> 22
    }
}
```

**BankRef (cross-bank pointer):**

```rust
/// Cross-bank reference. 16 bytes. Points to one entry in one bank.
/// This is what edges carry. Compact enough for 32 edges per entry
/// to cost only 512 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct BankRef {
    pub bank: BankId,    // 8 bytes
    pub entry: EntryId,  // 8 bytes
}
```

**Human-readable names** are stored once in bank metadata (the `.bank` file header), not in every reference. When debugging, the host resolves `BankId -> name` through the cluster's metadata table.

```rust
/// Bank metadata stored in .bank file header and in BankCluster registry.
pub struct BankMeta {
    pub id: BankId,
    pub name: String,           // "occipital.v4", human-readable, stored once
    pub vector_width: u16,
    pub created_tick: u64,
    pub config: BankConfig,
}
```

### Q5: Cross-Bank Recall — Biological Cascade (RESOLVED)

No orchestrator. No cross-bank queries. **Each region queries ONLY its own bank.** Cross-bank binding happens through thermogram wiring and activation propagation.

**How hearing a bark recalls "dog":**

```
1. AUDITORY CORTEX
   Sound wave -> auditory features extracted
   Region firmware queries auditory.bank: "what matches these features?"
   -> Hit: entry #412 (bark sound pattern), confidence 230
   -> Entry #412 has edges: [SoundsLike -> temporal-semantic #67]
   Region fires output into activity field

2. THERMOGRAM PROPAGATION
   Auditory cortex -> temporal cortex weights fire
   Temporal cortex receives activation from thermogram wiring
   (NOT from a database join -- from neural connection weights)

3. TEMPORAL CORTEX
   Receives activation pattern in hot registers
   Region firmware queries temporal-semantic.bank with incoming activation
   -> Hit: entry #67 (dog concept fragment: animal, four legs, loyal, pet)
   -> Entry #67 has edges: [LooksLike -> occipital #203, FeelsLike -> motor #89]
   Region fires output into activity field

4. THERMOGRAM PROPAGATION (parallel)
   Temporal -> occipital weights fire
   Temporal -> frontal-motor weights fire

5. OCCIPITAL CORTEX
   Receives activation, queries occipital.bank
   -> Hit: entry #203 (dog visual fragment: fur, four legs, snout, tail)
   Fires output into activity field

6. FRONTAL MOTOR CORTEX
   Receives activation, queries frontal-motor.bank
   -> Hit: entry #89 (dog motor fragment: pet, throw ball, walk)
   Fires output into activity field

7. CONVERGENCE FIELD
   All fragments arrive. Coherence above ignition threshold.
   The concept "dog" is now conscious -- fully reconstructed from
   distributed fragments, with no central coordinator.
```

**The critical insight:** Cross-bank edges are NOT database foreign keys that the system follows. They are **records of which other banks have related fragments.** The actual retrieval happens through thermogram connections firing neuron-to-neuron signals. The edges exist so that when a bank entry is accessed, the firmware KNOWS which other regions to send priming signals to — but the activation travels through the neural substrate, not through a query API.

**Firmware pattern for local recall:**

```ternsig
; Region firmware: temporal cortex tick
neuro.FIELD_READ   H0, 0          ; read incoming activation from activity field
neuro.STIM_READ    H1             ; read stimulation levels

; Query local bank -- H0 contains the activation pattern
lifecycle.BANK_QUERY H2, 0, 5     ; query bank 0, top 5 results -> H2

; If hit found, load the full entry vector (pattern completion)
cmp_gt             H2, 0          ; any results?
jump_ifn           .no_hit
lifecycle.BANK_LOAD H3, H2        ; load winning entry's full vector -> H3

; Fire completed pattern into activity field
neuro.FIELD_WRITE  H3, 0          ; write completed pattern to activity field
; Thermogram connections to other regions fire automatically
; (handled by kernel's reticular tick -- not firmware's job)

.no_hit
halt
```

**For non-brain consumers:** Simpler hosts that don't have thermogram wiring can still use `BankCluster::traverse()` directly from Rust. The biological cascade is the brain's path. A security camera can just call `cluster.query_all()` from its host code. Both work — the API supports both patterns.

### Q6: Persistence — Per-Region Dynamics (RESOLVED)

No global persistence frequency. Each bank manages its own persistence based on its own activity dynamics.

```rust
pub struct BankConfig {
    /// Maximum mutations before auto-persist. Default 100.
    /// A highly active region (e.g., hippocampus during learning) may set this
    /// lower (50) for more frequent saves. A stable region (e.g., motor cortex
    /// with well-learned patterns) may set this higher (500).
    pub persist_after_mutations: u32,

    /// Maximum ticks between persists, even if no mutations. Default 10_000.
    /// Ensures sleeping banks still get checkpointed periodically.
    pub persist_after_ticks: u64,

    /// Maximum entries before eviction kicks in. Default 10_000.
    pub max_entries: u32,

    /// Vector width for all entries in this bank.
    pub vector_width: u16,

    /// Max edges per entry. Default 32.
    pub max_edges_per_entry: u16,
}
```

**Why per-region:** The hippocampus encodes rapidly during waking — it might write 50 entries per minute. It needs to persist often (every 50 mutations). The motor cortex adds maybe 1 entry per hour — persisting every 500 mutations is fine, and the tick-based fallback catches it anyway.

Each bank tracks its own dirty state:

```rust
impl DataBank {
    /// Called after every mutation (insert, update, delete, edge add).
    fn on_mutation(&mut self) {
        self.mutations_since_persist += 1;
        self.dirty = true;
    }

    /// Called by host every tick. Returns true if bank should persist now.
    pub fn should_persist(&self, current_tick: u64) -> bool {
        if !self.dirty { return false; }
        self.mutations_since_persist >= self.config.persist_after_mutations
            || (current_tick - self.last_persist_tick) >= self.config.persist_after_ticks
    }
}
```

The host's reticular tick checks `should_persist()` on each bank and writes the ones that are due. Under Ternsig's transaction DomainOps (1.1.10), bulk writes during sleep consolidation are wrapped in `TxnBegin`/`TxnCommit` for atomicity.

### Q7: Partial Pattern Cascade — Biologically Grounded Recall (RESOLVED)

v2's PatternToken/CascadeManager was software-bound Rust doing graph hops in a centralized HashMap. v3 needs a biologically grounded mechanism where partial cues trigger distributed recall through neural propagation — not through software graph traversal.

#### The Mechanism: Sparse Pattern Completion

In neuroscience, hippocampal CA3 performs pattern completion: a partial cue (~20-30% of the original pattern) activates recurrent connections that reconstruct the full pattern. This is a fundamental property of attractor networks — any sufficiently large fragment of a stored pattern will converge to the complete pattern.

For databanks, the equivalent:

```
ENCODING (writing a new concept):
  1. Sensory input activates multiple regions simultaneously
  2. Each region's firmware computes its local representation
  3. Each region writes its fragment to its own bank
  4. Hippocampal firmware creates binding entry:
     - Sparse activation indices from each region
     - Cross-bank edges linking all fragments
  5. Thermogram learning strengthens cross-region connections
     (neurons that fired together now wire together)

RECALL (partial cue triggers full concept):
  1. Partial cue activates ONE region (e.g., hearing "jar")
  2. Region queries local bank with SPARSE vector:
     - Only non-zero signals in the query participate in similarity
     - Matching scores only the active dimensions, ignoring zeros
     - This IS pattern completion: partial input -> full stored entry
  3. Winning entry's COMPLETE vector loaded into hot registers
     (the pattern is now "completed" locally)
  4. Completed pattern fires through thermogram connections
  5. Downstream regions receive activation, do THEIR local completion
  6. Each step adds more fragments to the convergence field
  7. Ignition: enough fragments coherent -> concept is conscious

CASCADE DEPTH:
  - Depth 1: immediate neighbors (direct thermogram connections)
  - Depth 2: neighbors of neighbors (association chains)
  - Depth 3+: rare, expensive, only under high arousal (NE boost)
  - Cascade depth is modulated by norepinephrine:
    Low NE  -> depth 1 only (focused, specific recall)
    High NE -> depth 3+ (broad, associative, creative)
```

#### Sparse Similarity Matching

The mathematical core of pattern completion:

```rust
/// Sparse cosine similarity -- only non-zero query signals participate.
/// This IS pattern completion: a partial cue matches against stored patterns
/// by comparing only the dimensions where the cue has activation.
///
/// Returns similarity as i32 scaled x256.
pub fn sparse_cosine_similarity(query: &[Signal], stored: &[Signal]) -> i32 {
    debug_assert_eq!(query.len(), stored.len());

    let mut dot: i64 = 0;
    let mut query_norm: i64 = 0;
    let mut stored_norm: i64 = 0;

    for (q, s) in query.iter().zip(stored.iter()) {
        if q.magnitude == 0 { continue; }  // skip inactive query dimensions

        let q_val = q.polarity as i64 * q.magnitude as i64;
        let s_val = s.polarity as i64 * s.magnitude as i64;

        dot += q_val * s_val;
        query_norm += q_val * q_val;
        stored_norm += s_val * s_val;
    }

    if query_norm == 0 || stored_norm == 0 { return 0; }

    // Integer sqrt via Newton's method
    let denom = isqrt(query_norm) * isqrt(stored_norm);
    if denom == 0 { return 0; }

    ((dot * 256) / denom) as i32
}
```

**Why sparse:** When you hear "jar", your auditory cortex has a strong activation pattern for the phonemes — but your visual cortex has ZERO activation (you haven't seen anything). The query to auditory.bank is dense (many active signals). The thermogram-propagated activation to visual cortex is sparse (just a few priming signals from the cross-bank edges). Sparse similarity lets those few priming signals match against the full stored visual pattern for "jar" even though most of the query vector is zeros.

#### Cascade Lifecycle

The complete lifecycle of a partial pattern cascade, tick by tick:

```
Tick N:   Input "jar" arrives at auditory cortex
          Auditory firmware: dense query -> auditory.bank -> hit #412
          #412's full vector loaded into H registers
          Auditory output fires into activity field

Tick N+1: Thermogram propagation: auditory -> temporal weights fire
          Temporal cortex receives SPARSE activation (priming signals)
          Temporal firmware: sparse query -> temporal-semantic.bank -> hit #67
          #67's full vector loaded (PATTERN COMPLETED locally)
          Temporal output fires into activity field

Tick N+2: Thermogram propagation: temporal -> occipital, temporal -> motor
          Occipital: sparse query -> occipital.bank -> hit #203 (visual jar)
          Motor: sparse query -> motor.bank -> hit #89 (grip jar)
          Both outputs fire into activity field

Tick N+3: Convergence field: 4 fragments present
          Coherence above ignition threshold
          IGNITION: the concept "jar" is conscious
          All fragments available in workspace for reasoning, expression, etc.
```

**Total recall latency: 3-4 ticks (~300-400ms at 10Hz).** This matches human reaction times for concept recognition from partial cues.

#### What This Replaces from v2

| v2 Mechanism | v3 Replacement | Why Different |
|-------------|---------------|---------------|
| `PatternTokenStore::find_similar()` | `DataBank::query_sparse()` | Distributed per-region, not centralized |
| `CascadeManager::generate_cues()` | Thermogram activation propagation | No software cue generation — neural wiring does it |
| `CascadeManager::execute_hop()` | Downstream region firmware + local `query_sparse()` | Each region completes its own pattern autonomously |
| `PatternToken.verbalization` | Cross-bank edge (SoundsLike) to auditory bank entry + Broca firmware assembly | Language is a bank like any other, not a special field |
| `PatternToken.word_associations` | Hebbian learning in thermogram weights between auditory and semantic regions | Learned through co-activation, not explicit association tracking |
| `PatternToken.temperature` | `BankEntry.temperature` — same lifecycle, distributed per-fragment | Each fragment has its own temperature in its own bank |
| `IntegratedSignature` centroid | `BankEntry.vector` — the actual signal pattern, not a derived signature | Raw representation, not a statistical summary |
| Central `HashMap<PatternTokenId, PatternToken>` | Distributed `BankCluster` of per-region banks | No central store. No software graph hops. Neural cascade. |

**PatternTokens were the v2 prototype. Databanks are the biological realization.** The key insight: v2 did pattern completion in software (Rust code traversing a HashMap). v3 does pattern completion in wetware (firmware querying local banks, thermogram connections propagating activation, convergence field assembling the result). The mechanism is the same — sparse cue → full pattern. The substrate is fundamentally different.

---

## Ternsig 1.1 Compatibility

Databank-rs 0.1 is a standalone Rust library. Ternsig 1.1 provides the runtime integration contract that lets firmware access databanks through the TVMR yield pattern. This section defines how the two crates connect.

### Dependency Direction

```
ternsig 1.1      ←── does NOT depend on databank-rs
databank-rs 0.1  ←── depends on ternary-signal (Signal type only)
astromind-v3     ←── depends on BOTH, wires them together in the kernel
```

Ternsig defines the DomainOp variants. Databank-rs provides the storage engine. The v3 kernel is the fulfillment bridge — it receives DomainOps from firmware and calls databank-rs methods.

### DomainOp Integration (0.2 Scope, Designed Now)

Ternsig 1.1 ships with `#[non_exhaustive]` on `DomainOp` (scope item 1.1.5). This means databank DomainOps can be added in Ternsig 1.2 without breaking existing consumers. The variants are designed now for forward compatibility:

```rust
// Future DomainOp variants (ternsig 1.2, NOT 1.1)
// Databank-rs 0.1 is a library — the host calls it directly from Rust.
// These DomainOps exist so firmware can eventually query banks natively.

/// Query bank by vector similarity. Host reads query vector from source
/// register, runs similarity search on the named bank, writes top_k
/// results (EntryId + score) into target register.
BankQuery { target: Register, source: Register, bank_id: u8, top_k: u8 },

/// Write entry to bank. Host reads vector from source register,
/// creates a BankEntry, inserts into the named bank.
BankWrite { source: Register, bank_id: u8 },

/// Add edge between entries. Host reads source and destination
/// BankRefs from registers, creates an Edge with the specified type.
BankLink { src_ref: Register, dst_ref: Register, edge_type: u8, weight: u8 },

/// Follow edges from an entry. Host reads start BankRef from register,
/// traverses edges of the specified type to given depth, writes
/// results into target register.
BankTraverse { target: Register, start: Register, edge_type: u8, depth: u8 },

/// Flush a bank to disk. Host calls bank.flush() with atomic write.
BankFlush { bank_id: u8 },

/// Load full entry vector into register (pattern completion).
/// Host reads EntryId from source, loads the entry's full vector
/// into target register.
BankLoad { target: Register, source: Register, bank_id: u8 },
```

**For 0.1:** The v3 kernel calls databank-rs directly when it needs bank operations. Firmware yields existing `neuro.FIELD_WRITE` / `neuro.FIELD_READ` DomainOps, and the kernel's fulfillment layer translates these into bank queries where appropriate. This works because:

1. Firmware's job is to compute and fire patterns into fields
2. The kernel sees field activity and decides whether to involve banks
3. Banks are a kernel-side memory system, not a firmware-visible resource (yet)

**For 0.2:** Firmware directly yields `BankQuery` / `BankWrite` / `BankLoad` DomainOps. This gives firmware explicit control over memory encoding and recall. The hippocampal firmware, for example, would explicitly yield `BankWrite` to encode a new memory, rather than relying on the kernel to detect encoding-worthy activity.

### Crash Resistance via Ternsig 1.1

Ternsig 1.1's crash resistance primitives (scope items 1.1.9, 1.1.10, 1.1.11) protect databank persistence:

#### Transaction Protection for Bank Flushes

Bank persistence during sleep consolidation is wrapped in Ternsig transactions:

```ternsig
; Sleep consolidation — atomic bank + thermogram persistence
lifecycle.TXN_BEGIN    0

; Prune dead entries from all banks
; (kernel fulfills by calling bank.evict_cold())
lifecycle.SAVE_THERMO  C0      ; save structural weights (buffered)

; Flush banks that are dirty
; (kernel fulfills by calling cluster.flush_dirty())

lifecycle.TXN_COMMIT   0       ; ALL writes go to disk atomically
```

**If crash during consolidation:** Transaction rolls back. Banks reload from pre-consolidation `.bank` files. In-memory mutations since last flush are lost — but those are HOT entries (unproven, expendable). The COOL/COLD entries that matter are already on disk from their last successful flush.

#### Checkpoint Interaction

Ternsig 1.1's `CheckpointSave` / `CheckpointRestore` DomainOps protect cold register weights. Databanks have their OWN persistence via `.bank` files — they don't use thermogram checkpoints. The two systems are complementary:

| System | What It Protects | Persistence Path | Crash Recovery |
|--------|-----------------|-----------------|----------------|
| Thermogram checkpoints | Cold register weights (connectome) | `.thermo` files via DomainOp | Restore from checkpoint |
| Databank persistence | Representational entries (engrams) | `.bank` files via `DataBank::flush()` | Reload from last flush |
| Transaction DomainOps | Atomic multi-write consistency | WAL-based commit | Rollback incomplete txn |

**They compose:** A transaction can buffer both thermogram saves AND bank flushes. The kernel's fulfillment layer treats bank flushes like any other buffered write — held in memory until `TxnCommit`, discarded on `TxnRollback`.

#### InterpreterSnapshot and Banks

Ternsig 1.1's `InterpreterSnapshot` captures register state, NOT bank state. Banks are kernel-managed storage, not interpreter state. The snapshot/restore lifecycle:

```
On periodic snapshot:
  1. interpreter.snapshot() → captures hot/cold registers, chemical state
  2. cluster.flush_dirty() → persists dirty banks independently
  Both happen, but separately — different persistence granularities.

On crash recovery:
  1. cluster = BankCluster::load_all(bank_dir) → reload all banks
  2. interpreter.restore(snapshot) → restore register state
  3. interpreter.run_with_host(...) → resume execution
  Banks and interpreter state recover independently.
```

### Bank Extension Allocation

The DomainOp integration section above mentions `bank_id: u8` operands. This is a **per-interpreter bank slot**, NOT the global `BankId(u64)`. The mapping:

```rust
// In the v3 kernel's DomainFulfiller:
struct RegionBankMap {
    /// Maps firmware bank slot (0-255) to actual BankId.
    /// Set up during region initialization.
    slots: [Option<BankId>; 256],
}

// When firmware yields BankQuery { bank_id: 0, ... }:
// The kernel resolves slot 0 → BankId(0x...) → actual DataBank
```

This indirection means firmware doesn't need to know global BankIds. Each region's firmware uses small local slot numbers (0 = "my primary bank", 1 = "my edge bank", etc.). The kernel maps slots to real banks at region initialization.

### Register Encoding for Bank Results

When firmware yields `BankQuery` and the kernel fulfills it, query results need to fit in TVMR registers. The convention:

```
Hot register layout for BankQuery results (top_k results):
  H[target][0] = result count (0 to top_k)
  H[target][1] = entry_0 score (i32, scaled ×256)
  H[target][2] = entry_0 id_high (upper 32 bits of EntryId)
  H[target][3] = entry_0 id_low  (lower 32 bits of EntryId)
  H[target][4] = entry_1 score
  ...

Hot register layout for BankLoad result:
  H[target][0..N] = full entry vector (Signal values as i32)
  Each Signal (polarity × magnitude) packed as i32 for firmware math.
```

This uses the existing hot register buffer format. No new register types needed.

### Firmware Pattern: Bank Query + Pattern Completion

The complete firmware flow for a region querying its local bank:

```ternsig
.requires
  neuro      0x0005
  activation 0x0003

.registers
  H0: i32[64]   ; incoming activation from field
  H1: i32[16]   ; query results (top 4 × 4 values)
  H2: i32[64]   ; loaded entry vector (pattern completed)
  H3: i32[64]   ; output pattern
  C0: signal[64, 64]  key="temporal.semantic.w0"

.program
  ; Read incoming activation from activity field
  neuro.FIELD_READ    H0, 0

  ; Query local bank with incoming activation as search vector
  ; (In 0.1: kernel intercepts FIELD_READ, runs bank query internally)
  ; (In 0.2: firmware would yield BankQuery directly)

  ; Pattern completion: apply learned transform
  ternary.TERNARY_MATMUL  H3, C0, H0
  activation.RELU         H3, H3

  ; Write completed pattern to activity field
  ; Thermogram connections propagate to downstream regions
  neuro.FIELD_WRITE   H3, 0
  halt
```

### Version Compatibility Matrix

| databank-rs | ternsig | Integration Level | How Banks Are Accessed |
|-------------|---------|-------------------|----------------------|
| **0.1** | **1.1** | Library only | Kernel calls databank-rs Rust API directly. No firmware-visible bank ops. |
| **0.2** | **1.2** | DomainOp integration | Firmware yields BankQuery/BankWrite/BankLoad. Kernel fulfills via databank-rs. |
| **0.3+** | **1.3+** | Full extension | Bank extension (0x000B) with inline execution for simple queries. Complex ops still yield. |
