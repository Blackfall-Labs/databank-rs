# PLAN: databank-rs 0.2 — DomainOp Fulfillment & Delta Journal

**Date:** 2026-01-29
**Crate:** `databank-rs` (`E:\repos\blackfall-labs\databank-rs`)
**Branch:** main
**Status:** Draft — awaiting operator approval
**Depends on:** databank-rs 0.1 (complete), Ternsig 1.2 (bank DomainOps)
**Enables:** Ternsig 1.3 (inline bank execution), v3 Phase 3+ (firmware-driven bank access)

---

## Summary

databank-rs 0.2 is the **DomainOp fulfillment release**. It provides the host-side helpers that the v3 kernel needs to fulfill Ternsig 1.2's bank DomainOps (`BankQuery`, `BankWrite`, `BankLoad`, `BankLink`, `BankTraverse`, `BankTouch`, `BankDelete`, `BankCount`). It also adds a delta journal for crash recovery between full persists, and an IVF (Inverted File) index for sub-linear similarity search.

After 0.2, the v3 kernel's `DomainFulfiller` can call `bank_fulfiller.fulfill(op, cluster, bank_map)` and get back results packed in the register format firmware expects.

---

## Scope

### 0.2.1 — Signal/Register Conversion Helpers

**New file:** `src/bridge.rs`

Conversion between Signal vectors and i32 register slices (the format firmware uses):

```rust
use ternary_signal::Signal;

/// Convert a Signal vector to i32 register values.
/// Each Signal becomes: (polarity as i32) * (magnitude as i32)
/// polarity: -1, 0, +1 → signed_value range: [-255, 255]
pub fn signals_to_i32(signals: &[Signal]) -> Vec<i32>;

/// Convert i32 register values back to Signal vector.
/// Clamps to [-255, 255], splits into polarity + magnitude.
pub fn i32_to_signals(values: &[i32]) -> Vec<Signal>;

/// Pack an EntryId (u64) into two i32 values (high, low).
pub fn entry_id_to_i32_pair(id: EntryId) -> (i32, i32);

/// Unpack two i32 values into an EntryId.
pub fn i32_pair_to_entry_id(high: i32, low: i32) -> EntryId;

/// Pack BankRef into i32 slice: [bank_slot, entry_id_high, entry_id_low]
/// bank_slot is the per-interpreter slot index, NOT the global BankId.
pub fn bank_ref_to_i32_slice(slot: u8, entry: EntryId) -> [i32; 3];

/// Pack QueryResult list into i32 register layout:
///   [count, score_0, id_high_0, id_low_0, score_1, ...]
pub fn query_results_to_i32(results: &[QueryResult]) -> Vec<i32>;

/// Pack traverse results (Vec<(u8, EntryId)>) into i32 register layout:
///   [count, slot_0, id_high_0, id_low_0, slot_1, ...]
pub fn traverse_results_to_i32(results: &[(u8, EntryId)]) -> Vec<i32>;
```

### 0.2.2 — BankFulfiller

**New file:** `src/fulfiller.rs`

The main integration point. A stateless helper that the v3 kernel calls when firmware yields bank DomainOps.

```rust
use crate::{BankCluster, DataBank, QueryResult, EntryId, BankRef, EdgeType, Edge, Temperature};

/// Maps per-interpreter bank_slot (u8) → global BankId.
/// The kernel initializes this per-region during boot.
pub struct BankSlotMap {
    slots: [Option<BankId>; 256],
}

impl BankSlotMap {
    pub fn new() -> Self;
    pub fn bind(&mut self, slot: u8, bank_id: BankId);
    pub fn resolve(&self, slot: u8) -> Option<BankId>;
}

/// Result of fulfilling a bank DomainOp.
/// Contains the i32 register data to write back to the interpreter.
pub enum FulfillResult {
    /// Write data into target hot register.
    WriteRegister { register_index: u8, data: Vec<i32>, shape: Vec<usize> },
    /// No register output (write-only ops like BankLink, BankTouch, BankDelete).
    Ok,
    /// Error during fulfillment.
    Error(String),
}

/// Fulfills bank DomainOps using the cluster and slot map.
pub struct BankFulfiller;

impl BankFulfiller {
    /// Fulfill a BankQuery DomainOp.
    /// Reads query vector from source register data, runs sparse cosine
    /// similarity on the mapped bank, returns packed results.
    pub fn query(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        top_k: u8,
    ) -> FulfillResult;

    /// Fulfill a BankWrite DomainOp.
    /// Reads Signal vector from source register data, inserts into bank.
    /// Returns new EntryId packed as [id_high, id_low].
    pub fn write(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        temperature: Temperature,
        tick: u64,
    ) -> FulfillResult;

    /// Fulfill a BankLoad DomainOp.
    /// Reads EntryId from source register data, loads full entry vector.
    pub fn load(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult;

    /// Fulfill a BankLink DomainOp.
    /// Creates a typed edge between entries.
    pub fn link(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        edge_type: u8,
        tick: u64,
    ) -> FulfillResult;

    /// Fulfill a BankTraverse DomainOp.
    /// BFS/DFS edge traversal from starting entry.
    pub fn traverse(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        edge_type: u8,
        depth: u8,
    ) -> FulfillResult;

    /// Fulfill a BankTouch DomainOp.
    pub fn touch(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        tick: u64,
    ) -> FulfillResult;

    /// Fulfill a BankDelete DomainOp.
    pub fn delete(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult;

    /// Fulfill a BankCount DomainOp.
    pub fn count(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
    ) -> FulfillResult;
}
```

**Design decisions:**

- `BankFulfiller` is stateless — all state lives in `BankCluster` and `BankSlotMap`. This makes it trivially testable and avoids ownership complexity.
- Temperature for BankWrite is provided by the kernel (from the brain's current learning phase), not by firmware. Firmware decides WHAT to write; the kernel decides HOW it's stored.
- The kernel reads source register data before calling the fulfiller. The fulfiller returns data; the kernel writes it to the interpreter's target register. This avoids the fulfiller needing to know about the interpreter's internal structure.

### 0.2.3 — Delta Journal

**New file:** `src/journal.rs`

Append-only mutation journal for crash recovery between full `.bank` persists.

**Problem:** 0.1 persists via full `.bank` snapshots every N mutations. If the process crashes between snapshots, mutations since last snapshot are lost.

**Solution:** Append mutations to a `.journal` file as they occur. On load, replay the journal on top of the last `.bank` snapshot.

```rust
/// A single journal entry: one mutation to a bank.
#[derive(Debug, Clone)]
pub enum JournalEntry {
    /// New entry inserted.
    Insert { bank_id: BankId, entry: BankEntry },
    /// Entry removed.
    Remove { bank_id: BankId, entry_id: EntryId },
    /// Entry touched (access count + tick update).
    Touch { bank_id: BankId, entry_id: EntryId, tick: u64 },
    /// Edge added to an entry.
    AddEdge { bank_id: BankId, entry_id: EntryId, edge: Edge },
    /// Temperature changed.
    SetTemperature { bank_id: BankId, entry_id: EntryId, temperature: Temperature },
}

/// Append-only journal writer.
pub struct JournalWriter {
    // Internal: BufWriter on an append-mode file
}

impl JournalWriter {
    pub fn open(path: &Path) -> Result<Self>;
    pub fn append(&mut self, entry: &JournalEntry) -> Result<()>;
    pub fn flush(&mut self) -> Result<()>;
    pub fn close(self) -> Result<()>;
}

/// Journal reader for replay during crash recovery.
pub struct JournalReader;

impl JournalReader {
    /// Read all entries from journal file.
    /// Tolerates truncated final entry (crash mid-write).
    pub fn read_all(path: &Path) -> Result<Vec<JournalEntry>>;

    /// Replay journal entries onto an existing bank cluster.
    /// Returns count of entries replayed.
    pub fn replay(entries: &[JournalEntry], cluster: &mut BankCluster) -> Result<usize>;
}

/// Truncate (reset) journal after a full snapshot completes.
pub fn truncate_journal(path: &Path) -> Result<()>;
```

**Journal binary format (per entry):**
```
[0]      Entry type tag (u8): 0=Insert, 1=Remove, 2=Touch, 3=AddEdge, 4=SetTemperature
[1..9]   BankId (u64 LE)
[9..17]  EntryId (u64 LE)
[17..]   Payload (type-dependent, variable length)
[last 4] CRC32 of entire entry (for corruption detection)
```

**Recovery protocol:**
1. Load `.bank` snapshot (full state at last persist)
2. Check for `.journal` file alongside it
3. If journal exists, replay all valid entries onto the cluster
4. Truncate journal
5. Resume normal operation

### 0.2.4 — IVF Index (Inverted File)

**New file:** `src/ivf.rs`

Sub-linear approximate nearest neighbor search for larger banks.

```rust
/// Inverted File Index — partitions vector space into clusters.
/// Each entry is assigned to its nearest centroid. Query searches
/// only nprobe nearest clusters instead of all entries.
pub struct IvfIndex {
    centroids: Vec<Vec<i32>>,       // k centroids in i32 (signal-derived)
    assignments: Vec<Vec<EntryId>>, // per-centroid entry lists
    nprobe: usize,                  // clusters to search (default: 4)
    k: usize,                       // number of centroids
}

impl VectorIndex for IvfIndex {
    fn insert(&mut self, id: EntryId, vector: &[Signal]);
    fn remove(&mut self, id: EntryId);
    fn query(&self, query: &[Signal], entries: &HashMap<EntryId, BankEntry>, top_k: usize) -> Vec<QueryResult>;
    fn rebuild(&mut self, entries: &HashMap<EntryId, BankEntry>);
}
```

**Centroid initialization:** Random sample from entries (no k-means iteration for 0.2). Full k-means deferred to 0.3.

**Parameters:**
- `k` = sqrt(n) centroids (typical: bank with 4096 entries → 64 centroids)
- `nprobe` = 4 by default (search 4 nearest clusters per query)
- Expected speedup: ~16x over brute-force at 4096 entries

**Integration with DataBank:**

```rust
/// Bank index type selector.
pub enum IndexType {
    BruteForce,
    Ivf { k: usize, nprobe: usize },
}
```

Add to `BankConfig`:
```rust
pub index_type: IndexType,  // default: BruteForce
```

DataBank selects the index implementation based on config. Existing banks default to BruteForce (backward compatible).

### 0.2.5 — Cluster Journal Integration

**File:** `src/cluster.rs`

Add journal awareness to BankCluster:

```rust
impl BankCluster {
    /// Create cluster with journal writer for crash recovery.
    pub fn with_journal(journal_path: &Path) -> Result<Self>;

    /// Record a mutation to the journal.
    /// Called internally by insert/remove/touch/add_edge operations
    /// when a journal writer is configured.
    fn journal_mutation(&mut self, entry: JournalEntry) -> Result<()>;

    /// Load cluster from directory with journal replay.
    /// 1. Load all .bank files
    /// 2. Find and replay .journal file
    /// 3. Truncate journal
    pub fn load_with_journal(dir: &Path) -> Result<Self>;

    /// Flush dirty banks AND truncate journal.
    /// After a full snapshot, the journal is no longer needed.
    pub fn flush_dirty_with_journal(&mut self, dir: &Path, current_tick: u64) -> Result<usize>;
}
```

### 0.2.6 — Version Bump

**File:** `Cargo.toml`

Version: `0.1.0` → `0.2.0`

No new dependencies. The bridge module uses `ternary-signal` (already a dep) for Signal conversion.

---

## Implementation Order

### Step 1: bridge.rs (0.2.1)
Signal/i32 conversion helpers. Pure functions, no external deps. Unit tests for round-trip correctness.

### Step 2: fulfiller.rs (0.2.2)
BankFulfiller + BankSlotMap. Tests with in-memory clusters — fulfill each operation type and verify register output format.

### Step 3: journal.rs (0.2.3)
Binary journal format. Writer + reader + replay. Tests: write entries → read back, corrupt final entry tolerance, replay onto cluster.

### Step 4: ivf.rs (0.2.4)
IVF index. Implements VectorIndex trait. Tests: insert/query/rebuild, accuracy vs brute-force baseline, speedup measurement.

### Step 5: Cluster journal integration (0.2.5)
Wire journal into BankCluster. Tests: load_with_journal round-trip, flush_dirty_with_journal truncation.

### Step 6: Version bump + full test (0.2.6)
Bump to 0.2.0. Full test suite. Verify no regressions on 0.1 tests.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/lib.rs` | Add module declarations: bridge, fulfiller, journal, ivf |
| `src/bridge.rs` | **NEW** — Signal/register conversion helpers |
| `src/fulfiller.rs` | **NEW** — BankFulfiller, BankSlotMap, FulfillResult |
| `src/journal.rs` | **NEW** — JournalEntry, JournalWriter, JournalReader |
| `src/ivf.rs` | **NEW** — IvfIndex implementing VectorIndex trait |
| `src/cluster.rs` | Add journal-aware methods |
| `src/bank.rs` | Add IndexType config, select index impl |
| `src/types.rs` | Add IndexType enum to BankConfig |
| `Cargo.toml` | Version 0.1.0 → 0.2.0 |

---

## Testing Strategy

- **bridge tests:** Signal→i32→Signal round-trip, EntryId packing, QueryResult packing
- **fulfiller tests:** Each of 8 fulfill methods with mock clusters, verify register output layout
- **journal tests:** Write→read round-trip, truncated entry tolerance, replay correctness, truncation after flush
- **IVF tests:** Accuracy ≥90% of brute-force top-k on random data, rebuild correctness, insert/remove consistency
- **cluster journal tests:** load_with_journal recovery, flush_dirty_with_journal resets journal
- **Regression:** All 49 existing tests pass unchanged

---

## Deferred to 0.3

| Feature | Rationale |
|---------|-----------|
| Full k-means for IVF centroids | 0.2 uses random init. k-means requires iterative refinement. |
| Temperature transitions (consolidation lifecycle) | Needs sleep firmware to drive transitions. |
| query_all across cluster | Requires scoring normalization across banks with different vector widths. |
| Weighted edge traversal | BFS uses uniform depth. Weighted requires priority queue + score decay. |
| Schema migration for .bank v2 format | No breaking format changes in 0.2. Journal is a separate file. |
