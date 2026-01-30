# PLAN: databank-rs 0.3 — Consolidation Lifecycle & Cross-Bank Recall

**Date:** 2026-01-29
**Crate:** `databank-rs` (`E:\repos\blackfall-labs\databank-rs`)
**Branch:** main
**Status:** Draft — awaiting operator approval
**Depends on:** databank-rs 0.2 (fulfillment, journal, IVF), Ternsig 1.3 (consolidation DomainOps)
**Enables:** v3 Phase 4-5 (full cognition, distributed concept recall)

---

## Summary

databank-rs 0.3 is the **consolidation and recall release**. It adds temperature lifecycle transitions (promote/demote driven by sleep firmware), cross-bank cluster query, the Jar Test (distributed concept recall across banks), k-means IVF refinement, and the BankAccess trait implementation for Ternsig 1.3's inline execution.

After 0.3, the v3 brain has complete distributed representational memory: firmware-driven encoding, similarity recall, typed edge traversal, temperature-gated consolidation, crash-safe journaling, and cross-bank cascade recall.

---

## Scope

### 0.3.1 — Temperature Lifecycle

**File:** `src/entry.rs`

Add temperature transition methods to `BankEntry`:

```rust
impl BankEntry {
    /// Promote temperature one step: Hot→Warm, Warm→Cool, Cool→Cold.
    /// Returns true if promoted, false if already Cold.
    pub fn promote(&mut self) -> bool;

    /// Demote temperature one step: Cold→Cool, Cool→Warm, Warm→Hot.
    /// Returns true if demoted, false if already Hot.
    pub fn demote(&mut self) -> bool;

    /// Check if this entry qualifies for promotion based on access patterns.
    /// Criteria: access_count >= threshold AND age >= min_age ticks.
    pub fn promotion_eligible(&self, current_tick: u64, min_accesses: u32, min_age_ticks: u64) -> bool;

    /// Check if this entry should be demoted (contradictory evidence).
    /// Criteria: confidence dropped below threshold since last promotion.
    pub fn demotion_eligible(&self, confidence_threshold: u8) -> bool;
}
```

**File:** `src/bank.rs`

Add lifecycle operations to `DataBank`:

```rust
impl DataBank {
    /// Promote an entry's temperature. Returns Ok(true) if promoted.
    pub fn promote_entry(&mut self, id: EntryId) -> Result<bool>;

    /// Demote an entry's temperature. Returns Ok(true) if demoted.
    pub fn demote_entry(&mut self, id: EntryId) -> Result<bool>;

    /// Batch promote all eligible entries. Returns count promoted.
    pub fn consolidation_pass(
        &mut self,
        current_tick: u64,
        min_accesses: u32,
        min_age_ticks: u64,
    ) -> usize;

    /// Batch demote entries below confidence threshold. Returns count demoted.
    pub fn demotion_pass(&mut self, confidence_threshold: u8) -> usize;

    /// Evict lowest-scoring entries. Returns count evicted.
    /// Used during sleep pruning to free capacity.
    pub fn evict_n(&mut self, count: usize, current_tick: u64) -> usize;

    /// Compact internal data structures after mass eviction.
    /// Rebuilds the vector index and cleans up reverse edges.
    pub fn compact(&mut self);
}
```

### 0.3.2 — Cross-Bank Cluster Query

**File:** `src/cluster.rs`

Add cross-bank search to `BankCluster`:

```rust
impl BankCluster {
    /// Query across ALL banks in the cluster.
    /// Returns top_k results globally, with bank identification.
    ///
    /// Results are normalized: scores from different banks (potentially
    /// different vector widths) are comparable via z-score normalization
    /// within each bank before global ranking.
    pub fn query_all(
        &self,
        query_per_bank: &HashMap<BankId, Vec<Signal>>,
        top_k: usize,
    ) -> Vec<ClusterQueryResult>;

    /// Query a subset of banks by name pattern.
    /// E.g., "temporal.*" queries all temporal banks.
    pub fn query_by_prefix(
        &self,
        prefix: &str,
        query: &[Signal],
        top_k: usize,
    ) -> Vec<ClusterQueryResult>;
}

/// Result of a cross-bank query.
#[derive(Debug, Clone)]
pub struct ClusterQueryResult {
    pub bank_id: BankId,
    pub bank_name: String,
    pub entry_id: EntryId,
    pub score: i32,           // raw similarity score (scaled x256)
    pub normalized_score: i32, // z-score normalized for cross-bank comparison
}
```

**Design decisions:**

- `query_all` takes per-bank query vectors because banks have different `vector_width`. The caller provides a query vector matched to each bank's width. Banks without a query vector are skipped.
- `query_by_prefix` is a convenience for same-width banks (e.g., all `temporal.*` banks with width 64). It uses the same query vector for all matching banks.
- Z-score normalization: compute mean and std of similarity scores within each bank, then normalize. This makes scores from a 32-wide bank comparable to scores from a 128-wide bank.

### 0.3.3 — BankAccess Implementation

**New file:** `src/access.rs`

Implement Ternsig 1.3's `BankAccess` trait for `BankCluster`:

```rust
use crate::{BankCluster, BankSlotMap, Temperature};

/// BankAccess implementation backed by a BankCluster + BankSlotMap.
/// Used by the v3 kernel to provide inline bank execution to Ternsig interpreters.
pub struct ClusterBankAccess<'a> {
    cluster: &'a mut BankCluster,
    slot_map: &'a BankSlotMap,
    temperature: Temperature,
    tick: u64,
}

impl<'a> ClusterBankAccess<'a> {
    pub fn new(
        cluster: &'a mut BankCluster,
        slot_map: &'a BankSlotMap,
        temperature: Temperature,
        tick: u64,
    ) -> Self;
}
```

This implements the `BankAccess` trait from ternsig 1.3, converting between the trait's i32-based interface and databank-rs's Signal-based internal types using the bridge module from 0.2.

**Note:** This module depends on ternsig 1.3 for the `BankAccess` trait. Since databank-rs doesn't depend on ternsig directly, this is provided as a **feature-gated module**:

```toml
[features]
default = []
ternsig = ["dep:ternsig"]

[dependencies]
ternsig = { version = "1.3", optional = true }
```

When the `ternsig` feature is enabled, `src/access.rs` is compiled. The v3 kernel enables this feature. Standalone databank-rs usage (without ternsig) omits this module.

### 0.3.4 — Consolidation Fulfiller Extensions

**File:** `src/fulfiller.rs`

Add fulfillment methods for Ternsig 1.3's consolidation DomainOps:

```rust
impl BankFulfiller {
    /// Fulfill BankPromote: promote entry temperature.
    pub fn promote(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult;

    /// Fulfill BankDemote: demote entry temperature.
    pub fn demote(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult;

    /// Fulfill BankEvict: evict lowest-scoring entries.
    pub fn evict(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        count: u8,
        current_tick: u64,
    ) -> FulfillResult;

    /// Fulfill BankCompact: compact bank after eviction.
    pub fn compact(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
    ) -> FulfillResult;
}
```

### 0.3.5 — K-Means IVF Refinement

**File:** `src/ivf.rs`

Upgrade IVF index from random centroid initialization (0.2) to proper k-means:

```rust
impl IvfIndex {
    /// Rebuild with k-means clustering.
    /// max_iterations: typically 10-20.
    /// Reassigns all entries to nearest centroid after convergence.
    pub fn rebuild_kmeans(
        &mut self,
        entries: &HashMap<EntryId, BankEntry>,
        max_iterations: usize,
    );
}
```

K-means uses integer arithmetic only (consistent with ASTRO_004):
- Distance metric: sum of squared differences (i32 arithmetic)
- Centroid update: component-wise mean using i64 accumulator / count
- Convergence: stop when no entry changes cluster assignment, or max_iterations reached

### 0.3.6 — Journal Temperature Entries

**File:** `src/journal.rs`

Add journal entry types for temperature transitions:

```rust
pub enum JournalEntry {
    // ... existing from 0.2 ...

    /// Entry promoted (temperature increased).
    Promote { bank_id: BankId, entry_id: EntryId, new_temp: Temperature },
    /// Entry demoted (temperature decreased).
    Demote { bank_id: BankId, entry_id: EntryId, new_temp: Temperature },
    /// Batch eviction (entries removed during sleep).
    BatchEvict { bank_id: BankId, entry_ids: Vec<EntryId> },
}
```

Journal format tags: `5=Promote, 6=Demote, 7=BatchEvict`.

### 0.3.7 — The Jar Test

**New file:** `tests/jar_test.rs`

Integration test from PLAN_DATABANK_0.1.md: encode a distributed "jar" concept across 4 banks, persist, reload, verify edges and recall.

**Test scenario:**

1. Create 4 banks: `temporal.semantic` (64-wide), `occipital.v4` (128-wide), `parietal.spatial` (32-wide), `frontal.expression` (64-wide)
2. Encode "jar" as distributed fragments:
   - `temporal.semantic`: word meaning vector (Signal pattern for "jar")
   - `occipital.v4`: visual appearance vector (cylindrical shape, transparent)
   - `parietal.spatial`: spatial properties (upright, holds things)
   - `frontal.expression`: usage context (kitchen, container)
3. Create cross-bank edges:
   - semantic→visual: `IsA` edge (what it looks like)
   - semantic→spatial: `HasA` edge (spatial properties)
   - semantic→expression: `RelatedTo` edge (usage)
   - visual→spatial: `CoOccurred` edge (seen together)
4. Persist all 4 banks to `.bank` files
5. Drop cluster, reload from disk
6. Verify:
   - All 4 entries exist with correct vectors
   - All cross-bank edges intact
   - `query_sparse` on semantic bank with partial "jar" cue returns the entry
   - `traverse(semantic_entry, IsA, 1)` returns visual entry
   - Edge traversal from semantic reaches all 3 other banks
7. Temperature lifecycle:
   - Promote semantic entry: Hot → Warm
   - Verify promotion persists through save/load cycle
   - Evict the lowest-scoring entry from a bank
   - Verify evicted entry is gone after reload

### 0.3.8 — Version Bump

**File:** `Cargo.toml`

Version: `0.2.0` → `0.3.0`

Add optional ternsig dependency:
```toml
[features]
default = []
ternsig = ["dep:ternsig"]

[dependencies]
ternsig = { version = "1.3", path = "../../../ternsig", optional = true }
```

---

## Implementation Order

### Step 1: Temperature lifecycle (0.3.1)
Entry promote/demote methods, bank consolidation_pass/demotion_pass/evict_n/compact. Unit tests.

### Step 2: Cross-bank query (0.3.2)
ClusterQueryResult, query_all, query_by_prefix. Tests with multi-bank clusters.

### Step 3: Consolidation fulfiller (0.3.4)
Add promote/demote/evict/compact to BankFulfiller. Tests.

### Step 4: K-means IVF (0.3.5)
Upgrade IvfIndex with rebuild_kmeans. Accuracy tests vs brute-force.

### Step 5: Journal temperature entries (0.3.6)
Add Promote/Demote/BatchEvict journal types. Replay tests.

### Step 6: BankAccess implementation (0.3.3)
Feature-gated ternsig integration. Implement ClusterBankAccess. Tests with mock interpreter.

### Step 7: Jar Test (0.3.7)
Full distributed concept recall integration test.

### Step 8: Version bump + full test (0.3.8)
Bump to 0.3.0. Full test suite including Jar Test.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `src/lib.rs` | Add module declarations: access (feature-gated) |
| `src/entry.rs` | +promote/demote methods, eligibility checks |
| `src/bank.rs` | +consolidation_pass, demotion_pass, evict_n, compact |
| `src/cluster.rs` | +query_all, query_by_prefix, ClusterQueryResult |
| `src/fulfiller.rs` | +promote, demote, evict, compact fulfillment methods |
| `src/ivf.rs` | +rebuild_kmeans with integer k-means |
| `src/journal.rs` | +Promote, Demote, BatchEvict journal entry types |
| `src/access.rs` | **NEW** — ClusterBankAccess implementing BankAccess trait (feature-gated) |
| `tests/jar_test.rs` | **NEW** — Distributed concept recall integration test |
| `Cargo.toml` | Version 0.2.0 → 0.3.0, add optional ternsig dep |

---

## Testing Strategy

- **Temperature tests:** promote/demote individual entries, consolidation_pass batch promote, demotion_pass batch demote, promote persistence through save/load
- **Cross-bank query tests:** query_all returns globally ranked results, z-score normalization produces comparable scores across different vector widths
- **Fulfiller tests:** promote/demote/evict/compact fulfillment with mock clusters
- **K-means tests:** convergence on clustered data, accuracy ≥95% vs brute-force top-k
- **Journal tests:** Promote/Demote/BatchEvict write/read round-trip, replay correctness
- **BankAccess tests:** ClusterBankAccess query/load/count/write with real clusters
- **Jar Test:** Full distributed concept lifecycle (encode → link → persist → reload → recall → promote → evict)
- **Regression:** All 0.1 + 0.2 tests pass unchanged

---

## v3 Readiness Checklist

After databank-rs 0.3, the complete memory system is ready for v3:

| Capability | Version | Status |
|------------|---------|--------|
| Signal-vector entries with typed edges | 0.1 | Done |
| Sparse integer cosine similarity | 0.1 | Done |
| Binary .bank persistence | 0.1 | Done |
| BankCluster multi-bank management | 0.1 | Done |
| DomainOp fulfillment helpers | 0.2 | Planned |
| Delta journal crash recovery | 0.2 | Planned |
| IVF sub-linear search | 0.2 | Planned |
| Temperature lifecycle transitions | **0.3** | This plan |
| Cross-bank cluster query | **0.3** | This plan |
| BankAccess for inline execution | **0.3** | This plan |
| Consolidation fulfillment | **0.3** | This plan |
| K-means IVF refinement | **0.3** | This plan |
| Jar Test (distributed recall proof) | **0.3** | This plan |

---

## Post-0.3 (v3 Phase 5+)

| Feature | When | Rationale |
|---------|------|-----------|
| Weighted edge traversal | v3 Phase 5 | Priority queue + score decay for biological cascade |
| Bank-level access control | v3 Phase 5 | Per-region read/write permissions for safety |
| Schema migration .bank v2 | If needed | Only if format changes required |
| Network-distributed clusters | v3 Phase 6+ | Multi-process brain (far future) |
