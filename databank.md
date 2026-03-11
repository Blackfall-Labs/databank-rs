# databank-rs

Distributed representational memory for neuromorphic systems. Signal-vector banks with typed edges, temperature lifecycle, sparse pattern completion, and binary `.bank` persistence.

## What This Is

Thermograms store the connectome (HOW neurons wire). Databanks store the engrams (WHAT neurons represent).

Each brain region owns one or more DataBanks. A bank holds signal-vector entries — fixed-width patterns of ternary signals (polarity + magnitude) that represent concepts, percepts, episodes, or motor plans. Entries link to each other via typed, weighted edges that can cross bank boundaries.

The core capability is **sparse pattern completion**: a partial cue (with zeros in unknown dimensions) retrieves the best-matching full patterns using integer-only cosine similarity. This is how a fragment of a memory recalls the whole thing.

## Architecture

```
BankCluster (the brain's distributed memory)
├── DataBank "temporal.semantic"   (64d vectors)
├── DataBank "temporal.auditory"   (128d vectors)
├── DataBank "temporal.episodic"   (64d vectors)
├── DataBank "frontal.planning"    (64d vectors)
├── ...
│
├── Cross-bank edges (BankRef → BankRef)
│   IsA, HasA, PartOf, RelatedTo, SimilarTo,
│   Causes, Precedes, LooksLike, SoundsLike,
│   FeelsLike, CoOccurred, FollowedBy, Custom
│
└── Journal (crash recovery, append-only)
```

### Entry Layout

```
BankEntry
├── id:            EntryId    64-bit temporally sortable [ms:42][seq:22]
├── vector:        Vec<Signal> fixed-width ternary pattern (polarity + magnitude)
├── edges:         Vec<Edge>  typed, weighted, cross-bank references
├── origin:        BankId     which bank created this entry
├── temperature:   Temperature  HOT → WARM → COOL → COLD lifecycle
├── access_count:  u32        frequency of retrieval
├── confidence:    u8         0-255 reliability score
└── checksum:      u32        CRC32 integrity verification
```

### Bank Identity

```
BankId — 64-bit temporally sortable
Layout: [timestamp_s:32][region_tag:24][seq:8]
         Unix seconds    FNV-1a hash    counter

EntryId — 64-bit temporally sortable
Layout: [timestamp_ms:42][seq:22]
         milliseconds     ~4M entries/ms
```

## Key Properties

- **Integer-only similarity**: Sparse cosine similarity computed entirely with integer arithmetic. Score range [-256, 256]. No floats.
- **Sparse pattern completion**: Zero-valued query dimensions are skipped — a partial cue matches only on the dimensions it specifies.
- **Temperature lifecycle**: HOT (active learning) → WARM (session patterns) → COOL (proven) → COLD (frozen priors). Matches thermogram lifecycle.
- **Typed edges**: 12 semantic edge types (taxonomic, associative, causal, sensory, episodic) plus custom. Edges are directed, weighted (0-255), and cross bank boundaries.
- **Eviction scoring**: Hybrid score combining temperature, recency, access frequency, and confidence. Cold entries are hardest to evict.
- **Binary persistence**: `.bank` v1 format with xxhash64 integrity, atomic writes (temp file + rename), 32-byte header.
- **Crash recovery**: Optional append-only journal records mutations between full snapshots. Replayed on restart.
- **IVF indexing**: Inverted file index partitions vector space into k clusters for sub-linear search. Integer-only k-means.

## Usage

```rust
use databank_rs::{DataBank, BankCluster, BankConfig, BankId, Temperature, EdgeType, BankRef};
use ternary_signal::Signal;

// Create a bank cluster
let mut cluster = BankCluster::new();

// Create a bank for semantic memory (64-dimensional vectors)
let bank_id = BankId::new("temporal.semantic", 0);
let config = BankConfig { vector_width: 64, ..Default::default() };
let bank = cluster.get_or_create(bank_id, "temporal.semantic".into(), config);

// Store a concept (64-dimensional signal vector)
let concept: Vec<Signal> = (0..64).map(|i| Signal::new(1, (i * 4) as u8)).collect();
let entry_id = bank.insert(concept, Temperature::Hot, 0).unwrap();

// Sparse query — partial cue with zeros in unknown dimensions
let mut query = vec![Signal::new(0, 0); 64];
query[0] = Signal::new(1, 0);    // only specify known dimensions
query[1] = Signal::new(1, 4);
let results = bank.query_sparse(&query, 5); // top 5 matches

// Link entries across banks
let target_ref = BankRef { bank: bank_id, entry: entry_id };
cluster.link(
    BankRef { bank: bank_id, entry: entry_id },
    target_ref,
    EdgeType::RelatedTo,
    200, // weight
    0,   // tick
).unwrap();

// Traverse association chains
let chain = cluster.traverse(target_ref, EdgeType::RelatedTo, 3);

// Temperature lifecycle
bank.consolidation_pass(10_000, 5, 1000); // promote frequently accessed entries
bank.demotion_pass(50);                    // demote low-confidence entries

// Persistence
cluster.flush_dirty(std::path::Path::new("banks/"), 0).unwrap();
let restored = BankCluster::load_all(std::path::Path::new("banks/")).unwrap();
```

## Ternsig Integration

With the `ternsig` feature enabled, firmware programs can access banks directly:

```toml
databank-rs = { version = "0.3", features = ["ternsig"] }
```

This provides:
- `BankFulfiller` — stateless operation executor for DomainOp dispatch (query, write, load, link, traverse, touch, delete, promote, demote, evict, compact, count)
- `BankSlotMap` — maps per-interpreter bank slots (u8) to global BankIds
- `ClusterBankAccess` — implements the ternsig `BankAccess` trait for inline firmware execution without yielding DomainOps
- `bridge` module — bidirectional Signal/i32 conversion for register transport

## Memory Budget

| Bank Size | Vector Width | Entry Overhead | Approx Total |
|-----------|-------------|----------------|--------------|
| 1,024 entries | 64d | ~160 bytes/entry | ~320 KB |
| 4,096 entries | 64d | ~160 bytes/entry | ~1.3 MB |
| 4,096 entries | 128d | ~290 bytes/entry | ~2.3 MB |

Plus edges: ~24 bytes per edge, up to 32 per entry.

## Module Structure

```
src/
  lib.rs          re-exports
  types.rs        BankId, EntryId, BankRef, Edge, EdgeType, Temperature, BankConfig
  entry.rs        BankEntry: representational fragments with lifecycle
  bank.rs         DataBank: single region's memory with query + eviction
  cluster.rs      BankCluster: multi-bank manager with cross-bank linking
  similarity.rs   sparse_cosine_similarity (integer-only)
  index.rs        VectorIndex trait, BruteForceIndex
  ivf.rs          IvfIndex: inverted file index for sub-linear search
  codec.rs        .bank v1 binary format (xxhash64, atomic writes)
  journal.rs      crash recovery (append-only mutation log)
  bridge.rs       Signal <-> i32 register conversion
  fulfiller.rs    BankFulfiller + BankSlotMap for DomainOp dispatch
  access.rs       ClusterBankAccess (ternsig BankAccess trait impl)
  error.rs        DataBankError, Result
```

## License

MIT
