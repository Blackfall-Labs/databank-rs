//! The Jar Test — distributed concept recall integration test.
//!
//! Encodes a "jar" concept across 4 banks (semantic, visual, spatial, expression),
//! links them with typed edges, persists to disk, reloads, and verifies that
//! the full distributed representation survives and can be recalled.

use databank_rs::*;
use std::collections::HashMap;
use ternary_signal::Signal;

fn sig(polarity: i8, magnitude: u8) -> Signal {
    Signal::new(polarity, magnitude)
}

fn jar_semantic_vector() -> Vec<Signal> {
    // "jar" = container, glass, cylindrical, kitchen item
    let mut v = vec![sig(0, 0); 64];
    v[0] = sig(1, 200); // container concept
    v[1] = sig(1, 180); // glass material
    v[2] = sig(1, 150); // cylindrical shape
    v[3] = sig(1, 120); // kitchen context
    v[10] = sig(1, 100); // holdable
    v[20] = sig(1, 90);  // transparent
    v
}

fn jar_visual_vector() -> Vec<Signal> {
    // Visual appearance: cylindrical, transparent, has lid
    let mut v = vec![sig(0, 0); 128];
    v[0] = sig(1, 200); // cylindrical
    v[1] = sig(1, 180); // transparent
    v[5] = sig(1, 150); // vertical orientation
    v[10] = sig(1, 100); // has lid
    v[30] = sig(1, 80);  // small/medium size
    v
}

fn jar_spatial_vector() -> Vec<Signal> {
    // Spatial properties: upright, holds things, shelf placement
    let mut v = vec![sig(0, 0); 32];
    v[0] = sig(1, 200); // upright
    v[1] = sig(1, 180); // containment capacity
    v[5] = sig(1, 120); // shelf-level position
    v
}

fn jar_expression_vector() -> Vec<Signal> {
    // Usage context: kitchen, container, storage
    let mut v = vec![sig(0, 0); 64];
    v[0] = sig(1, 200); // kitchen
    v[1] = sig(1, 180); // storage
    v[2] = sig(1, 150); // food-related
    v
}

#[test]
fn jar_distributed_concept_full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();

    // =========================================================================
    // Phase 1: Create banks and encode the "jar" concept
    // =========================================================================

    let id_semantic = BankId::from_raw(1);
    let id_visual = BankId::from_raw(2);
    let id_spatial = BankId::from_raw(3);
    let id_expression = BankId::from_raw(4);

    let mut cluster = BankCluster::new();

    let semantic = cluster.get_or_create(
        id_semantic,
        "temporal.semantic".into(),
        BankConfig { vector_width: 64, persist_after_mutations: 1, persist_after_ticks: 0, ..BankConfig::default() },
    );
    let eid_semantic = semantic.insert(jar_semantic_vector(), Temperature::Hot, 0).unwrap();

    let visual = cluster.get_or_create(
        id_visual,
        "occipital.v4".into(),
        BankConfig { vector_width: 128, persist_after_mutations: 1, persist_after_ticks: 0, ..BankConfig::default() },
    );
    let eid_visual = visual.insert(jar_visual_vector(), Temperature::Hot, 0).unwrap();

    let spatial = cluster.get_or_create(
        id_spatial,
        "parietal.spatial".into(),
        BankConfig { vector_width: 32, persist_after_mutations: 1, persist_after_ticks: 0, ..BankConfig::default() },
    );
    let eid_spatial = spatial.insert(jar_spatial_vector(), Temperature::Hot, 0).unwrap();

    let expression = cluster.get_or_create(
        id_expression,
        "frontal.expression".into(),
        BankConfig { vector_width: 64, persist_after_mutations: 1, persist_after_ticks: 0, ..BankConfig::default() },
    );
    let eid_expression = expression.insert(jar_expression_vector(), Temperature::Hot, 0).unwrap();

    // =========================================================================
    // Phase 2: Create cross-bank edges
    // =========================================================================

    let ref_semantic = BankRef { bank: id_semantic, entry: eid_semantic };
    let ref_visual = BankRef { bank: id_visual, entry: eid_visual };
    let ref_spatial = BankRef { bank: id_spatial, entry: eid_spatial };
    let ref_expression = BankRef { bank: id_expression, entry: eid_expression };

    // semantic → visual: IsA (what it looks like)
    cluster.link(ref_semantic, ref_visual, EdgeType::IsA, 200, 0).unwrap();
    // semantic → spatial: HasA (spatial properties)
    cluster.link(ref_semantic, ref_spatial, EdgeType::HasA, 180, 0).unwrap();
    // semantic → expression: RelatedTo (usage)
    cluster.link(ref_semantic, ref_expression, EdgeType::RelatedTo, 160, 0).unwrap();
    // visual → spatial: CoOccurred (seen together)
    cluster.link(ref_visual, ref_spatial, EdgeType::CoOccurred, 140, 0).unwrap();

    // =========================================================================
    // Phase 3: Persist to disk
    // =========================================================================

    let flushed = cluster.flush_dirty(dir.path(), 100).unwrap();
    assert_eq!(flushed, 4, "All 4 banks should be flushed");

    // =========================================================================
    // Phase 4: Drop and reload
    // =========================================================================

    drop(cluster);
    let loaded = BankCluster::load_all(dir.path()).unwrap();
    assert_eq!(loaded.len(), 4, "All 4 banks should be loaded");

    // Verify banks exist by name
    assert!(loaded.get_by_name("temporal.semantic").is_some());
    assert!(loaded.get_by_name("occipital.v4").is_some());
    assert!(loaded.get_by_name("parietal.spatial").is_some());
    assert!(loaded.get_by_name("frontal.expression").is_some());

    // Verify entries
    let loaded_semantic = loaded.get(id_semantic).unwrap();
    assert_eq!(loaded_semantic.len(), 1);
    let entry = loaded_semantic.get(eid_semantic).unwrap();
    assert_eq!(entry.vector[0], sig(1, 200)); // container concept preserved

    // =========================================================================
    // Phase 5: Recall via sparse query
    // =========================================================================

    // Partial cue: "container" + "glass" → should recall "jar"
    let mut partial_cue = vec![sig(0, 0); 64];
    partial_cue[0] = sig(1, 200); // container
    partial_cue[1] = sig(1, 180); // glass
    let results = loaded_semantic.query_sparse(&partial_cue, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].entry_id, eid_semantic);
    assert!(results[0].score > 200, "Partial cue should strongly match. Score: {}", results[0].score);

    // =========================================================================
    // Phase 6: Edge traversal
    // =========================================================================

    // From semantic entry, follow IsA → should reach visual
    let isa_targets = loaded.traverse(ref_semantic, EdgeType::IsA, 1);
    assert_eq!(isa_targets.len(), 1);
    assert_eq!(isa_targets[0], ref_visual);

    // From semantic, all edge types depth 1 → should reach visual, spatial, expression
    let all_from_semantic_edges = loaded_semantic.edges_from(eid_semantic);
    assert_eq!(all_from_semantic_edges.len(), 3);

    // Full traversal: follow all edges from semantic, depth 2
    // semantic -HasA-> spatial
    // semantic -RelatedTo-> expression
    // semantic -IsA-> visual -CoOccurred-> spatial
    let has_a = loaded.traverse(ref_semantic, EdgeType::HasA, 2);
    assert_eq!(has_a.len(), 1);
    assert_eq!(has_a[0], ref_spatial);

    // =========================================================================
    // Phase 7: Cross-bank query
    // =========================================================================

    let mut query_map = HashMap::new();
    query_map.insert(id_semantic, jar_semantic_vector());
    query_map.insert(id_visual, jar_visual_vector());
    let cross_results = loaded.query_all(&query_map, 5);
    assert_eq!(cross_results.len(), 2, "Should find 1 result per queried bank");
    for r in &cross_results {
        assert!(r.score > 200, "Should have high similarity. Score: {}", r.score);
    }

    // =========================================================================
    // Phase 8: Temperature lifecycle
    // =========================================================================

    // Need mutable cluster for lifecycle ops
    let mut cluster2 = BankCluster::load_all(dir.path()).unwrap();

    // Promote semantic entry: Hot → Warm
    let bank = cluster2.get_mut(id_semantic).unwrap();
    assert!(bank.promote_entry(eid_semantic).unwrap());
    assert_eq!(bank.get(eid_semantic).unwrap().temperature, Temperature::Warm);

    // Persist and reload to verify promotion survives
    cluster2.flush_dirty(dir.path(), 200).unwrap();
    let loaded2 = BankCluster::load_all(dir.path()).unwrap();
    let entry = loaded2.get(id_semantic).unwrap().get(eid_semantic).unwrap();
    assert_eq!(entry.temperature, Temperature::Warm, "Promotion should survive save/load");

    // Evict from expression bank (which only has 1 entry, so evict it)
    let mut cluster3 = BankCluster::load_all(dir.path()).unwrap();
    let expr_bank = cluster3.get_mut(id_expression).unwrap();
    let evicted = expr_bank.evict_n(1, 300);
    assert_eq!(evicted, 1);
    assert_eq!(expr_bank.len(), 0);

    // Persist and verify eviction
    cluster3.flush_dirty(dir.path(), 300).unwrap();
    let loaded3 = BankCluster::load_all(dir.path()).unwrap();
    assert_eq!(loaded3.get(id_expression).unwrap().len(), 0, "Evicted entry should be gone after reload");
}
