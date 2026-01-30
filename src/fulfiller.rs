//! Bank DomainOp Fulfillment
//!
//! Stateless helpers that the v3 kernel calls when firmware yields bank
//! DomainOps. Maps per-interpreter bank slots to global BankIds and
//! converts between register i32 format and Signal vectors.

use crate::bridge;
use crate::cluster::BankCluster;
use crate::types::{BankId, Edge, EdgeType, EntryId, Temperature};

/// Maps per-interpreter bank_slot (u8) to global BankId.
/// The kernel initializes this per-region during boot.
pub struct BankSlotMap {
    slots: [Option<BankId>; 256],
}

impl BankSlotMap {
    pub fn new() -> Self {
        Self {
            slots: [None; 256],
        }
    }

    /// Bind a slot index to a global BankId.
    pub fn bind(&mut self, slot: u8, bank_id: BankId) {
        self.slots[slot as usize] = Some(bank_id);
    }

    /// Resolve a slot index to a global BankId.
    pub fn resolve(&self, slot: u8) -> Option<BankId> {
        self.slots[slot as usize]
    }

    /// Unbind a slot.
    pub fn unbind(&mut self, slot: u8) {
        self.slots[slot as usize] = None;
    }
}

impl Default for BankSlotMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of fulfilling a bank DomainOp.
#[derive(Debug, Clone)]
pub enum FulfillResult {
    /// Write data into target hot register.
    WriteRegister {
        register_index: u8,
        data: Vec<i32>,
        shape: Vec<usize>,
    },
    /// No register output (write-only ops like BankLink, BankTouch, BankDelete).
    Ok,
    /// Error during fulfillment.
    Error(String),
}

/// Stateless fulfiller for bank DomainOps.
pub struct BankFulfiller;

impl BankFulfiller {
    /// Fulfill a BankQuery DomainOp.
    pub fn query(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        top_k: u8,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        let query_signals = bridge::i32_to_signals(source_data);
        let results = bank.query_sparse(&query_signals, top_k as usize);
        let packed = bridge::query_results_to_i32(&results);
        let len = packed.len();

        FulfillResult::WriteRegister {
            register_index: 0, // caller sets this from the DomainOp target
            data: packed,
            shape: vec![len],
        }
    }

    /// Fulfill a BankWrite DomainOp.
    pub fn write(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        temperature: Temperature,
        tick: u64,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        let vector = bridge::i32_to_signals(source_data);
        match bank.insert(vector, temperature, tick) {
            Ok(entry_id) => {
                let (high, low) = bridge::entry_id_to_i32_pair(entry_id);
                FulfillResult::WriteRegister {
                    register_index: 0,
                    data: vec![high, low],
                    shape: vec![2],
                }
            }
            Err(e) => FulfillResult::Error(format!("BankWrite failed: {}", e)),
        }
    }

    /// Fulfill a BankLoad DomainOp.
    pub fn load(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        if source_data.len() < 2 {
            return FulfillResult::Error("BankLoad: source must have [id_high, id_low]".into());
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        match bank.get(entry_id) {
            Some(entry) => {
                let data = bridge::signals_to_i32(&entry.vector);
                let len = data.len();
                FulfillResult::WriteRegister {
                    register_index: 0,
                    data,
                    shape: vec![len],
                }
            }
            None => FulfillResult::Error(format!("Entry {:?} not found", entry_id)),
        }
    }

    /// Fulfill a BankLink DomainOp.
    pub fn link(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        edge_type: u8,
        tick: u64,
    ) -> FulfillResult {
        if source_data.len() < 6 {
            return FulfillResult::Error(
                "BankLink: source must have [from_hi, from_lo, to_slot, to_hi, to_lo, weight]"
                    .into(),
            );
        }

        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let from_entry = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        let to_slot = source_data[2] as u8;
        let to_bank_id = match slot_map.resolve(to_slot) {
            Some(id) => id,
            None => {
                return FulfillResult::Error(format!("Target bank slot {} not bound", to_slot))
            }
        };
        let to_entry = bridge::i32_pair_to_entry_id(source_data[3], source_data[4]);
        let weight = source_data[5].clamp(0, 255) as u8;

        let et = EdgeType::from_u8(edge_type).unwrap_or(EdgeType::RelatedTo);
        let edge = Edge {
            edge_type: et,
            target: crate::types::BankRef {
                bank: to_bank_id,
                entry: to_entry,
            },
            weight,
            created_tick: tick,
        };

        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        match bank.add_edge(from_entry, edge) {
            Ok(()) => FulfillResult::Ok,
            Err(e) => FulfillResult::Error(format!("BankLink failed: {}", e)),
        }
    }

    /// Fulfill a BankTraverse DomainOp.
    pub fn traverse(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        edge_type: u8,
        depth: u8,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };

        if source_data.len() < 2 {
            return FulfillResult::Error(
                "BankTraverse: source must have [id_high, id_low]".into(),
            );
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        let et = EdgeType::from_u8(edge_type).unwrap_or(EdgeType::RelatedTo);

        let start = crate::types::BankRef {
            bank: bank_id,
            entry: entry_id,
        };
        let refs = cluster.traverse(start, et, depth as usize);

        // Convert BankRefs to (slot, EntryId) pairs using reverse slot lookup
        let mut results: Vec<(u8, EntryId)> = Vec::new();
        for bref in &refs {
            // Find the slot for this BankId
            let mut found_slot = None;
            for s in 0..=255u8 {
                if slot_map.resolve(s) == Some(bref.bank) {
                    found_slot = Some(s);
                    break;
                }
            }
            if let Some(s) = found_slot {
                results.push((s, bref.entry));
            }
            // Skip refs whose banks aren't in the slot map
        }

        let packed = bridge::traverse_results_to_i32(&results);
        let len = packed.len();
        FulfillResult::WriteRegister {
            register_index: 0,
            data: packed,
            shape: vec![len],
        }
    }

    /// Fulfill a BankTouch DomainOp.
    pub fn touch(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
        tick: u64,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        if source_data.len() < 2 {
            return FulfillResult::Error("BankTouch: source must have [id_high, id_low]".into());
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        match bank.get_mut(entry_id) {
            Some(entry) => {
                entry.touch(tick);
                FulfillResult::Ok
            }
            None => FulfillResult::Error(format!("Entry {:?} not found", entry_id)),
        }
    }

    /// Fulfill a BankDelete DomainOp.
    pub fn delete(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        if source_data.len() < 2 {
            return FulfillResult::Error("BankDelete: source must have [id_high, id_low]".into());
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        match bank.remove(entry_id) {
            Some(_) => FulfillResult::Ok,
            None => FulfillResult::Error(format!("Entry {:?} not found", entry_id)),
        }
    }

    /// Fulfill BankPromote: promote entry temperature.
    /// source_data: [entry_id_high, entry_id_low]
    pub fn promote(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };
        if source_data.len() < 2 {
            return FulfillResult::Error("BankPromote: source must have [id_high, id_low]".into());
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        match bank.promote_entry(entry_id) {
            Ok(_) => FulfillResult::Ok,
            Err(e) => FulfillResult::Error(format!("BankPromote failed: {}", e)),
        }
    }

    /// Fulfill BankDemote: demote entry temperature.
    /// source_data: [entry_id_high, entry_id_low]
    pub fn demote(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        source_data: &[i32],
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };
        if source_data.len() < 2 {
            return FulfillResult::Error("BankDemote: source must have [id_high, id_low]".into());
        }
        let entry_id = bridge::i32_pair_to_entry_id(source_data[0], source_data[1]);
        match bank.demote_entry(entry_id) {
            Ok(_) => FulfillResult::Ok,
            Err(e) => FulfillResult::Error(format!("BankDemote failed: {}", e)),
        }
    }

    /// Fulfill BankEvict: evict lowest-scoring entries.
    pub fn evict(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
        count: u8,
        current_tick: u64,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };
        bank.evict_n(count as usize, current_tick);
        FulfillResult::Ok
    }

    /// Fulfill BankCompact: compact bank after eviction.
    pub fn compact(
        cluster: &mut BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get_mut(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };
        bank.compact();
        FulfillResult::Ok
    }

    /// Fulfill a BankCount DomainOp.
    pub fn count(
        cluster: &BankCluster,
        slot_map: &BankSlotMap,
        bank_slot: u8,
    ) -> FulfillResult {
        let bank_id = match slot_map.resolve(bank_slot) {
            Some(id) => id,
            None => return FulfillResult::Error(format!("Bank slot {} not bound", bank_slot)),
        };
        let bank = match cluster.get(bank_id) {
            Some(b) => b,
            None => return FulfillResult::Error(format!("Bank {:?} not found", bank_id)),
        };

        FulfillResult::WriteRegister {
            register_index: 0,
            data: vec![bank.len() as i32],
            shape: vec![1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BankConfig;
    use ternary_signal::Signal;

    fn make_signal(pol: i8, mag: u8) -> Signal {
        Signal {
            polarity: pol,
            magnitude: mag,
        }
    }

    fn setup_cluster() -> (BankCluster, BankSlotMap, BankId) {
        let mut cluster = BankCluster::new();
        let bank_id = BankId::new("test.semantic", 0);
        let config = BankConfig {
            vector_width: 4,
            ..BankConfig::default()
        };
        cluster.get_or_create(bank_id, "test.semantic".to_string(), config);
        let mut slot_map = BankSlotMap::new();
        slot_map.bind(0, bank_id);
        (cluster, slot_map, bank_id)
    }

    #[test]
    fn test_write_and_count() {
        let (mut cluster, slot_map, _) = setup_cluster();

        // Write an entry
        let source = bridge::signals_to_i32(&[
            make_signal(1, 100),
            make_signal(-1, 50),
            make_signal(0, 0),
            make_signal(1, 200),
        ]);
        let result = BankFulfiller::write(
            &mut cluster,
            &slot_map,
            0,
            &source,
            Temperature::Hot,
            1,
        );
        assert!(matches!(result, FulfillResult::WriteRegister { .. }));

        // Count should be 1
        let count = BankFulfiller::count(&cluster, &slot_map, 0);
        match count {
            FulfillResult::WriteRegister { data, .. } => assert_eq!(data[0], 1),
            other => panic!("Expected WriteRegister, got {:?}", other),
        }
    }

    #[test]
    fn test_write_load_roundtrip() {
        let (mut cluster, slot_map, _) = setup_cluster();

        let signals = vec![
            make_signal(1, 100),
            make_signal(-1, 50),
            make_signal(0, 0),
            make_signal(1, 200),
        ];
        let source = bridge::signals_to_i32(&signals);

        // Write
        let write_result =
            BankFulfiller::write(&mut cluster, &slot_map, 0, &source, Temperature::Hot, 1);
        let entry_data = match write_result {
            FulfillResult::WriteRegister { data, .. } => data,
            other => panic!("Expected WriteRegister, got {:?}", other),
        };
        assert_eq!(entry_data.len(), 2); // [id_high, id_low]

        // Load
        let load_result = BankFulfiller::load(&cluster, &slot_map, 0, &entry_data);
        match load_result {
            FulfillResult::WriteRegister { data, .. } => {
                assert_eq!(data, source);
            }
            other => panic!("Expected WriteRegister, got {:?}", other),
        }
    }

    #[test]
    fn test_query() {
        let (mut cluster, slot_map, _) = setup_cluster();

        // Insert a known pattern
        let pattern = bridge::signals_to_i32(&[
            make_signal(1, 200),
            make_signal(1, 200),
            make_signal(1, 200),
            make_signal(1, 200),
        ]);
        BankFulfiller::write(&mut cluster, &slot_map, 0, &pattern, Temperature::Hot, 1);

        // Query with partial cue (same direction)
        let query = bridge::signals_to_i32(&[
            make_signal(1, 100),
            make_signal(0, 0), // sparse: skip this
            make_signal(1, 100),
            make_signal(0, 0), // sparse: skip this
        ]);
        let result = BankFulfiller::query(&cluster, &slot_map, 0, &query, 5);
        match result {
            FulfillResult::WriteRegister { data, .. } => {
                assert!(data[0] >= 1, "Should find at least 1 result");
                assert!(data[1] > 0, "Score should be positive (same direction)");
            }
            other => panic!("Expected WriteRegister, got {:?}", other),
        }
    }

    #[test]
    fn test_touch_and_delete() {
        let (mut cluster, slot_map, _) = setup_cluster();

        let source = bridge::signals_to_i32(&[
            make_signal(1, 100),
            make_signal(1, 100),
            make_signal(1, 100),
            make_signal(1, 100),
        ]);
        let write_result =
            BankFulfiller::write(&mut cluster, &slot_map, 0, &source, Temperature::Hot, 1);
        let entry_data = match write_result {
            FulfillResult::WriteRegister { data, .. } => data,
            _ => panic!("write failed"),
        };

        // Touch
        let touch_result = BankFulfiller::touch(&mut cluster, &slot_map, 0, &entry_data, 10);
        assert!(matches!(touch_result, FulfillResult::Ok));

        // Delete
        let del_result = BankFulfiller::delete(&mut cluster, &slot_map, 0, &entry_data);
        assert!(matches!(del_result, FulfillResult::Ok));

        // Count should be 0
        match BankFulfiller::count(&cluster, &slot_map, 0) {
            FulfillResult::WriteRegister { data, .. } => assert_eq!(data[0], 0),
            _ => panic!("count failed"),
        }
    }

    #[test]
    fn test_promote_and_demote() {
        let (mut cluster, slot_map, _) = setup_cluster();
        let source = bridge::signals_to_i32(&[
            make_signal(1, 100),
            make_signal(1, 100),
            make_signal(1, 100),
            make_signal(1, 100),
        ]);
        let write_result =
            BankFulfiller::write(&mut cluster, &slot_map, 0, &source, Temperature::Hot, 1);
        let entry_data = match write_result {
            FulfillResult::WriteRegister { data, .. } => data,
            _ => panic!("write failed"),
        };

        // Promote
        let result = BankFulfiller::promote(&mut cluster, &slot_map, 0, &entry_data);
        assert!(matches!(result, FulfillResult::Ok));

        // Demote
        let result = BankFulfiller::demote(&mut cluster, &slot_map, 0, &entry_data);
        assert!(matches!(result, FulfillResult::Ok));
    }

    #[test]
    fn test_evict_and_compact() {
        let (mut cluster, slot_map, _) = setup_cluster();
        // Insert 3 entries
        for _ in 0..3 {
            let source = bridge::signals_to_i32(&[
                make_signal(1, 100),
                make_signal(1, 100),
                make_signal(1, 100),
                make_signal(1, 100),
            ]);
            BankFulfiller::write(&mut cluster, &slot_map, 0, &source, Temperature::Hot, 1);
        }
        // Count = 3
        match BankFulfiller::count(&cluster, &slot_map, 0) {
            FulfillResult::WriteRegister { data, .. } => assert_eq!(data[0], 3),
            _ => panic!("count failed"),
        }

        // Evict 1
        let result = BankFulfiller::evict(&mut cluster, &slot_map, 0, 1, 100);
        assert!(matches!(result, FulfillResult::Ok));
        match BankFulfiller::count(&cluster, &slot_map, 0) {
            FulfillResult::WriteRegister { data, .. } => assert_eq!(data[0], 2),
            _ => panic!("count failed"),
        }

        // Compact
        let result = BankFulfiller::compact(&mut cluster, &slot_map, 0);
        assert!(matches!(result, FulfillResult::Ok));
    }

    #[test]
    fn test_unbound_slot_error() {
        let cluster = BankCluster::new();
        let slot_map = BankSlotMap::new();

        let result = BankFulfiller::count(&cluster, &slot_map, 42);
        assert!(matches!(result, FulfillResult::Error(_)));
    }
}
