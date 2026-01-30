//! BankAccess trait implementation for Ternsig 1.3 inline bank execution.
//!
//! Provides `ClusterBankAccess` which wraps a BankCluster + BankSlotMap
//! and implements the `ternsig::vm::extension::BankAccess` trait, enabling
//! TVMR firmware to execute bank operations inline without yielding DomainOps.
//!
//! Feature-gated: only compiled when the `ternsig` feature is enabled.

use ternsig::vm::extension::BankAccess;

use crate::bridge;
use crate::cluster::BankCluster;
use crate::fulfiller::BankSlotMap;
use crate::types::Temperature;

/// BankAccess implementation backed by a BankCluster + BankSlotMap.
///
/// The v3 kernel creates this per-tick and injects it into the interpreter.
/// It translates between the trait's i32-based interface and databank-rs's
/// Signal-based internal types using the bridge module.
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
    ) -> Self {
        Self {
            cluster,
            slot_map,
            temperature,
            tick,
        }
    }
}

impl BankAccess for ClusterBankAccess<'_> {
    fn query(&self, bank_slot: u8, query: &[i32], top_k: usize) -> Option<Vec<(i64, i32)>> {
        let bank_id = self.slot_map.resolve(bank_slot)?;
        let bank = self.cluster.get(bank_id)?;
        let signals = bridge::i32_to_signals(query);
        let results = bank.query_sparse(&signals, top_k);
        Some(
            results
                .iter()
                .map(|r| (r.entry_id.0 as i64, r.score))
                .collect(),
        )
    }

    fn load(&self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> Option<Vec<i32>> {
        let bank_id = self.slot_map.resolve(bank_slot)?;
        let bank = self.cluster.get(bank_id)?;
        let entry_id = bridge::i32_pair_to_entry_id(entry_id_high, entry_id_low);
        let entry = bank.get(entry_id)?;
        Some(bridge::signals_to_i32(&entry.vector))
    }

    fn count(&self, bank_slot: u8) -> Option<i32> {
        let bank_id = self.slot_map.resolve(bank_slot)?;
        let bank = self.cluster.get(bank_id)?;
        Some(bank.len() as i32)
    }

    fn write(&mut self, bank_slot: u8, vector: &[i32]) -> Option<(i32, i32)> {
        let bank_id = self.slot_map.resolve(bank_slot)?;
        let bank = self.cluster.get_mut(bank_id)?;
        let signals = bridge::i32_to_signals(vector);
        let entry_id = bank.insert(signals, self.temperature, self.tick).ok()?;
        Some(bridge::entry_id_to_i32_pair(entry_id))
    }

    fn touch(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) {
        let Some(bank_id) = self.slot_map.resolve(bank_slot) else {
            return;
        };
        let Some(bank) = self.cluster.get_mut(bank_id) else {
            return;
        };
        let entry_id = bridge::i32_pair_to_entry_id(entry_id_high, entry_id_low);
        if let Some(entry) = bank.get_mut(entry_id) {
            entry.touch(self.tick);
        }
    }

    fn delete(&mut self, bank_slot: u8, entry_id_high: i32, entry_id_low: i32) -> bool {
        let Some(bank_id) = self.slot_map.resolve(bank_slot) else {
            return false;
        };
        let Some(bank) = self.cluster.get_mut(bank_id) else {
            return false;
        };
        let entry_id = bridge::i32_pair_to_entry_id(entry_id_high, entry_id_low);
        bank.remove(entry_id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BankConfig, BankId};

    fn setup() -> (BankCluster, BankSlotMap, BankId) {
        let mut cluster = BankCluster::new();
        let bank_id = BankId::from_raw(42);
        let config = BankConfig {
            vector_width: 4,
            ..BankConfig::default()
        };
        cluster.get_or_create(bank_id, "test.access".to_string(), config);
        let mut slot_map = BankSlotMap::new();
        slot_map.bind(0, bank_id);
        (cluster, slot_map, bank_id)
    }

    #[test]
    fn test_write_and_count() {
        let (mut cluster, slot_map, _) = setup();
        let mut access = ClusterBankAccess::new(&mut cluster, &slot_map, Temperature::Hot, 1);

        assert_eq!(access.count(0), Some(0));
        let result = access.write(0, &[100, -50, 0, 200]);
        assert!(result.is_some());
        assert_eq!(access.count(0), Some(1));
    }

    #[test]
    fn test_write_and_load() {
        let (mut cluster, slot_map, _) = setup();
        let mut access = ClusterBankAccess::new(&mut cluster, &slot_map, Temperature::Hot, 1);

        let (hi, lo) = access.write(0, &[100, -50, 0, 200]).unwrap();
        let loaded = access.load(0, hi, lo).unwrap();
        assert_eq!(loaded, vec![100, -50, 0, 200]);
    }

    #[test]
    fn test_write_and_query() {
        let (mut cluster, slot_map, _) = setup();
        let mut access = ClusterBankAccess::new(&mut cluster, &slot_map, Temperature::Hot, 1);

        access.write(0, &[100, 100, 100, 100]).unwrap();
        let results = access.query(0, &[100, 100, 100, 100], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 200); // high similarity
    }

    #[test]
    fn test_touch_and_delete() {
        let (mut cluster, slot_map, _) = setup();
        let mut access = ClusterBankAccess::new(&mut cluster, &slot_map, Temperature::Hot, 1);

        let (hi, lo) = access.write(0, &[50, 50, 50, 50]).unwrap();
        access.touch(0, hi, lo);
        assert!(access.delete(0, hi, lo));
        assert_eq!(access.count(0), Some(0));
    }

    #[test]
    fn test_unbound_slot_returns_none() {
        let (mut cluster, slot_map, _) = setup();
        let access = ClusterBankAccess::new(&mut cluster, &slot_map, Temperature::Hot, 1);
        assert_eq!(access.count(99), None);
        assert_eq!(access.query(99, &[1, 2, 3, 4], 5), None);
    }
}
