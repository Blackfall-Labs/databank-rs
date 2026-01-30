//! Signal/Register Conversion Bridge
//!
//! Converts between Signal vectors (databank-rs internal format) and
//! i32 register slices (TVMR firmware format). Also packs EntryId (u64)
//! into i32 pairs for register transport.

use crate::similarity::QueryResult;
use crate::types::EntryId;
use ternary_signal::Signal;

/// Convert a Signal vector to i32 register values.
/// Each Signal becomes: (polarity as i32) * (magnitude as i32)
/// Range: [-255, 255]
pub fn signals_to_i32(signals: &[Signal]) -> Vec<i32> {
    signals
        .iter()
        .map(|s| s.polarity as i32 * s.magnitude as i32)
        .collect()
}

/// Convert i32 register values back to Signal vector.
/// Clamps to [-255, 255], splits into polarity + magnitude.
pub fn i32_to_signals(values: &[i32]) -> Vec<Signal> {
    values
        .iter()
        .map(|&v| {
            let clamped = v.clamp(-255, 255);
            let polarity = if clamped > 0 {
                1
            } else if clamped < 0 {
                -1
            } else {
                0
            };
            let magnitude = clamped.unsigned_abs() as u8;
            Signal {
                polarity,
                magnitude,
            }
        })
        .collect()
}

/// Pack an EntryId (u64) into two i32 values (high, low).
pub fn entry_id_to_i32_pair(id: EntryId) -> (i32, i32) {
    let raw = id.0;
    let high = (raw >> 32) as i32;
    let low = (raw & 0xFFFF_FFFF) as i32;
    (high, low)
}

/// Unpack two i32 values into an EntryId.
pub fn i32_pair_to_entry_id(high: i32, low: i32) -> EntryId {
    let raw = ((high as u64) << 32) | (low as u32 as u64);
    EntryId(raw)
}

/// Pack BankRef-like data into i32 slice: [bank_slot, entry_id_high, entry_id_low].
pub fn bank_ref_to_i32_slice(slot: u8, entry: EntryId) -> [i32; 3] {
    let (high, low) = entry_id_to_i32_pair(entry);
    [slot as i32, high, low]
}

/// Pack QueryResult list into i32 register layout:
///   [count, score_0, id_high_0, id_low_0, score_1, ...]
pub fn query_results_to_i32(results: &[QueryResult]) -> Vec<i32> {
    let mut out = Vec::with_capacity(1 + results.len() * 3);
    out.push(results.len() as i32);
    for r in results {
        out.push(r.score);
        let (high, low) = entry_id_to_i32_pair(r.entry_id);
        out.push(high);
        out.push(low);
    }
    out
}

/// Pack traverse results (Vec<(u8, EntryId)>) into i32 register layout:
///   [count, slot_0, id_high_0, id_low_0, slot_1, ...]
pub fn traverse_results_to_i32(results: &[(u8, EntryId)]) -> Vec<i32> {
    let mut out = Vec::with_capacity(1 + results.len() * 3);
    out.push(results.len() as i32);
    for &(slot, entry_id) in results {
        out.push(slot as i32);
        let (high, low) = entry_id_to_i32_pair(entry_id);
        out.push(high);
        out.push(low);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_to_i32_roundtrip() {
        let signals = vec![
            Signal { polarity: 1, magnitude: 200 },
            Signal { polarity: -1, magnitude: 128 },
            Signal { polarity: 0, magnitude: 0 },
            Signal { polarity: 1, magnitude: 1 },
        ];
        let i32s = signals_to_i32(&signals);
        assert_eq!(i32s, vec![200, -128, 0, 1]);

        let back = i32_to_signals(&i32s);
        assert_eq!(back[0].polarity, 1);
        assert_eq!(back[0].magnitude, 200);
        assert_eq!(back[1].polarity, -1);
        assert_eq!(back[1].magnitude, 128);
        assert_eq!(back[2].polarity, 0);
        assert_eq!(back[2].magnitude, 0);
        assert_eq!(back[3].polarity, 1);
        assert_eq!(back[3].magnitude, 1);
    }

    #[test]
    fn test_i32_to_signal_clamping() {
        let values = vec![500, -500, 0, 255, -255];
        let signals = i32_to_signals(&values);
        assert_eq!(signals[0].polarity, 1);
        assert_eq!(signals[0].magnitude, 255);
        assert_eq!(signals[1].polarity, -1);
        assert_eq!(signals[1].magnitude, 255);
    }

    #[test]
    fn test_entry_id_packing() {
        let id = EntryId(0x0123456789ABCDEF);
        let (high, low) = entry_id_to_i32_pair(id);
        let back = i32_pair_to_entry_id(high, low);
        assert_eq!(back, id);
    }

    #[test]
    fn test_entry_id_zero() {
        let id = EntryId(0);
        let (high, low) = entry_id_to_i32_pair(id);
        assert_eq!(high, 0);
        assert_eq!(low, 0);
        let back = i32_pair_to_entry_id(high, low);
        assert_eq!(back, id);
    }

    #[test]
    fn test_entry_id_max() {
        let id = EntryId(u64::MAX);
        let (high, low) = entry_id_to_i32_pair(id);
        let back = i32_pair_to_entry_id(high, low);
        assert_eq!(back, id);
    }

    #[test]
    fn test_query_results_packing() {
        let results = vec![
            QueryResult { entry_id: EntryId(100), score: 200 },
            QueryResult { entry_id: EntryId(200), score: 150 },
        ];
        let packed = query_results_to_i32(&results);
        assert_eq!(packed[0], 2); // count
        assert_eq!(packed[1], 200); // score_0
        // id_high_0, id_low_0
        let (h, l) = entry_id_to_i32_pair(EntryId(100));
        assert_eq!(packed[2], h);
        assert_eq!(packed[3], l);
        assert_eq!(packed[4], 150); // score_1
    }

    #[test]
    fn test_traverse_results_packing() {
        let results = vec![(0u8, EntryId(42)), (3u8, EntryId(99))];
        let packed = traverse_results_to_i32(&results);
        assert_eq!(packed[0], 2); // count
        assert_eq!(packed[1], 0); // slot_0
        let (h, l) = entry_id_to_i32_pair(EntryId(42));
        assert_eq!(packed[2], h);
        assert_eq!(packed[3], l);
        assert_eq!(packed[4], 3); // slot_1
    }

    #[test]
    fn test_bank_ref_slice() {
        let id = EntryId(12345);
        let slice = bank_ref_to_i32_slice(5, id);
        assert_eq!(slice[0], 5);
        let back = i32_pair_to_entry_id(slice[1], slice[2]);
        assert_eq!(back, id);
    }
}
