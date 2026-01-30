//! Delta Journal for Crash Recovery
//!
//! Append-only mutation journal that records bank operations between full
//! `.bank` snapshots. On crash recovery, replay the journal on top of the
//! last snapshot to recover mutations.
//!
//! ## Binary Format (per entry)
//!
//! ```text
//! [0]       Tag (u8): 0=Insert, 1=Remove, 2=Touch, 3=AddEdge, 4=SetTemperature
//! [1..9]    BankId (u64 LE)
//! [9..17]   EntryId (u64 LE)
//! [17..]    Payload (variable, depends on tag)
//! [last 4]  CRC32 of all preceding bytes in this entry
//! ```

use crate::cluster::BankCluster;
use crate::types::{BankId, BankRef, Edge, EdgeType, EntryId, Temperature};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use ternary_signal::Signal;

/// A single journal entry: one mutation to a bank.
#[derive(Debug, Clone)]
pub enum JournalEntry {
    /// New entry inserted.
    Insert {
        bank_id: BankId,
        entry_id: EntryId,
        vector: Vec<Signal>,
        temperature: Temperature,
        tick: u64,
    },
    /// Entry removed.
    Remove {
        bank_id: BankId,
        entry_id: EntryId,
    },
    /// Entry touched (access count + tick update).
    Touch {
        bank_id: BankId,
        entry_id: EntryId,
        tick: u64,
    },
    /// Edge added to an entry.
    AddEdge {
        bank_id: BankId,
        entry_id: EntryId,
        edge: Edge,
    },
    /// Temperature changed.
    SetTemperature {
        bank_id: BankId,
        entry_id: EntryId,
        temperature: Temperature,
    },
    /// Entry promoted (temperature increased).
    Promote {
        bank_id: BankId,
        entry_id: EntryId,
        new_temp: Temperature,
    },
    /// Entry demoted (temperature decreased).
    Demote {
        bank_id: BankId,
        entry_id: EntryId,
        new_temp: Temperature,
    },
    /// Batch eviction (entries removed during sleep).
    BatchEvict {
        bank_id: BankId,
        entry_ids: Vec<EntryId>,
    },
}

// Tag constants
const TAG_INSERT: u8 = 0;
const TAG_REMOVE: u8 = 1;
const TAG_TOUCH: u8 = 2;
const TAG_ADD_EDGE: u8 = 3;
const TAG_SET_TEMP: u8 = 4;
const TAG_PROMOTE: u8 = 5;
const TAG_DEMOTE: u8 = 6;
const TAG_BATCH_EVICT: u8 = 7;

/// Append-only journal writer.
pub struct JournalWriter {
    writer: BufWriter<std::fs::File>,
}

impl JournalWriter {
    /// Open or create a journal file for appending.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Append a journal entry.
    pub fn append(&mut self, entry: &JournalEntry) -> io::Result<()> {
        let bytes = encode_entry(entry);
        self.writer.write_all(&bytes)?;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// Journal reader for replay during crash recovery.
pub struct JournalReader;

impl JournalReader {
    /// Read all valid entries from a journal file.
    /// Tolerates truncated final entry (crash mid-write).
    pub fn read_all(path: &Path) -> crate::Result<Vec<JournalEntry>> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(crate::DataBankError::Io(e)),
        };

        let mut entries = Vec::new();
        let mut cursor = 0;

        while cursor < data.len() {
            match decode_entry(&data[cursor..]) {
                Some((entry, consumed)) => {
                    entries.push(entry);
                    cursor += consumed;
                }
                None => {
                    // Truncated or corrupt entry — stop here
                    log::warn!(
                        "Journal truncated at byte {}/{}, recovered {} entries",
                        cursor,
                        data.len(),
                        entries.len()
                    );
                    break;
                }
            }
        }

        Ok(entries)
    }

    /// Replay journal entries onto an existing bank cluster.
    /// Returns count of entries replayed.
    pub fn replay(entries: &[JournalEntry], cluster: &mut BankCluster) -> crate::Result<usize> {
        let mut count = 0;
        for entry in entries {
            match entry {
                JournalEntry::Insert {
                    bank_id,
                    vector,
                    temperature,
                    tick,
                    ..
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        let _ = bank.insert(vector.clone(), *temperature, *tick);
                        count += 1;
                    }
                }
                JournalEntry::Remove {
                    bank_id, entry_id, ..
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        bank.remove(*entry_id);
                        count += 1;
                    }
                }
                JournalEntry::Touch {
                    bank_id,
                    entry_id,
                    tick,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        if let Some(entry) = bank.get_mut(*entry_id) {
                            entry.touch(*tick);
                            count += 1;
                        }
                    }
                }
                JournalEntry::AddEdge {
                    bank_id,
                    entry_id,
                    edge,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        let _ = bank.add_edge(*entry_id, edge.clone());
                        count += 1;
                    }
                }
                JournalEntry::SetTemperature {
                    bank_id,
                    entry_id,
                    temperature,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        if let Some(entry) = bank.get_mut(*entry_id) {
                            entry.temperature = *temperature;
                            count += 1;
                        }
                    }
                }
                JournalEntry::Promote {
                    bank_id,
                    entry_id,
                    new_temp,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        if let Some(entry) = bank.get_mut(*entry_id) {
                            entry.temperature = *new_temp;
                            count += 1;
                        }
                    }
                }
                JournalEntry::Demote {
                    bank_id,
                    entry_id,
                    new_temp,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        if let Some(entry) = bank.get_mut(*entry_id) {
                            entry.temperature = *new_temp;
                            count += 1;
                        }
                    }
                }
                JournalEntry::BatchEvict {
                    bank_id,
                    entry_ids,
                } => {
                    if let Some(bank) = cluster.get_mut(*bank_id) {
                        for eid in entry_ids {
                            bank.remove(*eid);
                        }
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }
}

/// Truncate (reset) a journal file after a full snapshot completes.
pub fn truncate_journal(path: &Path) -> io::Result<()> {
    if path.exists() {
        std::fs::write(path, &[])?;
    }
    Ok(())
}

// =============================================================================
// Binary encoding/decoding
// =============================================================================

fn encode_entry(entry: &JournalEntry) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    match entry {
        JournalEntry::Insert {
            bank_id,
            entry_id,
            vector,
            temperature,
            tick,
        } => {
            buf.push(TAG_INSERT);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.extend_from_slice(&(*tick).to_le_bytes());
            buf.push(temperature_to_u8(*temperature));
            buf.extend_from_slice(&(vector.len() as u16).to_le_bytes());
            for s in vector {
                buf.push(s.polarity as u8);
                buf.push(s.magnitude);
            }
        }
        JournalEntry::Remove { bank_id, entry_id } => {
            buf.push(TAG_REMOVE);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
        }
        JournalEntry::Touch {
            bank_id,
            entry_id,
            tick,
        } => {
            buf.push(TAG_TOUCH);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.extend_from_slice(&tick.to_le_bytes());
        }
        JournalEntry::AddEdge {
            bank_id,
            entry_id,
            edge,
        } => {
            buf.push(TAG_ADD_EDGE);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.push(edge.edge_type.as_u8());
            buf.extend_from_slice(&edge.target.bank.0.to_le_bytes());
            buf.extend_from_slice(&edge.target.entry.0.to_le_bytes());
            buf.push(edge.weight);
            buf.extend_from_slice(&edge.created_tick.to_le_bytes());
        }
        JournalEntry::SetTemperature {
            bank_id,
            entry_id,
            temperature,
        } => {
            buf.push(TAG_SET_TEMP);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.push(temperature_to_u8(*temperature));
        }
        JournalEntry::Promote {
            bank_id,
            entry_id,
            new_temp,
        } => {
            buf.push(TAG_PROMOTE);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.push(temperature_to_u8(*new_temp));
        }
        JournalEntry::Demote {
            bank_id,
            entry_id,
            new_temp,
        } => {
            buf.push(TAG_DEMOTE);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&entry_id.0.to_le_bytes());
            buf.push(temperature_to_u8(*new_temp));
        }
        JournalEntry::BatchEvict {
            bank_id,
            entry_ids,
        } => {
            buf.push(TAG_BATCH_EVICT);
            buf.extend_from_slice(&bank_id.0.to_le_bytes());
            buf.extend_from_slice(&(entry_ids.len() as u16).to_le_bytes());
            for eid in entry_ids {
                buf.extend_from_slice(&eid.0.to_le_bytes());
            }
        }
    }

    // Append CRC32
    let crc = crc32(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());
    buf
}

fn decode_entry(data: &[u8]) -> Option<(JournalEntry, usize)> {
    if data.is_empty() {
        return None;
    }
    let tag = data[0];

    match tag {
        TAG_INSERT => decode_insert(data),
        TAG_REMOVE => decode_remove(data),
        TAG_TOUCH => decode_touch(data),
        TAG_ADD_EDGE => decode_add_edge(data),
        TAG_SET_TEMP => decode_set_temp(data),
        TAG_PROMOTE => decode_promote(data),
        TAG_DEMOTE => decode_demote(data),
        TAG_BATCH_EVICT => decode_batch_evict(data),
        _ => None,
    }
}

fn decode_insert(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + entry_id(8) + tick(8) + temp(1) + vec_len(2) + signals(N*2) + crc(4)
    let min_len = 1 + 8 + 8 + 8 + 1 + 2 + 4;
    if data.len() < min_len {
        return None;
    }
    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let tick = u64::from_le_bytes(data[17..25].try_into().ok()?);
    let temperature = u8_to_temperature(data[25])?;
    let vec_len = u16::from_le_bytes(data[26..28].try_into().ok()?) as usize;

    let body_len = 28 + vec_len * 2;
    let total = body_len + 4; // + crc
    if data.len() < total {
        return None;
    }

    // Verify CRC
    let stored_crc = u32::from_le_bytes(data[body_len..total].try_into().ok()?);
    let computed_crc = crc32(&data[..body_len]);
    if stored_crc != computed_crc {
        return None;
    }

    let mut vector = Vec::with_capacity(vec_len);
    for i in 0..vec_len {
        let offset = 28 + i * 2;
        let polarity = data[offset] as i8;
        let magnitude = data[offset + 1];
        vector.push(Signal {
            polarity,
            magnitude,
        });
    }

    Some((
        JournalEntry::Insert {
            bank_id,
            entry_id,
            vector,
            temperature,
            tick,
        },
        total,
    ))
}

fn decode_remove(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + entry_id(8) + crc(4) = 21
    if data.len() < 21 {
        return None;
    }
    let body_len = 17;
    let stored_crc = u32::from_le_bytes(data[body_len..21].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }

    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));

    Some((JournalEntry::Remove { bank_id, entry_id }, 21))
}

fn decode_touch(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + entry_id(8) + tick(8) + crc(4) = 29
    if data.len() < 29 {
        return None;
    }
    let body_len = 25;
    let stored_crc = u32::from_le_bytes(data[body_len..29].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }

    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let tick = u64::from_le_bytes(data[17..25].try_into().ok()?);

    Some((
        JournalEntry::Touch {
            bank_id,
            entry_id,
            tick,
        },
        29,
    ))
}

fn decode_add_edge(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + entry_id(8) + edge_type(1) + target_bank(8) + target_entry(8) + weight(1) + tick(8) + crc(4) = 47
    if data.len() < 47 {
        return None;
    }
    let body_len = 43;
    let stored_crc = u32::from_le_bytes(data[body_len..47].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }

    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let edge_type = EdgeType::from_u8(data[17]).unwrap_or(EdgeType::RelatedTo);
    let target_bank = BankId(u64::from_le_bytes(data[18..26].try_into().ok()?));
    let target_entry = EntryId(u64::from_le_bytes(data[26..34].try_into().ok()?));
    let weight = data[34];
    let created_tick = u64::from_le_bytes(data[35..43].try_into().ok()?);

    Some((
        JournalEntry::AddEdge {
            bank_id,
            entry_id,
            edge: Edge {
                edge_type,
                target: BankRef {
                    bank: target_bank,
                    entry: target_entry,
                },
                weight,
                created_tick,
            },
        },
        47,
    ))
}

fn decode_set_temp(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + entry_id(8) + temp(1) + crc(4) = 22
    if data.len() < 22 {
        return None;
    }
    let body_len = 18;
    let stored_crc = u32::from_le_bytes(data[body_len..22].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }

    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let temperature = u8_to_temperature(data[17])?;

    Some((
        JournalEntry::SetTemperature {
            bank_id,
            entry_id,
            temperature,
        },
        22,
    ))
}

fn decode_promote(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // Same layout as SetTemperature: tag(1) + bank_id(8) + entry_id(8) + temp(1) + crc(4) = 22
    if data.len() < 22 {
        return None;
    }
    let body_len = 18;
    let stored_crc = u32::from_le_bytes(data[body_len..22].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }
    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let new_temp = u8_to_temperature(data[17])?;
    Some((JournalEntry::Promote { bank_id, entry_id, new_temp }, 22))
}

fn decode_demote(data: &[u8]) -> Option<(JournalEntry, usize)> {
    if data.len() < 22 {
        return None;
    }
    let body_len = 18;
    let stored_crc = u32::from_le_bytes(data[body_len..22].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }
    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let entry_id = EntryId(u64::from_le_bytes(data[9..17].try_into().ok()?));
    let new_temp = u8_to_temperature(data[17])?;
    Some((JournalEntry::Demote { bank_id, entry_id, new_temp }, 22))
}

fn decode_batch_evict(data: &[u8]) -> Option<(JournalEntry, usize)> {
    // tag(1) + bank_id(8) + count(2) + entry_ids(N*8) + crc(4)
    let min_len = 1 + 8 + 2 + 4;
    if data.len() < min_len {
        return None;
    }
    let bank_id = BankId(u64::from_le_bytes(data[1..9].try_into().ok()?));
    let count = u16::from_le_bytes(data[9..11].try_into().ok()?) as usize;
    let body_len = 11 + count * 8;
    let total = body_len + 4;
    if data.len() < total {
        return None;
    }
    let stored_crc = u32::from_le_bytes(data[body_len..total].try_into().ok()?);
    if stored_crc != crc32(&data[..body_len]) {
        return None;
    }
    let mut entry_ids = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 11 + i * 8;
        let eid = EntryId(u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?));
        entry_ids.push(eid);
    }
    Some((JournalEntry::BatchEvict { bank_id, entry_ids }, total))
}

// =============================================================================
// Helpers
// =============================================================================

fn temperature_to_u8(t: Temperature) -> u8 {
    match t {
        Temperature::Hot => 0,
        Temperature::Warm => 1,
        Temperature::Cool => 2,
        Temperature::Cold => 3,
    }
}

fn u8_to_temperature(v: u8) -> Option<Temperature> {
    match v {
        0 => Some(Temperature::Hot),
        1 => Some(Temperature::Warm),
        2 => Some(Temperature::Cool),
        3 => Some(Temperature::Cold),
        _ => None,
    }
}

/// Simple CRC32 (IEEE polynomial).
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(pol: i8, mag: u8) -> Signal {
        Signal {
            polarity: pol,
            magnitude: mag,
        }
    }

    #[test]
    fn test_insert_roundtrip() {
        let entry = JournalEntry::Insert {
            bank_id: BankId(12345),
            entry_id: EntryId(67890),
            vector: vec![make_signal(1, 100), make_signal(-1, 200)],
            temperature: Temperature::Warm,
            tick: 42,
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::Insert {
                bank_id,
                entry_id,
                vector,
                temperature,
                tick,
            } => {
                assert_eq!(bank_id, BankId(12345));
                assert_eq!(entry_id, EntryId(67890));
                assert_eq!(vector.len(), 2);
                assert_eq!(vector[0].polarity, 1);
                assert_eq!(vector[0].magnitude, 100);
                assert_eq!(temperature, Temperature::Warm);
                assert_eq!(tick, 42);
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_remove_roundtrip() {
        let entry = JournalEntry::Remove {
            bank_id: BankId(111),
            entry_id: EntryId(222),
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::Remove { bank_id, entry_id } => {
                assert_eq!(bank_id, BankId(111));
                assert_eq!(entry_id, EntryId(222));
            }
            _ => panic!("Expected Remove"),
        }
    }

    #[test]
    fn test_touch_roundtrip() {
        let entry = JournalEntry::Touch {
            bank_id: BankId(333),
            entry_id: EntryId(444),
            tick: 999,
        };
        let bytes = encode_entry(&entry);
        let (decoded, _) = decode_entry(&bytes).expect("should decode");
        match decoded {
            JournalEntry::Touch {
                bank_id,
                entry_id,
                tick,
            } => {
                assert_eq!(bank_id, BankId(333));
                assert_eq!(entry_id, EntryId(444));
                assert_eq!(tick, 999);
            }
            _ => panic!("Expected Touch"),
        }
    }

    #[test]
    fn test_add_edge_roundtrip() {
        let entry = JournalEntry::AddEdge {
            bank_id: BankId(100),
            entry_id: EntryId(200),
            edge: Edge {
                edge_type: EdgeType::IsA,
                target: BankRef {
                    bank: BankId(300),
                    entry: EntryId(400),
                },
                weight: 128,
                created_tick: 50,
            },
        };
        let bytes = encode_entry(&entry);
        let (decoded, _) = decode_entry(&bytes).expect("should decode");
        match decoded {
            JournalEntry::AddEdge {
                edge, entry_id, ..
            } => {
                assert_eq!(entry_id, EntryId(200));
                assert_eq!(edge.weight, 128);
                assert_eq!(edge.target.bank, BankId(300));
                assert_eq!(edge.target.entry, EntryId(400));
            }
            _ => panic!("Expected AddEdge"),
        }
    }

    #[test]
    fn test_set_temp_roundtrip() {
        let entry = JournalEntry::SetTemperature {
            bank_id: BankId(500),
            entry_id: EntryId(600),
            temperature: Temperature::Cool,
        };
        let bytes = encode_entry(&entry);
        let (decoded, _) = decode_entry(&bytes).expect("should decode");
        match decoded {
            JournalEntry::SetTemperature { temperature, .. } => {
                assert_eq!(temperature, Temperature::Cool);
            }
            _ => panic!("Expected SetTemperature"),
        }
    }

    #[test]
    fn test_multiple_entries_sequential() {
        let entries = vec![
            JournalEntry::Remove {
                bank_id: BankId(1),
                entry_id: EntryId(2),
            },
            JournalEntry::Touch {
                bank_id: BankId(3),
                entry_id: EntryId(4),
                tick: 5,
            },
            JournalEntry::SetTemperature {
                bank_id: BankId(6),
                entry_id: EntryId(7),
                temperature: Temperature::Cold,
            },
        ];

        let mut all_bytes = Vec::new();
        for e in &entries {
            all_bytes.extend(encode_entry(e));
        }

        // Decode sequentially
        let mut cursor = 0;
        let mut decoded_count = 0;
        while cursor < all_bytes.len() {
            let (_, consumed) = decode_entry(&all_bytes[cursor..]).expect("should decode");
            cursor += consumed;
            decoded_count += 1;
        }
        assert_eq!(decoded_count, 3);
    }

    #[test]
    fn test_truncated_entry_returns_none() {
        let entry = JournalEntry::Remove {
            bank_id: BankId(1),
            entry_id: EntryId(2),
        };
        let bytes = encode_entry(&entry);
        // Truncate by removing last 2 bytes (partial CRC)
        let truncated = &bytes[..bytes.len() - 2];
        assert!(decode_entry(truncated).is_none());
    }

    #[test]
    fn test_corrupt_crc_returns_none() {
        let entry = JournalEntry::Remove {
            bank_id: BankId(1),
            entry_id: EntryId(2),
        };
        let mut bytes = encode_entry(&entry);
        // Flip a byte in the CRC
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(decode_entry(&bytes).is_none());
    }

    #[test]
    fn test_promote_roundtrip() {
        let entry = JournalEntry::Promote {
            bank_id: BankId(700),
            entry_id: EntryId(800),
            new_temp: Temperature::Warm,
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::Promote { bank_id, entry_id, new_temp } => {
                assert_eq!(bank_id, BankId(700));
                assert_eq!(entry_id, EntryId(800));
                assert_eq!(new_temp, Temperature::Warm);
            }
            _ => panic!("Expected Promote"),
        }
    }

    #[test]
    fn test_demote_roundtrip() {
        let entry = JournalEntry::Demote {
            bank_id: BankId(900),
            entry_id: EntryId(1000),
            new_temp: Temperature::Hot,
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::Demote { bank_id, entry_id, new_temp } => {
                assert_eq!(bank_id, BankId(900));
                assert_eq!(entry_id, EntryId(1000));
                assert_eq!(new_temp, Temperature::Hot);
            }
            _ => panic!("Expected Demote"),
        }
    }

    #[test]
    fn test_batch_evict_roundtrip() {
        let entry = JournalEntry::BatchEvict {
            bank_id: BankId(1100),
            entry_ids: vec![EntryId(1), EntryId(2), EntryId(3)],
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::BatchEvict { bank_id, entry_ids } => {
                assert_eq!(bank_id, BankId(1100));
                assert_eq!(entry_ids.len(), 3);
                assert_eq!(entry_ids[0], EntryId(1));
                assert_eq!(entry_ids[2], EntryId(3));
            }
            _ => panic!("Expected BatchEvict"),
        }
    }

    #[test]
    fn test_batch_evict_empty_roundtrip() {
        let entry = JournalEntry::BatchEvict {
            bank_id: BankId(42),
            entry_ids: vec![],
        };
        let bytes = encode_entry(&entry);
        let (decoded, consumed) = decode_entry(&bytes).expect("should decode");
        assert_eq!(consumed, bytes.len());
        match decoded {
            JournalEntry::BatchEvict { entry_ids, .. } => {
                assert!(entry_ids.is_empty());
            }
            _ => panic!("Expected BatchEvict"),
        }
    }

    #[test]
    fn test_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.journal");

        // Write entries
        {
            let mut writer = JournalWriter::open(&path).unwrap();
            writer
                .append(&JournalEntry::Remove {
                    bank_id: BankId(1),
                    entry_id: EntryId(2),
                })
                .unwrap();
            writer
                .append(&JournalEntry::Touch {
                    bank_id: BankId(3),
                    entry_id: EntryId(4),
                    tick: 10,
                })
                .unwrap();
            writer.flush().unwrap();
        }

        // Read back
        let entries = JournalReader::read_all(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[0], JournalEntry::Remove { .. }));
        assert!(matches!(&entries[1], JournalEntry::Touch { .. }));

        // Truncate
        truncate_journal(&path).unwrap();
        let after = JournalReader::read_all(&path).unwrap();
        assert_eq!(after.len(), 0);
    }
}
