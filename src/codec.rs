//! Binary `.bank` v1 format codec.
//!
//! Header (32 bytes):
//! ```text
//! [0..4]   Magic: b"BANK"
//! [4..6]   Version: u16 LE = 1
//! [6..8]   Flags: u16 LE = 0
//! [8..12]  Total size: u32 LE (patched after encode)
//! [12..20] Checksum: u64 LE xxhash64 (patched after encode)
//! [20..28] BankId: u64 LE
//! [28..30] Vector width: u16 LE
//! [30..32] Entry count: u16 LE
//! ```
//!
//! Followed by: bank name, config, entries (with edges), and state counters.

use std::collections::HashMap;
use std::path::Path;

use ternary_signal::Signal;

use crate::bank::DataBank;
use crate::entry::BankEntry;
use crate::error::{DataBankError, Result};
use crate::types::*;

const MAGIC: &[u8; 4] = b"BANK";
const VERSION: u16 = 1;
const HEADER_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a DataBank into the binary `.bank` v1 format.
pub fn encode(bank: &DataBank) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(4096);

    // -- Header (32 bytes, with placeholders for size + checksum) --
    buf.extend_from_slice(MAGIC);
    write_u16(&mut buf, VERSION);
    write_u16(&mut buf, 0); // flags
    write_u32(&mut buf, 0); // total_size placeholder
    write_u64(&mut buf, 0); // checksum placeholder
    write_u64(&mut buf, bank.id.0);
    write_u16(&mut buf, bank.config().vector_width);
    write_u16(&mut buf, bank.len() as u16);

    // -- Bank name --
    write_str(&mut buf, &bank.name);

    // -- Config --
    write_u32(&mut buf, bank.config().persist_after_mutations);
    write_u64(&mut buf, bank.config().persist_after_ticks);
    write_u32(&mut buf, bank.config().max_entries);
    write_u16(&mut buf, bank.config().vector_width);
    write_u16(&mut buf, bank.config().max_edges_per_entry);

    // -- Entries --
    for (_, entry) in bank.entries() {
        encode_entry(&mut buf, entry);
    }

    // -- State counters --
    write_u32(&mut buf, bank.next_seq());
    write_u32(&mut buf, bank.mutations_since_persist());
    write_u64(&mut buf, bank.last_persist_tick());

    // -- Patch header --
    let total_size = buf.len() as u32;
    buf[8..12].copy_from_slice(&total_size.to_le_bytes());

    let checksum = xxhash_rust::xxh3::xxh3_64(&buf[HEADER_SIZE..]);
    buf[12..20].copy_from_slice(&checksum.to_le_bytes());

    Ok(buf)
}

fn encode_entry(buf: &mut Vec<u8>, entry: &BankEntry) {
    // EntryId
    write_u64(buf, entry.id.0);

    // Vector
    write_u16(buf, entry.vector.len() as u16);
    for s in &entry.vector {
        buf.push(s.polarity as u8);
        buf.push(s.magnitude);
    }

    // Edges
    write_u16(buf, entry.edges.len() as u16);
    for edge in &entry.edges {
        buf.push(edge.edge_type.as_u8());
        write_u64(buf, edge.target.bank.0);
        write_u64(buf, edge.target.entry.0);
        buf.push(edge.weight);
        write_u64(buf, edge.created_tick);
    }

    // Origin bank
    write_u64(buf, entry.origin.0);

    // Temperature
    buf.push(entry.temperature.as_u8());

    // Ticks
    write_u64(buf, entry.created_tick);
    write_u64(buf, entry.last_accessed_tick);

    // Access count + confidence
    write_u32(buf, entry.access_count);
    buf.push(entry.confidence);

    // Debug tag
    match &entry.debug_tag {
        Some(tag) => {
            buf.push(1);
            write_str(buf, tag);
        }
        None => buf.push(0),
    }

    // Checksum
    write_u32(buf, entry.checksum);
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a binary `.bank` v1 buffer into a DataBank.
pub fn decode(data: &[u8]) -> Result<DataBank> {
    if data.len() < HEADER_SIZE {
        return Err(DataBankError::Codec("data too short for header".into()));
    }

    // -- Header --
    if &data[0..4] != MAGIC {
        return Err(DataBankError::Codec(format!(
            "bad magic: expected BANK, got {:?}",
            &data[0..4]
        )));
    }

    let mut pos = 4;
    let version = read_u16(data, &mut pos);
    if version != VERSION {
        return Err(DataBankError::Codec(format!(
            "unsupported version: {version}"
        )));
    }

    let _flags = read_u16(data, &mut pos);
    let total_size = read_u32(data, &mut pos);
    if data.len() < total_size as usize {
        return Err(DataBankError::Codec(format!(
            "truncated: expected {total_size} bytes, got {}",
            data.len()
        )));
    }

    let stored_checksum = read_u64(data, &mut pos);
    let bank_id = BankId(read_u64(data, &mut pos));
    let vector_width = read_u16(data, &mut pos);
    let entry_count = read_u16(data, &mut pos);

    // Verify checksum
    let computed_checksum = xxhash_rust::xxh3::xxh3_64(&data[HEADER_SIZE..total_size as usize]);
    if stored_checksum != computed_checksum {
        return Err(DataBankError::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed_checksum,
        });
    }

    // -- Bank name --
    let name = read_str(data, &mut pos)?;

    // -- Config --
    let persist_after_mutations = read_u32(data, &mut pos);
    let persist_after_ticks = read_u64(data, &mut pos);
    let max_entries = read_u32(data, &mut pos);
    let cfg_vector_width = read_u16(data, &mut pos);
    let max_edges_per_entry = read_u16(data, &mut pos);

    let config = BankConfig {
        persist_after_mutations,
        persist_after_ticks,
        max_entries,
        vector_width: cfg_vector_width,
        max_edges_per_entry,
    };

    // -- Entries --
    let mut entries = HashMap::with_capacity(entry_count as usize);
    let mut reverse_edges: HashMap<EntryId, Vec<(BankRef, EdgeType)>> = HashMap::new();

    for _ in 0..entry_count {
        let entry = decode_entry(data, &mut pos, vector_width, bank_id)?;

        // Rebuild reverse edges
        for edge in &entry.edges {
            reverse_edges
                .entry(edge.target.entry)
                .or_default()
                .push((
                    BankRef {
                        bank: bank_id,
                        entry: entry.id,
                    },
                    edge.edge_type,
                ));
        }

        entries.insert(entry.id, entry);
    }

    // -- State counters --
    let next_seq = read_u32(data, &mut pos);
    let mutations_since_persist = read_u32(data, &mut pos);
    let last_persist_tick = read_u64(data, &mut pos);

    Ok(DataBank::restore(
        bank_id,
        name,
        config,
        entries,
        reverse_edges,
        next_seq,
        mutations_since_persist,
        last_persist_tick,
    ))
}

fn decode_entry(
    data: &[u8],
    pos: &mut usize,
    expected_width: u16,
    bank_id: BankId,
) -> Result<BankEntry> {
    let entry_id = EntryId(read_u64(data, pos));

    // Vector
    let vec_len = read_u16(data, pos) as usize;
    if vec_len != expected_width as usize {
        return Err(DataBankError::Codec(format!(
            "entry vector width {vec_len} != bank width {expected_width}"
        )));
    }
    let mut vector = Vec::with_capacity(vec_len);
    for _ in 0..vec_len {
        let polarity = read_u8(data, pos) as i8;
        let magnitude = read_u8(data, pos);
        vector.push(Signal::new(polarity, magnitude));
    }

    // Edges
    let edge_count = read_u16(data, pos) as usize;
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let edge_type_raw = read_u8(data, pos);
        let edge_type = EdgeType::from_u8(edge_type_raw).ok_or_else(|| {
            DataBankError::Codec(format!("invalid edge type: {edge_type_raw}"))
        })?;
        let target_bank = BankId(read_u64(data, pos));
        let target_entry = EntryId(read_u64(data, pos));
        let weight = read_u8(data, pos);
        let created_tick = read_u64(data, pos);
        edges.push(Edge {
            edge_type,
            target: BankRef {
                bank: target_bank,
                entry: target_entry,
            },
            weight,
            created_tick,
        });
    }

    // Origin
    let origin = BankId(read_u64(data, pos));
    let _ = bank_id; // origin comes from the entry itself

    // Temperature
    let temp_raw = read_u8(data, pos);
    let temperature = Temperature::from_u8(temp_raw)
        .ok_or_else(|| DataBankError::Codec(format!("invalid temperature: {temp_raw}")))?;

    // Ticks
    let created_tick = read_u64(data, pos);
    let last_accessed_tick = read_u64(data, pos);

    // Access + confidence
    let access_count = read_u32(data, pos);
    let confidence = read_u8(data, pos);

    // Debug tag
    let has_tag = read_u8(data, pos);
    let debug_tag = if has_tag != 0 {
        Some(read_str(data, pos)?)
    } else {
        None
    };

    // Checksum
    let checksum = read_u32(data, pos);

    Ok(BankEntry {
        id: entry_id,
        vector,
        edges,
        origin,
        temperature,
        created_tick,
        last_accessed_tick,
        access_count,
        confidence,
        debug_tag,
        checksum,
    })
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Save a bank to disk atomically (temp file + rename).
pub fn save_atomic(bank: &DataBank, path: &Path) -> Result<()> {
    let data = encode(bank)?;
    let temp = path.with_extension("bank.tmp");

    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&temp, &data)?;
    std::fs::rename(&temp, path)?;
    Ok(())
}

/// Load a bank from a `.bank` file.
pub fn load(path: &Path) -> Result<DataBank> {
    let data = std::fs::read(path)?;
    decode(&data)
}

// ---------------------------------------------------------------------------
// Primitive read/write helpers (little-endian)
// ---------------------------------------------------------------------------

fn write_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    write_u16(buf, s.len() as u16);
    buf.extend_from_slice(s.as_bytes());
}

fn read_u8(data: &[u8], pos: &mut usize) -> u8 {
    let v = data[*pos];
    *pos += 1;
    v
}

fn read_u16(data: &[u8], pos: &mut usize) -> u16 {
    let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
    *pos += 2;
    v
}

fn read_u32(data: &[u8], pos: &mut usize) -> u32 {
    let v = u32::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
    ]);
    *pos += 4;
    v
}

fn read_u64(data: &[u8], pos: &mut usize) -> u64 {
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos += 8;
    v
}

fn read_str(data: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_u16(data, pos) as usize;
    if *pos + len > data.len() {
        return Err(DataBankError::Codec("string extends past end of data".into()));
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|e| DataBankError::Codec(format!("invalid UTF-8: {e}")))?
        .to_string();
    *pos += len;
    Ok(s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bank_with_entries() -> DataBank {
        let id = BankId::from_raw(0x0001_0002_0003_0004);
        let config = BankConfig {
            vector_width: 4,
            max_entries: 100,
            max_edges_per_entry: 8,
            ..BankConfig::default()
        };
        let mut bank = DataBank::new(id, "test.codec.bank".into(), config);

        // Insert entries
        let v1 = vec![
            Signal::new(1, 100),
            Signal::new(-1, 50),
            Signal::new(0, 0),
            Signal::new(1, 200),
        ];
        let id1 = bank.insert(v1, Temperature::Hot, 10).unwrap();

        let v2 = vec![
            Signal::new(-1, 80),
            Signal::new(1, 160),
            Signal::new(1, 30),
            Signal::new(-1, 90),
        ];
        let id2 = bank.insert(v2, Temperature::Warm, 20).unwrap();

        // Add an edge
        let edge = Edge {
            edge_type: EdgeType::SoundsLike,
            target: BankRef {
                bank: BankId::from_raw(0xDEAD),
                entry: EntryId::from_raw(0xBEEF),
            },
            weight: 180,
            created_tick: 15,
        };
        bank.add_edge(id1, edge).unwrap();

        // Touch an entry
        if let Some(e) = bank.get_mut(id2) {
            e.touch(30);
            e.debug_tag = Some("test_entry".into());
        }

        bank
    }

    #[test]
    fn encode_decode_round_trip() {
        let original = make_bank_with_entries();
        let encoded = encode(&original).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.name, original.name);
        assert_eq!(decoded.len(), original.len());
        assert_eq!(decoded.config().vector_width, original.config().vector_width);
        assert_eq!(decoded.config().max_entries, original.config().max_entries);

        // Verify entries match
        for (id, orig_entry) in original.entries() {
            let dec_entry = decoded.get(*id).expect("entry should exist after decode");
            assert_eq!(dec_entry.vector, orig_entry.vector);
            assert_eq!(dec_entry.temperature, orig_entry.temperature);
            assert_eq!(dec_entry.edges.len(), orig_entry.edges.len());
            assert_eq!(dec_entry.access_count, orig_entry.access_count);
            assert_eq!(dec_entry.debug_tag, orig_entry.debug_tag);
            assert_eq!(dec_entry.checksum, orig_entry.checksum);
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut data = encode(&make_bank_with_entries()).unwrap();
        data[0] = b'X';
        assert!(decode(&data).is_err());
    }

    #[test]
    fn bad_checksum_rejected() {
        let mut data = encode(&make_bank_with_entries()).unwrap();
        // Corrupt a byte in the body
        if data.len() > HEADER_SIZE + 1 {
            data[HEADER_SIZE + 1] ^= 0xFF;
        }
        let result = decode(&data);
        assert!(matches!(result, Err(DataBankError::ChecksumMismatch { .. })));
    }

    #[test]
    fn truncated_data_rejected() {
        let data = encode(&make_bank_with_entries()).unwrap();
        let truncated = &data[..HEADER_SIZE + 2]; // way too short
        assert!(decode(truncated).is_err());
    }

    #[test]
    fn empty_bank_round_trip() {
        let id = BankId::from_raw(42);
        let config = BankConfig {
            vector_width: 32,
            ..BankConfig::default()
        };
        let original = DataBank::new(id, "empty.bank".into(), config);
        let encoded = encode(&original).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.name, "empty.bank");
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn file_round_trip() {
        let bank = make_bank_with_entries();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bank");

        save_atomic(&bank, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.id, bank.id);
        assert_eq!(loaded.name, bank.name);
        assert_eq!(loaded.len(), bank.len());
    }
}
