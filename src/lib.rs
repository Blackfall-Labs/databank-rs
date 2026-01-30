//! Distributed representational memory for neuromorphic systems.
//!
//! Databank-rs provides per-region signal-vector banks with typed edges,
//! temperature lifecycle, sparse pattern completion, and binary `.bank`
//! persistence. Thermograms store the connectome (HOW neurons wire);
//! databanks store the engrams (WHAT neurons represent).

#[cfg(feature = "ternsig")]
pub mod access;
pub mod bank;
pub mod bridge;
pub mod cluster;
pub mod codec;
pub mod entry;
pub mod error;
pub mod fulfiller;
pub mod index;
pub mod ivf;
pub mod journal;
pub mod similarity;
pub mod types;

#[cfg(feature = "ternsig")]
pub use access::ClusterBankAccess;
pub use bank::DataBank;
pub use bridge::{
    entry_id_to_i32_pair, i32_pair_to_entry_id, i32_to_signals, query_results_to_i32,
    signals_to_i32, traverse_results_to_i32,
};
pub use cluster::{BankCluster, ClusterQueryResult};
pub use entry::BankEntry;
pub use error::{DataBankError, Result};
pub use fulfiller::{BankFulfiller, BankSlotMap, FulfillResult};
pub use ivf::{IndexType, IvfIndex};
pub use journal::{JournalEntry, JournalReader, JournalWriter};
pub use similarity::QueryResult;
pub use types::{BankConfig, BankId, BankRef, Edge, EdgeType, EntryId, Temperature};
