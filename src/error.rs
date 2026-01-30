use crate::types::{BankId, EntryId};

/// All errors that can occur in databank operations.
#[derive(Debug, thiserror::Error)]
pub enum DataBankError {
    /// Entry vector length does not match the bank's fixed vector width.
    #[error("vector width mismatch: bank expects {expected}, got {got}")]
    VectorWidthMismatch { expected: u16, got: u16 },

    /// Bank has reached its maximum entry capacity.
    #[error("bank is full (capacity: {capacity})")]
    BankFull { capacity: u32 },

    /// Requested entry does not exist in the bank.
    #[error("entry not found: {id:?}")]
    EntryNotFound { id: EntryId },

    /// Entry has reached its maximum edge count.
    #[error("edge limit reached (max: {max})")]
    EdgeLimitReached { max: u16 },

    /// Requested bank does not exist in the cluster.
    #[error("bank not found: {id:?}")]
    BankNotFound { id: BankId },

    /// File I/O error during persistence.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Binary format error (bad magic, truncated, invalid structure).
    #[error("codec error: {0}")]
    Codec(String),

    /// Checksum verification failed after decode.
    #[error("checksum mismatch: expected {expected:#018x}, got {actual:#018x}")]
    ChecksumMismatch { expected: u64, actual: u64 },
}

/// Convenience alias for databank results.
pub type Result<T> = std::result::Result<T, DataBankError>;
