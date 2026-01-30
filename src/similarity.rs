use serde::{Deserialize, Serialize};
use ternary_signal::Signal;

use crate::types::EntryId;

/// Result of a similarity query: entry ID + score.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct QueryResult {
    pub entry_id: EntryId,
    /// Similarity score scaled ×256. Range: [-256, 256].
    /// 256 = identical, 0 = orthogonal, -256 = opposite.
    pub score: i32,
}

/// Sparse cosine similarity using only integer arithmetic.
///
/// Only non-zero query signals participate — this IS pattern completion.
/// A partial cue with zeros in unknown dimensions matches against the
/// full stored vector only on the dimensions the cue specifies.
///
/// Returns a score scaled ×256 (i32). Returns 0 for zero-norm inputs.
///
/// Compliant with ASTRO_004: no floating point. Integer-only arithmetic.
pub fn sparse_cosine_similarity(query: &[Signal], stored: &[Signal]) -> i32 {
    let len = query.len().min(stored.len());

    let mut dot: i64 = 0;
    let mut norm_q: i64 = 0;
    let mut norm_s: i64 = 0;

    for i in 0..len {
        let q = query[i];
        // Skip inactive query dimensions (sparse: zeros don't participate)
        if q.polarity == 0 && q.magnitude == 0 {
            continue;
        }

        let q_val = q.polarity as i64 * q.magnitude as i64;
        let s_val = stored[i].polarity as i64 * stored[i].magnitude as i64;

        dot += q_val * s_val;
        norm_q += q_val * q_val;
        norm_s += s_val * s_val;
    }

    if norm_q == 0 || norm_s == 0 {
        return 0;
    }

    // cosine = dot / sqrt(norm_q * norm_s)
    // scaled = dot * 256 / sqrt(norm_q * norm_s)
    let denom = isqrt(norm_q * norm_s);
    if denom == 0 {
        return 0;
    }

    ((dot * 256) / denom) as i32
}

/// Integer square root via Newton's method. 5 iterations is sufficient
/// for the full i64 range. Returns floor(sqrt(n)).
fn isqrt(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }

    // Initial guess: overestimate so Newton converges downward
    let mut x = 1i64 << (((64 - n.leading_zeros()) + 1) / 2);

    for _ in 0..8 {
        let next = (x + n / x) / 2;
        if next >= x {
            break;
        }
        x = next;
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(polarity: i8, magnitude: u8) -> Signal {
        Signal::new(polarity, magnitude)
    }

    fn zero() -> Signal {
        Signal::new(0, 0)
    }

    #[test]
    fn identical_vectors_max_similarity() {
        let a = vec![sig(1, 100), sig(-1, 50), sig(1, 200)];
        let b = a.clone();
        let score = sparse_cosine_similarity(&a, &b);
        // Should be ~256 (scaled max)
        assert!(score >= 250, "expected ~256, got {score}");
    }

    #[test]
    fn opposite_vectors_negative_similarity() {
        let a = vec![sig(1, 100), sig(1, 50), sig(1, 200)];
        let b = vec![sig(-1, 100), sig(-1, 50), sig(-1, 200)];
        let score = sparse_cosine_similarity(&a, &b);
        assert!(score <= -250, "expected ~-256, got {score}");
    }

    #[test]
    fn orthogonal_vectors_zero_similarity() {
        // "Orthogonal" in signal space: one dimension active in each
        let a = vec![sig(1, 100), zero(), zero()];
        let b = vec![zero(), sig(1, 100), zero()];
        let score = sparse_cosine_similarity(&a, &b);
        // Query dim 0 is active, stored dim 0 is zero → dot = 0
        // Query dims 1,2 are zero → skipped
        assert_eq!(score, 0);
    }

    #[test]
    fn sparse_query_pattern_completion() {
        // Full stored pattern: dog concept
        let stored = vec![sig(1, 200), sig(1, 150), sig(-1, 50), sig(1, 100)];
        // Sparse query: only first two dimensions known
        let query = vec![sig(1, 200), sig(1, 150), zero(), zero()];
        let score = sparse_cosine_similarity(&query, &stored);
        // Should be high — the known dimensions match perfectly
        assert!(score >= 250, "expected high similarity, got {score}");
    }

    #[test]
    fn zero_query_returns_zero() {
        let query = vec![zero(), zero(), zero()];
        let stored = vec![sig(1, 100), sig(1, 200), sig(-1, 50)];
        assert_eq!(sparse_cosine_similarity(&query, &stored), 0);
    }

    #[test]
    fn zero_stored_returns_zero() {
        let query = vec![sig(1, 100), sig(1, 200), sig(-1, 50)];
        let stored = vec![zero(), zero(), zero()];
        assert_eq!(sparse_cosine_similarity(&query, &stored), 0);
    }

    #[test]
    fn different_lengths_uses_min() {
        let a = vec![sig(1, 100), sig(1, 100)];
        let b = vec![sig(1, 100), sig(1, 100), sig(1, 100)];
        let score = sparse_cosine_similarity(&a, &b);
        assert!(score >= 250, "expected ~256 on overlapping dims, got {score}");
    }

    #[test]
    fn isqrt_correctness() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3);
        assert_eq!(isqrt(100), 10);
        assert_eq!(isqrt(10000), 100);
        assert_eq!(isqrt(1_000_000), 1000);
    }
}
