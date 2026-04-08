use mlua::prelude::*;

use crate::stats::{mean_impl, table_to_vec};

/// Assign fractional ranks with average tie-breaking.
/// Returns ranks in the same order as input (1-based).
fn rank_impl(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        // Find the end of the tie group
        let mut j = i + 1;
        while j < n && indexed[j].0 == indexed[i].0 {
            j += 1;
        }
        // Average rank for the tie group (1-based)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation: Pearson correlation of ranks.
fn spearman_impl(xs: &[f64], ys: &[f64]) -> Result<f64, &'static str> {
    if xs.len() != ys.len() {
        return Err("arrays must have equal length");
    }
    if xs.len() < 2 {
        return Err("need at least 2 values");
    }
    let rx = rank_impl(xs);
    let ry = rank_impl(ys);

    let mean_rx = mean_impl(&rx);
    let mean_ry = mean_impl(&ry);
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in rx.iter().zip(ry.iter()) {
        let dx = x - mean_rx;
        let dy = y - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 {
        return Err("zero variance in ranks");
    }
    Ok(cov / denom)
}

/// Kendall's tau-b: handles ties. O(n²) pairwise comparison.
fn kendall_tau_impl(xs: &[f64], ys: &[f64]) -> Result<f64, &'static str> {
    if xs.len() != ys.len() {
        return Err("arrays must have equal length");
    }
    let n = xs.len();
    if n < 2 {
        return Err("need at least 2 values");
    }

    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_x: i64 = 0;
    let mut ties_y: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = xs[i].total_cmp(&xs[j]);
            let dy = ys[i].total_cmp(&ys[j]);
            match (dx, dy) {
                (std::cmp::Ordering::Equal, std::cmp::Ordering::Equal) => {
                    ties_x += 1;
                    ties_y += 1;
                }
                (std::cmp::Ordering::Equal, _) => ties_x += 1,
                (_, std::cmp::Ordering::Equal) => ties_y += 1,
                _ if dx == dy => concordant += 1,
                _ => discordant += 1,
            }
        }
    }

    let n0 = n as f64 * (n - 1) as f64 / 2.0;
    let denom = ((n0 - ties_x as f64) * (n0 - ties_y as f64)).sqrt();
    if denom == 0.0 {
        return Err("zero variance (all values tied)");
    }
    Ok((concordant - discordant) as f64 / denom)
}

/// NDCG@k (Normalized Discounted Cumulative Gain), linear gain variant.
/// Uses DCG = Σ rel_i / log₂(i+2) (0-indexed), not the exponential (2^rel-1) variant.
/// `relevance` contains relevance scores in the order of the ranking.
fn ndcg_impl(relevance: &[f64], k: usize) -> f64 {
    if k == 0 || relevance.is_empty() {
        return 0.0;
    }
    let k = k.min(relevance.len());

    let dcg: f64 = relevance[..k]
        .iter()
        .enumerate()
        .map(|(i, &r)| r / (2.0 + i as f64).log2())
        .sum();

    // Ideal: sort descending
    let mut ideal = relevance.to_vec();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal[..k]
        .iter()
        .enumerate()
        .map(|(i, &r)| r / (2.0 + i as f64).log2())
        .sum();

    if idcg == 0.0 {
        return 0.0;
    }
    dcg / idcg
}

/// MRR (Mean Reciprocal Rank).
/// `rankings` is a list of rank positions (1-based) where the relevant item was found.
/// Returns the mean of 1/rank across all queries.
fn mrr_impl(rankings: &[f64]) -> Result<f64, &'static str> {
    if rankings.is_empty() {
        return Err("need at least 1 ranking");
    }
    for &r in rankings {
        if r < 1.0 {
            return Err("rank values must be >= 1");
        }
        if r.fract() != 0.0 {
            return Err("rank values must be integers");
        }
    }
    let sum: f64 = rankings.iter().map(|&r| 1.0 / r).sum();
    Ok(sum / rankings.len() as f64)
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "rank",
        lua.create_function(|lua, table: LuaTable| {
            let v = table_to_vec(&table)?;
            let ranks = rank_impl(&v);
            let out = lua.create_table()?;
            for (i, r) in ranks.into_iter().enumerate() {
                out.raw_set(i + 1, r)?;
            }
            Ok(out)
        })?,
    )?;

    t.set(
        "spearman_correlation",
        lua.create_function(|_, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            spearman_impl(&xs, &ys)
                .map_err(|e| LuaError::runtime(format!("spearman_correlation: {e}")))
        })?,
    )?;

    t.set(
        "kendall_tau",
        lua.create_function(|_, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            kendall_tau_impl(&xs, &ys).map_err(|e| LuaError::runtime(format!("kendall_tau: {e}")))
        })?,
    )?;

    t.set(
        "ndcg",
        lua.create_function(|_, (table, k): (LuaTable, usize)| {
            let v = table_to_vec(&table)?;
            Ok(ndcg_impl(&v, k))
        })?,
    )?;

    t.set(
        "mrr",
        lua.create_function(|_, table: LuaTable| {
            let v = table_to_vec(&table)?;
            mrr_impl(&v).map_err(|e| LuaError::runtime(format!("mrr: {e}")))
        })?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_no_ties() {
        let ranks = rank_impl(&[3.0, 1.0, 2.0]);
        assert_eq!(ranks, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn rank_with_ties() {
        let ranks = rank_impl(&[3.0, 1.0, 3.0]);
        // positions sorted: 1.0(idx=1)→rank1, 3.0(idx=0)→rank2.5, 3.0(idx=2)→rank2.5
        assert_eq!(ranks, vec![2.5, 1.0, 2.5]);
    }

    #[test]
    fn rank_all_equal() {
        let ranks = rank_impl(&[5.0, 5.0, 5.0]);
        assert_eq!(ranks, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn spearman_perfect() {
        let r = spearman_impl(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0]).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spearman_inverse() {
        let r = spearman_impl(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        assert!((r - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn kendall_tau_concordant() {
        let tau = kendall_tau_impl(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((tau - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_tau_discordant() {
        let tau = kendall_tau_impl(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]).unwrap();
        assert!((tau - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn ndcg_perfect() {
        // Already in ideal order
        let score = ndcg_impl(&[3.0, 2.0, 1.0], 3);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_imperfect() {
        // Reversed: worst order
        let score = ndcg_impl(&[1.0, 2.0, 3.0], 3);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn mrr_basic() {
        // Found at rank 1, 2, 5 → (1 + 0.5 + 0.2) / 3
        let r = mrr_impl(&[1.0, 2.0, 5.0]).unwrap();
        let expected = (1.0 + 0.5 + 0.2) / 3.0;
        assert!((r - expected).abs() < 1e-10);
    }
}
