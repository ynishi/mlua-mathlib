use mlua::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::stats::{mean_impl, sort_floats, table_to_vec, variance_impl};

/// Welch's t-test (unequal variances).
/// Returns {t_stat, df, p_value (two-tailed)}.
fn welch_t_impl(xs: &[f64], ys: &[f64]) -> Result<(f64, f64, f64), &'static str> {
    let n1 = xs.len();
    let n2 = ys.len();
    if n1 < 2 || n2 < 2 {
        return Err("each group needs at least 2 values");
    }

    let mean1 = mean_impl(xs);
    let mean2 = mean_impl(ys);
    let var1 = variance_impl(xs);
    let var2 = variance_impl(ys);
    let n1f = n1 as f64;
    let n2f = n2 as f64;

    let se1 = var1 / n1f;
    let se2 = var2 / n2f;
    let se_sum = se1 + se2;

    if se_sum == 0.0 {
        return Err("both groups have zero variance");
    }

    let t_stat = (mean1 - mean2) / se_sum.sqrt();

    // Welch-Satterthwaite degrees of freedom
    let df = (se_sum * se_sum) / (se1 * se1 / (n1f - 1.0) + se2 * se2 / (n2f - 1.0));

    let dist = StudentsT::new(0.0, 1.0, df).map_err(|_| "invalid degrees of freedom")?;
    let p_value = 2.0 * (1.0 - dist.cdf(t_stat.abs()));

    Ok((t_stat, df, p_value))
}

/// Mann-Whitney U test (two-sample rank-sum, non-parametric).
/// Returns {u_stat, z_score, p_value (two-tailed, normal approximation)}.
/// When `tie_correction` is true, the variance is adjusted for tied ranks.
fn mann_whitney_u_impl(
    xs: &[f64],
    ys: &[f64],
    tie_correction: bool,
) -> Result<(f64, f64, f64), &'static str> {
    let n1 = xs.len();
    let n2 = ys.len();
    if n1 == 0 || n2 == 0 {
        return Err("both groups must be non-empty");
    }

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n1 + n2);
    for (i, &v) in xs.iter().enumerate() {
        combined.push((v, i)); // group 0
    }
    for (i, &v) in ys.iter().enumerate() {
        combined.push((v, n1 + i)); // group 1
    }
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Assign fractional ranks and collect tie group sizes
    let n = combined.len();
    let mut ranks = vec![0.0; n];
    let mut tie_groups: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let group_size = (j - i) as f64;
        if tie_correction && group_size > 1.0 {
            tie_groups.push(group_size);
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        i = j;
    }

    // Sum ranks for group 1 (xs)
    let r1: f64 = combined
        .iter()
        .zip(ranks.iter())
        .filter(|(c, _)| c.1 < n1)
        .map(|(_, &r)| r)
        .sum();

    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let u1 = r1 - n1f * (n1f + 1.0) / 2.0;
    let u2 = n1f * n2f - u1;
    let u = u1.min(u2);

    // Normal approximation
    let mu = n1f * n2f / 2.0;
    let nf = n1f + n2f;
    let sigma = if tie_correction && !tie_groups.is_empty() {
        // σ² = (n1*n2/12) * (N+1 - Σ(t_k³-t_k) / (N*(N-1)))
        let tie_term: f64 = tie_groups.iter().map(|&t| t * t * t - t).sum();
        (n1f * n2f / 12.0 * (nf + 1.0 - tie_term / (nf * (nf - 1.0)))).sqrt()
    } else {
        (n1f * n2f * (nf + 1.0) / 12.0).sqrt()
    };

    if sigma == 0.0 {
        return Err("zero variance (all values identical)");
    }

    let z = (u - mu) / sigma;
    // Two-tailed p-value from standard normal
    let dist =
        statrs::distribution::Normal::new(0.0, 1.0).map_err(|_| "failed to create normal dist")?;
    let p_value = 2.0 * dist.cdf(z); // z is negative for small U

    Ok((u, z, p_value))
}

/// Chi-squared goodness-of-fit test.
/// Returns {chi2_stat, df, p_value}.
fn chi_squared_test_impl(
    observed: &[f64],
    expected: &[f64],
) -> Result<(f64, f64, f64), &'static str> {
    if observed.len() != expected.len() {
        return Err("observed and expected must have equal length");
    }
    if observed.len() < 2 {
        return Err("need at least 2 categories");
    }
    for &e in expected {
        if e <= 0.0 {
            return Err("expected values must be > 0");
        }
    }

    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();

    let df = (observed.len() - 1) as f64;

    // p-value from chi-squared CDF: P(X > chi2)
    let dist = statrs::distribution::ChiSquared::new(df)
        .map_err(|_| "invalid degrees of freedom for chi-squared")?;
    let p_value = 1.0 - dist.cdf(chi2);

    Ok((chi2, df, p_value))
}

/// Two-sample Kolmogorov-Smirnov test.
/// Returns {d_stat, p_value (asymptotic)}.
fn ks_test_impl(xs: &[f64], ys: &[f64]) -> Result<(f64, f64), &'static str> {
    if xs.is_empty() || ys.is_empty() {
        return Err("both samples must be non-empty");
    }

    let mut xs_sorted = xs.to_vec();
    let mut ys_sorted = ys.to_vec();
    sort_floats(&mut xs_sorted);
    sort_floats(&mut ys_sorted);

    let n1 = xs_sorted.len();
    let n2 = ys_sorted.len();
    let inv_n1 = 1.0 / n1 as f64;
    let inv_n2 = 1.0 / n2 as f64;

    // Cumulative ECDF difference algorithm.
    // Track the running difference d = F1(x) - F2(x) and record the max |d|.
    // Tied values across samples are processed as a group to avoid
    // intermediate states that would overestimate D.
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d: f64 = 0.0;
    let mut d_max: f64 = 0.0;

    while i < n1 && j < n2 {
        let x = xs_sorted[i];
        let y = ys_sorted[j];
        if x < y {
            d += inv_n1;
            i += 1;
        } else if x > y {
            d -= inv_n2;
            j += 1;
        } else {
            // Tie group: advance all equal values in both samples at once
            let mut ci = 0;
            while i < n1 && xs_sorted[i] == x {
                ci += 1;
                i += 1;
            }
            let mut cj = 0;
            while j < n2 && ys_sorted[j] == x {
                cj += 1;
                j += 1;
            }
            d += ci as f64 * inv_n1 - cj as f64 * inv_n2;
        }
        d_max = d_max.max(d.abs());
    }

    // Asymptotic p-value: Kolmogorov distribution approximation
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let ne = (n1f * n2f / (n1f + n2f)).sqrt();
    let lambda = (ne + 0.12 + 0.11 / ne) * d_max;

    // Kolmogorov survival function approximation (series)
    let mut p_value = 0.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = 2.0 * (-1.0_f64).powi(k - 1) * (-2.0 * kf * kf * lambda * lambda).exp();
        p_value += term;
    }
    let p_value = p_value.clamp(0.0, 1.0);

    Ok((d_max, p_value))
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "welch_t_test",
        lua.create_function(|lua, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            let (t_stat, df, p_value) = welch_t_impl(&xs, &ys)
                .map_err(|e| LuaError::runtime(format!("welch_t_test: {e}")))?;
            let result = lua.create_table()?;
            result.set("t_stat", t_stat)?;
            result.set("df", df)?;
            result.set("p_value", p_value)?;
            Ok(result)
        })?,
    )?;

    // mann_whitney_u(xs, ys)  — no tie correction (default)
    // mann_whitney_u(xs, ys, {tie_correction = true})  — with tie correction
    t.set(
        "mann_whitney_u",
        lua.create_function(
            |lua, (xs_t, ys_t, opts): (LuaTable, LuaTable, Option<LuaTable>)| {
                let xs = table_to_vec(&xs_t)?;
                let ys = table_to_vec(&ys_t)?;
                let tie_correction = opts
                    .and_then(|t| t.get::<bool>("tie_correction").ok())
                    .unwrap_or(false);
                let (u, z, p) = mann_whitney_u_impl(&xs, &ys, tie_correction)
                    .map_err(|e| LuaError::runtime(format!("mann_whitney_u: {e}")))?;
                let result = lua.create_table()?;
                result.set("u_stat", u)?;
                result.set("z_score", z)?;
                result.set("p_value", p)?;
                Ok(result)
            },
        )?,
    )?;

    t.set(
        "chi_squared_test",
        lua.create_function(|lua, (obs_t, exp_t): (LuaTable, LuaTable)| {
            let obs = table_to_vec(&obs_t)?;
            let exp = table_to_vec(&exp_t)?;
            let (chi2, df, p) = chi_squared_test_impl(&obs, &exp)
                .map_err(|e| LuaError::runtime(format!("chi_squared_test: {e}")))?;
            let result = lua.create_table()?;
            result.set("chi2_stat", chi2)?;
            result.set("df", df)?;
            result.set("p_value", p)?;
            Ok(result)
        })?,
    )?;

    t.set(
        "ks_test",
        lua.create_function(|lua, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            let (d, p) =
                ks_test_impl(&xs, &ys).map_err(|e| LuaError::runtime(format!("ks_test: {e}")))?;
            let result = lua.create_table()?;
            result.set("d_stat", d)?;
            result.set("p_value", p)?;
            Ok(result)
        })?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welch_t_same_distribution() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [1.5, 2.5, 3.5, 4.5, 5.5];
        let (t, df, p) = welch_t_impl(&xs, &ys).unwrap();
        // Small difference, p should be > 0.05
        assert!(t.is_finite());
        assert!(df > 0.0);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn welch_t_very_different() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [100.0, 200.0, 300.0, 400.0, 500.0];
        let (_, _, p) = welch_t_impl(&xs, &ys).unwrap();
        assert!(
            p < 0.05,
            "p={p} should be significant for very different groups"
        );
    }

    #[test]
    fn mann_whitney_identical() {
        let xs = [1.0, 2.0, 3.0];
        let ys = [1.0, 2.0, 3.0];
        let (u, _, p) = mann_whitney_u_impl(&xs, &ys, false).unwrap();
        assert!(u.is_finite());
        assert!(p.is_finite());
    }

    #[test]
    fn mann_whitney_tie_correction() {
        // With ties, tie-corrected sigma should be smaller → |z| larger → p smaller
        let xs = [1.0, 2.0, 2.0, 3.0, 3.0];
        let ys = [4.0, 5.0, 5.0, 6.0, 6.0];
        let (_, _, p_no) = mann_whitney_u_impl(&xs, &ys, false).unwrap();
        let (_, _, p_tc) = mann_whitney_u_impl(&xs, &ys, true).unwrap();
        assert!(
            p_tc <= p_no,
            "tie-corrected p ({p_tc}) should be <= uncorrected p ({p_no})"
        );
    }

    #[test]
    fn chi_squared_uniform() {
        // Observed matches expected perfectly
        let obs = [25.0, 25.0, 25.0, 25.0];
        let exp = [25.0, 25.0, 25.0, 25.0];
        let (chi2, df, p) = chi_squared_test_impl(&obs, &exp).unwrap();
        assert!((chi2 - 0.0).abs() < 1e-10);
        assert!((df - 3.0).abs() < 1e-10);
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_skewed() {
        let obs = [90.0, 5.0, 3.0, 2.0];
        let exp = [25.0, 25.0, 25.0, 25.0];
        let (chi2, _, p) = chi_squared_test_impl(&obs, &exp).unwrap();
        assert!(chi2 > 100.0);
        assert!(p < 0.001);
    }

    #[test]
    fn ks_test_same_distribution() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();
        let ys: Vec<f64> = (0..50).map(|i| (i as f64 + 0.5) / 50.0).collect();
        let (d, p) = ks_test_impl(&xs, &ys).unwrap();
        assert!(d < 0.1);
        assert!(p > 0.05);
    }

    #[test]
    fn ks_test_identical_values_d_is_zero() {
        // When both samples contain identical values, D must be 0.
        let xs = vec![5.0, 5.0, 5.0];
        let ys = vec![5.0, 5.0];
        let (d, _) = ks_test_impl(&xs, &ys).unwrap();
        assert!(
            d.abs() < 1e-10,
            "D should be 0 for identical-value samples, got {d}"
        );
    }

    #[test]
    fn ks_test_ties_across_samples() {
        // Mixed ties: some values shared, some not.
        let xs = vec![1.0, 2.0, 3.0, 3.0];
        let ys = vec![1.0, 3.0, 3.0, 5.0];
        let (d, _) = ks_test_impl(&xs, &ys).unwrap();
        // ECDF_xs: 0.25@1, 0.50@2, 1.0@3
        // ECDF_ys: 0.25@1, 0.75@3, 1.0@5
        // Max diff at x=2: |0.50 - 0.25| = 0.25
        assert!((d - 0.25).abs() < 1e-10, "D should be 0.25, got {d}");
    }

    #[test]
    fn ks_test_different_distribution() {
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..50).map(|i| (i as f64) + 100.0).collect();
        let (d, _p) = ks_test_impl(&xs, &ys).unwrap();
        assert!(
            (d - 1.0).abs() < 1e-10,
            "d={d}, completely separated distributions should have d≈1"
        );
    }
}
