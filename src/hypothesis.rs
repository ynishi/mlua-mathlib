use mlua::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::stats::{mean_impl, sort_floats, table_to_vec, variance_impl};

/// Ceiling on permutation draws, mirroring the resampling cap: `draws` and
/// `seed` are adjacent bare numbers and Lua has no named arguments.
const MAX_PERMUTATION_DRAWS: usize = 10_000_000;

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

/// Which tail a permutation test counts against.
#[derive(Clone, Copy, PartialEq)]
enum Alternative {
    /// `|permuted| >= |observed|` — a difference in either direction.
    TwoSided,
    /// `permuted >= observed` — xs greater than ys.
    Greater,
    /// `permuted <= observed` — xs less than ys.
    Less,
}

impl Alternative {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "two_sided" => Ok(Self::TwoSided),
            "greater" => Ok(Self::Greater),
            "less" => Ok(Self::Less),
            other => Err(format!(
                "alternative must be \"two_sided\", \"greater\" or \"less\", got \"{other}\""
            )),
        }
    }

    /// Whether a reshuffled statistic counts as at least as extreme.
    ///
    /// `gamma` admits the ulps that separate two sums of the same multiset in
    /// different orders — without it, permutations that genuinely tie with the
    /// observed value fall out of the count and the p-value comes back too
    /// small, which is the wrong direction for a significance test.
    fn counts(self, permuted: f64, observed: f64, gamma: f64) -> bool {
        match self {
            Self::TwoSided => permuted.abs() >= observed.abs() - gamma,
            Self::Greater => permuted >= observed - gamma,
            Self::Less => permuted <= observed + gamma,
        }
    }
}

/// Permutation test on the difference in means.
///
/// Shuffles the pooled observations, splits them back at the original group
/// sizes, and counts how often the reshuffled difference is at least as
/// extreme as the observed one. Where the other tests here assume a shape —
/// normality for Welch's t, continuity for Mann-Whitney and Kolmogorov-Smirnov
/// — this assumes only that the labels are exchangeable under the null.
///
/// # The p-value is `(1 + extreme) / (1 + draws)`
///
/// Not `extreme / draws`. The observed arrangement is itself one of the
/// permutations under the null, so counting it keeps the test valid; without
/// it the estimate can reach exactly zero, which claims more than any finite
/// number of draws can support. The floor is therefore `1 / (1 + draws)`
/// [Phipson & Smyth 2010].
///
/// # Ties are counted with a tolerance
///
/// `observed` is summed once in the caller's order while each `permuted` is
/// summed in a shuffled one, and a sum of the same multiset differs by ulps
/// between the two. An exact comparison therefore drops permutations that
/// genuinely tie, understating the p-value — the anti-conservative direction.
/// The tolerance scales with the length and magnitude the sums are built from,
/// which is where that error comes from.
fn permutation_test_impl(
    xs: &[f64],
    ys: &[f64],
    draws: usize,
    seed: u64,
    alternative: Alternative,
) -> Result<(f64, f64, usize), String> {
    if xs.is_empty() || ys.is_empty() {
        return Err("both groups must be non-empty".into());
    }
    if draws == 0 {
        return Err("needs at least one draw".into());
    }
    if draws > MAX_PERMUTATION_DRAWS {
        return Err(format!(
            "draws is capped at {MAX_PERMUTATION_DRAWS}, got {draws}; check the argument order — \
             it is (xs, ys, draws, seed)"
        ));
    }

    let n1 = xs.len();
    let observed = mean_impl(xs) - mean_impl(ys);

    let mut pool: Vec<f64> = Vec::with_capacity(n1 + ys.len());
    pool.extend_from_slice(xs);
    pool.extend_from_slice(ys);

    let n = pool.len();
    let n2 = n - n1;
    let total: f64 = pool.iter().sum();
    let scale: f64 = pool.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
    let gamma = 4.0 * n as f64 * f64::EPSILON * (observed.abs() + scale);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut extreme = 0usize;
    for _ in 0..draws {
        // Only the first n1 slots decide the split, so a partial shuffle is
        // enough; and the second group's sum is the pooled total minus the
        // first's, so it never has to be walked.
        pool.partial_shuffle(&mut rng, n1);
        let first: f64 = pool[..n1].iter().sum();
        let permuted = first / n1 as f64 - (total - first) / n2 as f64;
        if alternative.counts(permuted, observed, gamma) {
            extreme += 1;
        }
    }

    let p_value = (1 + extreme) as f64 / (1 + draws) as f64;
    Ok((observed, p_value, extreme))
}

/// Reject a p-value vector that is empty or holds a value outside [0, 1].
fn validate_p_values(p: &[f64]) -> Result<(), String> {
    if p.is_empty() {
        return Err("expected at least one p-value".into());
    }
    for (i, &v) in p.iter().enumerate() {
        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
            return Err(format!("p[{}] is not a p-value: {v}", i + 1));
        }
    }
    Ok(())
}

/// Holm-Bonferroni step-down adjustment.
///
/// Controls the family-wise error rate — the chance of *any* false rejection
/// among the family — and is uniformly more powerful than plain Bonferroni,
/// with the same assumptions. Compare each returned value against the
/// uncorrected level.
///
/// The running maximum enforces monotonicity: an adjusted value never falls
/// below one for a smaller raw p-value, which is what makes the step-down
/// procedure coherent to read.
fn holm_impl(p: &[f64]) -> Result<Vec<f64>, String> {
    validate_p_values(p)?;
    let n = p.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));

    let mut adjusted = vec![0.0; n];
    let mut running: f64 = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        running = running.max(((n - rank) as f64 * p[i]).min(1.0));
        adjusted[i] = running;
    }
    Ok(adjusted)
}

/// Benjamini-Hochberg step-up adjustment (false discovery rate).
///
/// Controls the expected *share* of false rejections among the rejections
/// made, which is the weaker guarantee Holm's family-wise control gives up
/// power for. Prefer it when the family is large and a few false positives
/// are tolerable.
///
/// The running minimum walks from the largest p-value down, enforcing the
/// monotonicity the step-up procedure requires.
fn benjamini_hochberg_impl(p: &[f64]) -> Result<Vec<f64>, String> {
    validate_p_values(p)?;
    let n = p.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));

    let mut adjusted = vec![0.0; n];
    let mut running: f64 = 1.0;
    for (step, &i) in order.iter().enumerate() {
        let rank = n - step; // 1-based rank in ascending order
        running = running.min((n as f64 / rank as f64 * p[i]).min(1.0));
        adjusted[i] = running;
    }
    Ok(adjusted)
}

/// Cohen's d — the difference in means in pooled standard deviations.
///
/// Answers "by how much", which a p-value does not: a vanishing difference
/// reaches any significance level given enough observations. Conventional
/// reading is 0.2 small / 0.5 medium / 0.8 large, though those thresholds are
/// rules of thumb rather than properties of the statistic.
///
/// Assumes roughly comparable spread in the two groups; where that fails,
/// [`cliffs_delta_impl`] makes no such assumption. Note also that `welch_t_test`
/// in this module deliberately does *not* assume equal variance, so the two do
/// not rest on the same footing.
///
/// Biased upward on small samples — Hedges' correction factor
/// `1 - 3/(4(nx+ny)-9)` is about 0.7 at the smallest accepted input of two and
/// two, and approaches 1 as the groups grow. Read `d` with that in mind at
/// small `n`.
fn cohens_d_impl(xs: &[f64], ys: &[f64]) -> Result<f64, &'static str> {
    if xs.len() < 2 || ys.len() < 2 {
        return Err("each group needs at least 2 values");
    }
    let (nx, ny) = (xs.len() as f64, ys.len() as f64);
    let pooled_var =
        ((nx - 1.0) * variance_impl(xs) + (ny - 1.0) * variance_impl(ys)) / (nx + ny - 2.0);
    if pooled_var <= 0.0 {
        return Err("pooled variance is zero; both groups are constant");
    }
    Ok((mean_impl(xs) - mean_impl(ys)) / pooled_var.sqrt())
}

/// Cliff's delta — `P(x > y) - P(x < y)`, in `[-1, 1]`.
///
/// Ordinal and distribution-free: it reads only the direction of each
/// pairwise comparison, so a heavy tail or a non-normal shape does not
/// distort it the way it distorts [`cohens_d_impl`]. The non-parametric
/// counterpart of the Mann-Whitney U test, which shares its pairwise
/// comparison count.
///
/// Cost is `O(len(xs) * len(ys))` — the pairs are counted directly.
fn cliffs_delta_impl(xs: &[f64], ys: &[f64]) -> Result<f64, &'static str> {
    if xs.is_empty() || ys.is_empty() {
        return Err("both groups must be non-empty");
    }
    let mut greater = 0i64;
    let mut less = 0i64;
    for &x in xs {
        for &y in ys {
            if x > y {
                greater += 1;
            } else if x < y {
                less += 1;
            }
        }
    }
    Ok((greater - less) as f64 / (xs.len() * ys.len()) as f64)
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

    // permutation_test(xs, ys, draws, seed)
    // permutation_test(xs, ys, draws, seed, {alternative = "greater"})
    t.set(
        "permutation_test",
        lua.create_function(
            |lua,
             (xs_t, ys_t, draws, seed, opts): (
                LuaTable,
                LuaTable,
                usize,
                u64,
                Option<LuaTable>,
            )| {
                let xs = table_to_vec(&xs_t)?;
                let ys = table_to_vec(&ys_t)?;
                // `Option<String>` rather than `.ok()`: a missing key is the
                // default, but a key holding the wrong type is an error. With
                // `.ok()` a typo'd key ("alternatve") fell through to
                // two_sided and the caller got a two-sided p-value with no
                // sign that the option had been ignored.
                let alternative = match opts {
                    Some(o) => {
                        // Reject an unrecognised key outright. A missing
                        // "alternative" is indistinguishable from a typo'd one
                        // otherwise, and the caller would receive a two-sided
                        // p-value having asked for a one-sided test.
                        for pair in o.pairs::<LuaValue, LuaValue>() {
                            let (key, _) = pair?;
                            let name = key.to_string()?;
                            if name != "alternative" {
                                return Err(LuaError::runtime(format!(
                                    "permutation_test: unknown option \"{name}\"; the only option \
                                     is \"alternative\""
                                )));
                            }
                        }
                        match o.get::<Option<String>>("alternative").map_err(|e| {
                            LuaError::runtime(format!("permutation_test: alternative: {e}"))
                        })? {
                            Some(s) => Alternative::parse(&s)
                                .map_err(|e| LuaError::runtime(format!("permutation_test: {e}")))?,
                            None => Alternative::TwoSided,
                        }
                    }
                    None => Alternative::TwoSided,
                };
                let (observed, p_value, extreme) =
                    permutation_test_impl(&xs, &ys, draws, seed, alternative)
                        .map_err(|e| LuaError::runtime(format!("permutation_test: {e}")))?;
                let result = lua.create_table()?;
                result.set("observed", observed)?;
                result.set("p_value", p_value)?;
                result.set("extreme_draws", extreme)?;
                result.set("draws", draws)?;
                Ok(result)
            },
        )?,
    )?;

    t.set(
        "holm",
        lua.create_function(|_, p_t: LuaTable| {
            let p = table_to_vec(&p_t)?;
            holm_impl(&p).map_err(|e| LuaError::runtime(format!("holm: {e}")))
        })?,
    )?;

    t.set(
        "benjamini_hochberg",
        lua.create_function(|_, p_t: LuaTable| {
            let p = table_to_vec(&p_t)?;
            benjamini_hochberg_impl(&p)
                .map_err(|e| LuaError::runtime(format!("benjamini_hochberg: {e}")))
        })?,
    )?;

    t.set(
        "cohens_d",
        lua.create_function(|_, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            cohens_d_impl(&xs, &ys).map_err(|e| LuaError::runtime(format!("cohens_d: {e}")))
        })?,
    )?;

    t.set(
        "cliffs_delta",
        lua.create_function(|_, (xs_t, ys_t): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_t)?;
            let ys = table_to_vec(&ys_t)?;
            cliffs_delta_impl(&xs, &ys).map_err(|e| LuaError::runtime(format!("cliffs_delta: {e}")))
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

    // ── permutation test ────────────────────────────────────

    #[test]
    fn permutation_finds_a_separated_pair_significant() {
        let low = [1.0, 2.0, 3.0, 4.0, 5.0];
        let high = [11.0, 12.0, 13.0, 14.0, 15.0];
        let (obs, p, _) =
            permutation_test_impl(&high, &low, 2000, 1, Alternative::TwoSided).unwrap();
        assert!((obs - 10.0).abs() < 1e-12);
        // No reshuffle of these ten values separates them as cleanly, so only
        // the floor remains.
        assert!(p < 0.01, "got p={p}");
    }

    #[test]
    fn permutation_finds_interleaved_groups_unremarkable() {
        let a = [1.0, 3.0, 5.0, 7.0, 9.0];
        let b = [2.0, 4.0, 6.0, 8.0, 10.0];
        let (_, p, _) = permutation_test_impl(&a, &b, 2000, 2, Alternative::TwoSided).unwrap();
        assert!(p > 0.2, "got p={p}");
    }

    #[test]
    fn the_p_value_never_reaches_zero() {
        // (1 + extreme) / (1 + draws), so the floor is 1/(1+draws). Without the
        // correction a perfectly separated pair would report exactly 0.
        let low = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let high = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let (_, p, extreme) =
            permutation_test_impl(&high, &low, 999, 3, Alternative::TwoSided).unwrap();
        let floor = 1.0 / 1000.0;
        assert!(p >= floor, "p can never fall below 1/(1+draws): {p}");
        assert!(
            (p - (1 + extreme) as f64 / 1000.0).abs() < 1e-12,
            "p is exactly (1+extreme)/(1+draws): p={p}, extreme={extreme}"
        );
        // Only the two fully separated arrangements out of C(12,6)=924 reach
        // the observed difference, so p stays near the floor without touching
        // zero.
        assert!(p < 0.02, "got p={p}, extreme={extreme}");
    }

    #[test]
    fn a_one_sided_alternative_reads_the_named_direction() {
        let low = [1.0, 2.0, 3.0, 4.0, 5.0];
        let high = [6.0, 7.0, 8.0, 9.0, 10.0];
        // high - low is positive, so "greater" is the supported direction and
        // "less" is not.
        let (_, p_greater, _) =
            permutation_test_impl(&high, &low, 2000, 4, Alternative::Greater).unwrap();
        let (_, p_less, _) =
            permutation_test_impl(&high, &low, 2000, 4, Alternative::Less).unwrap();
        assert!(p_greater < 0.05, "got {p_greater}");
        assert!(p_less > 0.95, "got {p_less}");
    }

    #[test]
    fn permutation_counts_ties_that_only_floating_point_separates() {
        // Two identical multisets: every arrangement ties with the observed
        // difference, so the exact p-value is 1. But `observed` is summed in
        // the caller's order and each permutation in a shuffled one, and
        // 0.1 + 0.2 + 0.3 is not associative in binary — an exact `>=` drops
        // 20 of the 90 distinct arrangements and reports p ≈ 0.80.
        let xs = [0.1, 0.2, 0.3];
        let ys = [0.2, 0.3, 0.1];
        let (observed, p, extreme) =
            permutation_test_impl(&xs, &ys, 2000, 1, Alternative::TwoSided).unwrap();
        assert!(observed != 0.0, "the residue is real: {observed:e}");
        assert_eq!(extreme, 2000, "every draw ties under the null");
        assert!((p - 1.0).abs() < 1e-12, "got p={p}");
    }

    #[test]
    fn permutation_handles_unequal_group_sizes() {
        // The split is pool[..n1] / pool[n1..], so unequal sizes are where a
        // partition bug would show. 2 vs 7.
        let small = [10.0, 11.0];
        let large = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let (obs, p, _) =
            permutation_test_impl(&small, &large, 2000, 5, Alternative::TwoSided).unwrap();
        assert!((obs - (10.5 - 4.0)).abs() < 1e-12, "got {obs}");
        assert!(
            p < 0.1,
            "the small group sits above all of the large: p={p}"
        );

        // Reversing the roles flips the sign of the observed difference and
        // leaves the two-sided p-value where it was.
        let (obs_rev, p_rev, _) =
            permutation_test_impl(&large, &small, 2000, 5, Alternative::TwoSided).unwrap();
        assert!((obs_rev + obs).abs() < 1e-12, "{obs_rev} vs {obs}");
        assert!((p_rev - p).abs() < 0.05, "{p_rev} vs {p}");
    }

    #[test]
    fn permutation_is_reproducible_and_refuses_bad_input() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let first = permutation_test_impl(&a, &b, 500, 7, Alternative::TwoSided).unwrap();
        let again = permutation_test_impl(&a, &b, 500, 7, Alternative::TwoSided).unwrap();
        assert_eq!(first.1.to_bits(), again.1.to_bits());

        assert!(permutation_test_impl(&[], &b, 100, 1, Alternative::TwoSided).is_err());
        assert!(permutation_test_impl(&a, &b, 0, 1, Alternative::TwoSided).is_err());
        let err =
            permutation_test_impl(&a, &b, MAX_PERMUTATION_DRAWS + 1, 1, Alternative::TwoSided)
                .unwrap_err();
        assert!(err.contains("argument order"), "got: {err}");
        assert!(Alternative::parse("sideways").is_err());
    }

    // ── multiple-comparison adjustment ──────────────────────

    #[test]
    fn holm_matches_the_worked_example() {
        // Holm on [0.01, 0.04, 0.03] with n=3: sorted 0.01, 0.03, 0.04 gets
        // multipliers 3, 2, 1 -> 0.03, 0.06, 0.04, then the running max makes
        // the last one 0.06.
        let adj = holm_impl(&[0.01, 0.04, 0.03]).unwrap();
        assert!((adj[0] - 0.03).abs() < 1e-12, "got {adj:?}");
        assert!((adj[2] - 0.06).abs() < 1e-12, "got {adj:?}");
        assert!((adj[1] - 0.06).abs() < 1e-12, "got {adj:?}");
    }

    #[test]
    fn holm_is_monotone_and_never_below_the_raw_value() {
        // Deliberately unsorted: a sorted fixture would let a reversed rank
        // multiplier (1..n instead of n..1) pass this test, since the result
        // stays monotone and above the raw values either way.
        let raw = [0.2, 0.001, 0.9, 0.02, 0.008];
        let adj = holm_impl(&raw).unwrap();
        for (r, a) in raw.iter().zip(adj.iter()) {
            assert!(a >= r, "adjustment must not lower a p-value: {a} < {r}");
            assert!((0.0..=1.0).contains(a));
        }
        // Order is preserved: rank the pairs by the raw value and the
        // adjustments must be non-decreasing along that ranking.
        let mut pairs: Vec<(f64, f64)> = raw.iter().copied().zip(adj.iter().copied()).collect();
        pairs.sort_by(|x, y| x.0.total_cmp(&y.0));
        for i in 1..pairs.len() {
            assert!(pairs[i].1 >= pairs[i - 1].1, "not monotone: {pairs:?}");
        }
        // The smallest raw value carries the full n multiplier — this is what
        // pins the direction of the ranking.
        assert!(
            (pairs[0].1 - 5.0 * 0.001).abs() < 1e-12,
            "got {:?}",
            pairs[0]
        );
    }

    #[test]
    fn holm_on_a_single_p_value_is_the_identity() {
        let adj = holm_impl(&[0.02]).unwrap();
        assert!((adj[0] - 0.02).abs() < 1e-12);
    }

    #[test]
    fn benjamini_hochberg_matches_the_worked_example() {
        // n=4, sorted 0.01, 0.02, 0.03, 0.04 -> 4/1*0.01, 4/2*0.02, 4/3*0.03,
        // 4/4*0.04 = 0.04, 0.04, 0.04, 0.04 after the running minimum.
        let adj = benjamini_hochberg_impl(&[0.01, 0.02, 0.03, 0.04]).unwrap();
        for a in &adj {
            assert!((a - 0.04).abs() < 1e-12, "got {adj:?}");
        }
    }

    #[test]
    fn benjamini_hochberg_is_no_stricter_than_holm() {
        // FDR control is the weaker guarantee, so its adjustments never exceed
        // the family-wise ones.
        let raw = [0.001, 0.008, 0.02, 0.2, 0.9];
        let bh = benjamini_hochberg_impl(&raw).unwrap();
        let holm = holm_impl(&raw).unwrap();
        for (b, h) in bh.iter().zip(holm.iter()) {
            assert!(b <= h, "BH {b} exceeded Holm {h}");
        }
    }

    #[test]
    fn adjustments_reject_a_value_outside_the_unit_interval() {
        let err = holm_impl(&[0.5, 1.5]).unwrap_err();
        assert!(err.contains("p[2]"), "error should name the element: {err}");
        assert!(benjamini_hochberg_impl(&[]).is_err());
    }

    // ── effect size ─────────────────────────────────────────

    #[test]
    fn cohens_d_is_zero_for_identical_groups() {
        let g = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(cohens_d_impl(&g, &g).unwrap().abs() < 1e-12);
    }

    #[test]
    fn cohens_d_counts_in_pooled_standard_deviations() {
        // Two groups of unit sample variance, means one apart -> d = 1.
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 2.0, 3.0];
        let d = cohens_d_impl(&xs, &ys).unwrap();
        assert!((d + 1.0).abs() < 1e-12, "got {d}");
        // Antisymmetric in its arguments.
        assert!((cohens_d_impl(&ys, &xs).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cohens_d_refuses_two_constant_groups() {
        assert!(cohens_d_impl(&[2.0, 2.0], &[5.0, 5.0]).is_err());
    }

    #[test]
    fn cliffs_delta_spans_minus_one_to_one() {
        let low = [1.0, 2.0, 3.0];
        let high = [4.0, 5.0, 6.0];
        assert!((cliffs_delta_impl(&high, &low).unwrap() - 1.0).abs() < 1e-12);
        assert!((cliffs_delta_impl(&low, &high).unwrap() + 1.0).abs() < 1e-12);
        assert!(cliffs_delta_impl(&low, &low).unwrap().abs() < 1e-12);
    }

    #[test]
    fn cliffs_delta_ignores_the_size_of_the_gap() {
        // Ordinal: only the direction of each comparison is read, so widening
        // the separation does not move it. Cohen's d does move.
        let a = [1.0, 2.0];
        let near = [3.0, 4.0];
        let far = [300.0, 400.0];
        let d_near = cliffs_delta_impl(&a, &near).unwrap();
        let d_far = cliffs_delta_impl(&a, &far).unwrap();
        assert!((d_near - d_far).abs() < 1e-12, "{d_near} vs {d_far}");
        assert!(cohens_d_impl(&a, &near).unwrap().abs() < cohens_d_impl(&a, &far).unwrap().abs());
    }

    #[test]
    fn cliffs_delta_counts_ties_as_neither_direction() {
        // Half the pairs tie, half favour xs -> delta = 0.5.
        let xs = [1.0, 2.0];
        let ys = [1.0, 1.0];
        assert!((cliffs_delta_impl(&xs, &ys).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn effect_sizes_refuse_empty_input() {
        assert!(cliffs_delta_impl(&[], &[1.0]).is_err());
        assert!(cohens_d_impl(&[1.0], &[1.0, 2.0]).is_err());
    }
}
