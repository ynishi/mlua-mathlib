use mlua::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

/// Extract a `Vec<f64>` from a Lua table (sequence).
/// Rejects NaN and Infinity values to ensure deterministic behavior.
pub(crate) fn table_to_vec(table: &LuaTable) -> LuaResult<Vec<f64>> {
    let len = table.raw_len();
    if len == 0 {
        return Err(LuaError::runtime("expected non-empty array"));
    }
    let mut v = Vec::with_capacity(len);
    for i in 1..=len {
        let val: f64 = table.raw_get(i)?;
        if val.is_nan() || val.is_infinite() {
            return Err(LuaError::runtime(format!(
                "element at index {i} is {val} (NaN/Infinity not allowed)"
            )));
        }
        v.push(val);
    }
    Ok(v)
}

/// Sort helper using total_cmp (NaN-safe, deterministic).
pub(crate) fn sort_floats(v: &mut [f64]) {
    v.sort_by(|a, b| a.total_cmp(b));
}

/// Arithmetic mean.
pub(crate) fn mean_impl(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    values.iter().sum::<f64>() / n
}

/// Variance using Welford's online algorithm (numerically stable).
pub(crate) fn variance_impl(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in values.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    m2 / (n - 1) as f64 // sample variance
}

/// Percentile with linear interpolation (inclusive method).
/// `p` is in [0, 100]. The rank is `p/100 * (n-1)`, matching R's type 7 and
/// numpy's `linear` — not the exclusive `p * (n+1)` variant. Shared with
/// [`crate::resample`] so bootstrap intervals take their endpoints the same
/// way `percentile` does.
pub(crate) fn percentile_impl(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    // Map p to index using linear interpolation
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = lo + 1;
    let frac = rank - lo as f64;
    if hi >= n {
        sorted[n - 1]
    } else {
        sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}

/// Compute histogram bin counts and width.
fn histogram_bin(values: &[f64], bins: usize, min: f64, max: f64) -> (Vec<u64>, f64) {
    let range = max - min;
    if range <= max.abs() * f64::EPSILON {
        let mut counts = vec![0u64; bins];
        // On 64-bit platforms usize→u64 is infallible; u64::MAX is a safe fallback for 128-bit.
        counts[0] = u64::try_from(values.len()).unwrap_or(u64::MAX);
        return (counts, 1.0 / bins as f64);
    }
    let width = range / bins as f64;
    let mut counts = vec![0u64; bins];
    for &val in values {
        let idx = ((val - min) / width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }
    (counts, width)
}

/// Convert histogram bins to a Lua table with counts and edges.
fn histogram_to_table(
    lua: &Lua,
    bin_counts: &[u64],
    bins: usize,
    min: f64,
    width: f64,
) -> LuaResult<LuaTable> {
    let counts = lua.create_table()?;
    for (i, &c) in bin_counts.iter().enumerate() {
        counts.raw_set(i + 1, c)?;
    }
    let edges = lua.create_table()?;
    for i in 0..=bins {
        edges.raw_set(i + 1, min + (i as f64) * width)?;
    }
    let result = lua.create_table()?;
    result.set("counts", counts)?;
    result.set("edges", edges)?;
    Ok(result)
}

/// Numerically stable softmax.
fn softmax_impl(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    register_descriptive(lua, t)?;
    register_bivariate(lua, t)?;
    register_timeseries(lua, t)?;
    register_combinatorics(lua, t)?;
    register_transforms(lua, t)?;
    Ok(())
}

fn register_descriptive(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "mean",
        lua.create_function(|_, table: LuaTable| {
            let v = table_to_vec(&table)?;
            Ok(mean_impl(&v))
        })?,
    )?;

    t.set(
        "variance",
        lua.create_function(|_, table: LuaTable| {
            let v = table_to_vec(&table)?;
            Ok(variance_impl(&v))
        })?,
    )?;

    t.set(
        "stddev",
        lua.create_function(|_, table: LuaTable| {
            let v = table_to_vec(&table)?;
            Ok(variance_impl(&v).sqrt())
        })?,
    )?;

    t.set(
        "median",
        lua.create_function(|_, table: LuaTable| {
            let mut v = table_to_vec(&table)?;
            sort_floats(&mut v);
            Ok(percentile_impl(&v, 50.0))
        })?,
    )?;

    t.set(
        "percentile",
        lua.create_function(|_, (table, p): (LuaTable, f64)| {
            if !(0.0..=100.0).contains(&p) {
                return Err(LuaError::runtime(format!(
                    "percentile: p must be in [0, 100], got {p}"
                )));
            }
            let mut v = table_to_vec(&table)?;
            sort_floats(&mut v);
            Ok(percentile_impl(&v, p))
        })?,
    )?;

    t.set(
        "iqr",
        lua.create_function(|_, table: LuaTable| {
            let mut v = table_to_vec(&table)?;
            sort_floats(&mut v);
            let q1 = percentile_impl(&v, 25.0);
            let q3 = percentile_impl(&v, 75.0);
            Ok(q3 - q1)
        })?,
    )?;

    t.set(
        "softmax",
        lua.create_function(|lua, table: LuaTable| {
            let v = table_to_vec(&table)?;
            let result = softmax_impl(&v);
            let out = lua.create_table()?;
            for (i, val) in result.into_iter().enumerate() {
                out.raw_set(i + 1, val)?;
            }
            Ok(out)
        })?,
    )?;

    Ok(())
}

fn register_bivariate(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "covariance",
        lua.create_function(|_, (xs_table, ys_table): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_table)?;
            let ys = table_to_vec(&ys_table)?;
            if xs.len() != ys.len() {
                return Err(LuaError::runtime(
                    "covariance: arrays must have equal length",
                ));
            }
            let n = xs.len();
            if n < 2 {
                return Err(LuaError::runtime("covariance: need at least 2 values"));
            }
            let mut mean_x = 0.0;
            let mut mean_y = 0.0;
            let mut co_m = 0.0;
            for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
                let k = (i + 1) as f64;
                let dx = x - mean_x;
                mean_x += dx / k;
                let dy = y - mean_y;
                mean_y += dy / k;
                co_m += dx * (y - mean_y);
            }
            Ok(co_m / (n - 1) as f64)
        })?,
    )?;

    t.set(
        "correlation",
        lua.create_function(|_, (xs_table, ys_table): (LuaTable, LuaTable)| {
            let xs = table_to_vec(&xs_table)?;
            let ys = table_to_vec(&ys_table)?;
            if xs.len() != ys.len() {
                return Err(LuaError::runtime(
                    "correlation: arrays must have equal length",
                ));
            }
            let n = xs.len();
            if n < 2 {
                return Err(LuaError::runtime("correlation: need at least 2 values"));
            }
            let mean_x = mean_impl(&xs);
            let mean_y = mean_impl(&ys);
            let mut cov = 0.0;
            let mut var_x = 0.0;
            let mut var_y = 0.0;
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                let dx = x - mean_x;
                let dy = y - mean_y;
                cov += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            let denom = (var_x * var_y).sqrt();
            if denom == 0.0 {
                return Err(LuaError::runtime("correlation: zero variance"));
            }
            Ok(cov / denom)
        })?,
    )?;

    Ok(())
}

fn register_timeseries(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    // moving_average: simple moving average with given window size
    t.set(
        "moving_average",
        lua.create_function(|lua, (table, window): (LuaTable, usize)| {
            let v = table_to_vec(&table)?;
            if window == 0 || window > v.len() {
                return Err(LuaError::runtime(format!(
                    "moving_average: window must be in [1, {}], got {window}",
                    v.len()
                )));
            }
            let out = lua.create_table()?;
            let mut sum: f64 = v[..window].iter().sum();
            out.raw_set(1, sum / window as f64)?;
            for i in window..v.len() {
                sum += v[i] - v[i - window];
                out.raw_set(i - window + 2, sum / window as f64)?;
            }
            Ok(out)
        })?,
    )?;

    // ewma: exponentially weighted moving average
    t.set(
        "ewma",
        lua.create_function(|lua, (table, alpha): (LuaTable, f64)| {
            if !(0.0..=1.0).contains(&alpha) {
                return Err(LuaError::runtime(format!(
                    "ewma: alpha must be in [0, 1], got {alpha}"
                )));
            }
            let v = table_to_vec(&table)?;
            let out = lua.create_table()?;
            // Safety: v is non-empty (table_to_vec rejects empty arrays).
            let mut ewma = v[0];
            out.raw_set(1, ewma)?;
            for (i, &val) in v.iter().enumerate().skip(1) {
                ewma = alpha * val + (1.0 - alpha) * ewma;
                out.raw_set(i + 1, ewma)?;
            }
            Ok(out)
        })?,
    )?;

    // autocorrelation at given lag
    t.set(
        "autocorrelation",
        lua.create_function(|_, (table, lag): (LuaTable, usize)| {
            let v = table_to_vec(&table)?;
            if lag == 0 {
                return Ok(1.0);
            }
            if lag >= v.len() {
                return Err(LuaError::runtime(format!(
                    "autocorrelation: lag must be < array length ({}), got {lag}",
                    v.len()
                )));
            }
            let mean = mean_impl(&v);
            let var: f64 = v.iter().map(|&x| (x - mean) * (x - mean)).sum();
            if var == 0.0 {
                return Err(LuaError::runtime("autocorrelation: zero variance"));
            }
            let cov: f64 = v[..v.len() - lag]
                .iter()
                .zip(v[lag..].iter())
                .map(|(&x, &y)| (x - mean) * (y - mean))
                .sum();
            Ok(cov / var)
        })?,
    )?;

    Ok(())
}

fn register_combinatorics(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    // permutations: generate all n! permutations of {1, ..., n}
    t.set(
        "permutations",
        lua.create_function(|lua, n: usize| {
            if n == 0 {
                return Err(LuaError::runtime("permutations: n must be >= 1"));
            }
            if n > 8 {
                return Err(LuaError::runtime(
                    "permutations: n > 8 not supported (8! = 40320)",
                ));
            }

            let mut arr: Vec<usize> = (1..=n).collect();
            let mut result: Vec<Vec<usize>> = Vec::new();

            // Heap's algorithm (iterative)
            let mut c = vec![0usize; n];
            result.push(arr.clone());

            let mut i = 0;
            while i < n {
                if c[i] < i {
                    if i % 2 == 0 {
                        arr.swap(0, i);
                    } else {
                        arr.swap(c[i], i);
                    }
                    result.push(arr.clone());
                    c[i] += 1;
                    i = 0;
                } else {
                    c[i] = 0;
                    i += 1;
                }
            }

            // Convert to Lua table of tables
            let out = lua.create_table()?;
            for (idx, perm) in result.iter().enumerate() {
                let row = lua.create_table()?;
                for (j, &val) in perm.iter().enumerate() {
                    row.raw_set(j + 1, val)?;
                }
                out.raw_set(idx + 1, row)?;
            }
            Ok(out)
        })?,
    )?;

    Ok(())
}

fn register_transforms(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "histogram",
        lua.create_function(|lua, (table, bins): (LuaTable, usize)| {
            if bins == 0 {
                return Err(LuaError::runtime("histogram: bins must be > 0"));
            }
            let v = table_to_vec(&table)?;
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let (bin_counts, width) = histogram_bin(&v, bins, min, max);
            histogram_to_table(lua, &bin_counts, bins, min, width)
        })?,
    )?;

    t.set(
        "wilson_ci",
        lua.create_function(|lua, (successes, total, confidence): (f64, f64, f64)| {
            if total <= 0.0 {
                return Err(LuaError::runtime("wilson_ci: total must be > 0"));
            }
            if !(0.0..=1.0).contains(&confidence) {
                return Err(LuaError::runtime("wilson_ci: confidence must be in [0, 1]"));
            }
            // z value from normal PPF
            let dist = Normal::new(0.0, 1.0).map_err(|e| LuaError::runtime(e.to_string()))?;
            let z = dist.inverse_cdf(1.0 - (1.0 - confidence) / 2.0);

            let p_hat = successes / total;
            let z2 = z * z;
            let denom = 1.0 + z2 / total;
            let center = (p_hat + z2 / (2.0 * total)) / denom;
            let margin =
                (z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total).sqrt()) / denom;

            // Clamped to [0, 1] as a probability, and to p̂ so the interval
            // always contains the estimate it brackets. In exact arithmetic it
            // already does — at p̂ = 1 the two halves of the numerator sum to
            // the denominator, so the upper end is exactly 1 — but the residue
            // lands a few ulp short in floating point, which would hand a
            // caller reading `lower` a bound that excludes its own estimate.
            let result = lua.create_table()?;
            result.set("lower", (center - margin).max(0.0).min(p_hat))?;
            result.set("upper", (center + margin).min(1.0).max(p_hat))?;
            result.set("center", center)?;
            Ok(result)
        })?,
    )?;

    t.set(
        "log_normalize",
        lua.create_function(|lua, table: LuaTable| {
            let v = table_to_vec(&table)?;
            if let Some((i, &val)) = v.iter().enumerate().find(|(_, &x)| x <= 0.0) {
                return Err(LuaError::runtime(format!(
                    "log_normalize: element at index {} is {val} (all values must be > 0)",
                    i + 1
                )));
            }
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_max = (1.0 + max).ln();
            let out = lua.create_table()?;
            for (i, &val) in v.iter().enumerate() {
                let normalized = (1.0 + val).ln() / log_max * 100.0;
                out.raw_set(i + 1, normalized)?;
            }
            Ok(out)
        })?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_impl_basic() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean_impl(&vals) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn mean_impl_single() {
        assert!((mean_impl(&[42.0]) - 42.0).abs() < 1e-15);
    }

    #[test]
    fn variance_impl_sample() {
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let expected = 4.571428571428571;
        assert!(
            (variance_impl(&vals) - expected).abs() < 1e-10,
            "sample variance mismatch: got {}",
            variance_impl(&vals)
        );
    }

    #[test]
    fn variance_impl_single_is_zero() {
        assert!((variance_impl(&[99.0])).abs() < 1e-15);
    }

    #[test]
    fn percentile_impl_median() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_impl(&sorted, 50.0) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn percentile_impl_interpolation() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let q25 = percentile_impl(&sorted, 25.0);
        assert!(
            (q25 - 3.25).abs() < 1e-10,
            "25th percentile should be 3.25, got {q25}"
        );
    }

    #[test]
    fn percentile_impl_single_element() {
        assert!((percentile_impl(&[42.0], 75.0) - 42.0).abs() < 1e-15);
    }

    #[test]
    fn histogram_bin_uniform() {
        let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (counts, width) = histogram_bin(&vals, 5, 1.0, 10.0);
        let total: u64 = counts.iter().sum();
        assert_eq!(total, 10, "all values must be binned");
        assert!(
            (width - 1.8).abs() < 1e-10,
            "width should be 9/5=1.8, got {width}"
        );
    }

    #[test]
    fn histogram_bin_all_same() {
        let vals = [5.0, 5.0, 5.0];
        let (counts, _) = histogram_bin(&vals, 3, 5.0, 5.0);
        assert_eq!(counts[0], 3, "all values should be in first bin");
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 0);
    }

    #[test]
    fn softmax_impl_sums_to_one() {
        let vals = [1.0, 2.0, 3.0];
        let result = softmax_impl(&vals);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn softmax_impl_preserves_order() {
        let vals = [1.0, 2.0, 3.0];
        let result = softmax_impl(&vals);
        assert!(
            result[0] < result[1] && result[1] < result[2],
            "softmax must preserve ordering"
        );
    }

    #[test]
    fn softmax_impl_numerical_stability() {
        let vals = [1000.0, 1001.0, 1002.0];
        let result = softmax_impl(&vals);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax with large values must still sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn sort_floats_basic() {
        let mut v = vec![3.0, 1.0, 2.0];
        sort_floats(&mut v);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }
}
