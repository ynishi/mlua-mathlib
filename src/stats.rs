use mlua::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

/// Extract a `Vec<f64>` from a Lua table (sequence).
/// Rejects NaN and Infinity values to ensure deterministic behavior.
fn table_to_vec(table: &LuaTable) -> LuaResult<Vec<f64>> {
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
fn sort_floats(v: &mut [f64]) {
    v.sort_by(|a, b| a.total_cmp(b));
}

/// Arithmetic mean.
fn mean_impl(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    values.iter().sum::<f64>() / n
}

/// Variance using Welford's online algorithm (numerically stable).
fn variance_impl(values: &[f64]) -> f64 {
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

/// Percentile with linear interpolation (exclusive method).
/// `p` is in [0, 100].
fn percentile_impl(sorted: &[f64], p: f64) -> f64 {
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
    if (max - min).abs() < f64::EPSILON {
        let mut counts = vec![0u64; bins];
        counts[0] = values.len() as u64;
        return (counts, 1.0 / bins as f64);
    }
    let width = (max - min) / bins as f64;
    let mut counts = vec![0u64; bins];
    for &val in values {
        let mut idx = ((val - min) / width) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
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

    // ── v0.2 statistics ────────────────────────────────

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
            let mean_x = mean_impl(&xs);
            let mean_y = mean_impl(&ys);
            let cov: f64 = xs
                .iter()
                .zip(ys.iter())
                .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
                .sum::<f64>()
                / (n - 1) as f64;
            Ok(cov)
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

            let result = lua.create_table()?;
            result.set("lower", (center - margin).max(0.0))?;
            result.set("upper", (center + margin).min(1.0))?;
            result.set("center", center)?;
            Ok(result)
        })?,
    )?;

    t.set(
        "log_normalize",
        lua.create_function(|lua, table: LuaTable| {
            let v = table_to_vec(&table)?;
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max <= 0.0 {
                return Err(LuaError::runtime(
                    "log_normalize: all values must be positive (max must be > 0)",
                ));
            }
            let log_max = (1.0 + max).ln();
            let out = lua.create_table()?;
            for (i, &val) in v.iter().enumerate() {
                let normalized = if val <= 0.0 {
                    0.0
                } else {
                    (1.0 + val).ln() / log_max * 100.0
                };
                out.raw_set(i + 1, normalized)?;
            }
            Ok(out)
        })?,
    )?;

    Ok(())
}
