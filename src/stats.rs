use mlua::prelude::*;

/// Extract a `Vec<f64>` from a Lua table (sequence).
fn table_to_vec(table: &LuaTable) -> LuaResult<Vec<f64>> {
    let len = table.raw_len();
    if len == 0 {
        return Err(LuaError::runtime("expected non-empty array"));
    }
    let mut v = Vec::with_capacity(len);
    for i in 1..=len {
        let val: f64 = table.raw_get(i)?;
        v.push(val);
    }
    Ok(v)
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
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(percentile_impl(&v, p))
        })?,
    )?;

    t.set(
        "iqr",
        lua.create_function(|_, table: LuaTable| {
            let mut v = table_to_vec(&table)?;
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
