use mlua::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::{
    beta as beta_fn, erf as erf_fn, factorial as factorial_fn, gamma as gamma_fn,
};

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set("erf", lua.create_function(|_, x: f64| Ok(erf_fn::erf(x)))?)?;

    t.set(
        "erfc",
        lua.create_function(|_, x: f64| Ok(erf_fn::erfc(x)))?,
    )?;

    t.set(
        "lgamma",
        lua.create_function(|_, x: f64| Ok(gamma_fn::ln_gamma(x)))?,
    )?;

    t.set(
        "beta",
        lua.create_function(|_, (a, b): (f64, f64)| Ok(beta_fn::beta(a, b)))?,
    )?;

    t.set(
        "ln_beta",
        lua.create_function(|_, (a, b): (f64, f64)| Ok(beta_fn::ln_beta(a, b)))?,
    )?;

    t.set(
        "regularized_incomplete_beta",
        lua.create_function(|_, (x, a, b): (f64, f64, f64)| Ok(beta_fn::beta_reg(a, b, x)))?,
    )?;

    t.set(
        "regularized_incomplete_gamma",
        lua.create_function(|_, (a, x): (f64, f64)| Ok(gamma_fn::gamma_lr(a, x)))?,
    )?;

    t.set(
        "digamma",
        lua.create_function(|_, x: f64| Ok(gamma_fn::digamma(x)))?,
    )?;

    t.set(
        "factorial",
        lua.create_function(|_, n: u64| {
            if n > 170 {
                return Err(LuaError::runtime("factorial: n > 170 overflows f64"));
            }
            Ok(factorial_fn::factorial(n))
        })?,
    )?;

    t.set(
        "ln_factorial",
        lua.create_function(|_, n: u64| Ok(factorial_fn::ln_factorial(n)))?,
    )?;

    // logsumexp: numerically stable log(Σ exp(x_i))
    t.set(
        "logsumexp",
        lua.create_function(|_, table: LuaTable| {
            let v = crate::stats::table_to_vec(&table)?;
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if max.is_infinite() {
                return Ok(f64::NEG_INFINITY);
            }
            let sum: f64 = v.iter().map(|&x| (x - max).exp()).sum();
            Ok(max + sum.ln())
        })?,
    )?;

    // logit: log(p / (1-p))
    t.set(
        "logit",
        lua.create_function(|_, p: f64| {
            if p <= 0.0 || p >= 1.0 {
                return Err(LuaError::runtime(format!(
                    "logit: p must be in (0, 1), got {p}"
                )));
            }
            Ok((p / (1.0 - p)).ln())
        })?,
    )?;

    // expit (sigmoid): 1 / (1 + exp(-x))
    t.set(
        "expit",
        lua.create_function(|_, x: f64| {
            // Numerically stable sigmoid
            if x >= 0.0 {
                let ez = (-x).exp();
                Ok(1.0 / (1.0 + ez))
            } else {
                let ez = x.exp();
                Ok(ez / (1.0 + ez))
            }
        })?,
    )?;

    // normal_ppf: inverse CDF of N(0,1)
    t.set(
        "normal_ppf",
        lua.create_function(|_, p: f64| {
            if !(0.0..=1.0).contains(&p) {
                return Err(LuaError::runtime(format!(
                    "normal_ppf: p must be in [0, 1], got {p}"
                )));
            }
            let dist = Normal::new(0.0, 1.0).map_err(|e| LuaError::runtime(e.to_string()))?;
            Ok(dist.inverse_cdf(p))
        })?,
    )?;

    Ok(())
}
