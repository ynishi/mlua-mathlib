use mlua::prelude::*;
use statrs::distribution::{
    Beta, ContinuousCDF, DiscreteCDF, Gamma, Normal, Poisson as StatrsPoisson,
};

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    // ── CDF ──────────────────────────────────────────────

    t.set(
        "normal_cdf",
        lua.create_function(|_, (x, mu, sigma): (f64, f64, f64)| {
            let dist = Normal::new(mu, sigma)
                .map_err(|e| LuaError::runtime(format!("normal_cdf: {e}")))?;
            Ok(dist.cdf(x))
        })?,
    )?;

    t.set(
        "beta_cdf",
        lua.create_function(|_, (x, alpha, beta): (f64, f64, f64)| {
            let dist =
                Beta::new(alpha, beta).map_err(|e| LuaError::runtime(format!("beta_cdf: {e}")))?;
            Ok(dist.cdf(x))
        })?,
    )?;

    t.set(
        "gamma_cdf",
        lua.create_function(|_, (x, shape, scale): (f64, f64, f64)| {
            if scale <= 0.0 {
                return Err(LuaError::runtime("gamma_cdf: scale must be > 0"));
            }
            // statrs::Gamma uses rate (= 1/scale), rand_distr uses scale
            let dist = Gamma::new(shape, 1.0 / scale)
                .map_err(|e| LuaError::runtime(format!("gamma_cdf: {e}")))?;
            Ok(dist.cdf(x))
        })?,
    )?;

    t.set(
        "poisson_cdf",
        lua.create_function(|_, (k, lambda): (u64, f64)| {
            let dist = StatrsPoisson::new(lambda)
                .map_err(|e| LuaError::runtime(format!("poisson_cdf: {e}")))?;
            Ok(dist.cdf(k))
        })?,
    )?;

    // ── PPF (inverse CDF) ────────────────────────────────

    t.set(
        "normal_ppf_params",
        lua.create_function(|_, (p, mu, sigma): (f64, f64, f64)| {
            if !(0.0..=1.0).contains(&p) {
                return Err(LuaError::runtime(format!(
                    "normal_ppf_params: p must be in [0, 1], got {p}"
                )));
            }
            let dist = Normal::new(mu, sigma)
                .map_err(|e| LuaError::runtime(format!("normal_ppf_params: {e}")))?;
            Ok(dist.inverse_cdf(p))
        })?,
    )?;

    t.set(
        "beta_ppf",
        lua.create_function(|_, (p, alpha, beta): (f64, f64, f64)| {
            if !(0.0..=1.0).contains(&p) {
                return Err(LuaError::runtime(format!(
                    "beta_ppf: p must be in [0, 1], got {p}"
                )));
            }
            let dist =
                Beta::new(alpha, beta).map_err(|e| LuaError::runtime(format!("beta_ppf: {e}")))?;
            Ok(dist.inverse_cdf(p))
        })?,
    )?;

    // ── Distribution utilities ───────────────────────────

    t.set(
        "beta_mean",
        lua.create_function(|_, (alpha, beta): (f64, f64)| {
            if alpha <= 0.0 || beta <= 0.0 {
                return Err(LuaError::runtime("beta_mean: alpha and beta must be > 0"));
            }
            Ok(alpha / (alpha + beta))
        })?,
    )?;

    t.set(
        "beta_variance",
        lua.create_function(|_, (alpha, beta): (f64, f64)| {
            if alpha <= 0.0 || beta <= 0.0 {
                return Err(LuaError::runtime(
                    "beta_variance: alpha and beta must be > 0",
                ));
            }
            let ab = alpha + beta;
            Ok((alpha * beta) / (ab * ab * (ab + 1.0)))
        })?,
    )?;

    Ok(())
}
