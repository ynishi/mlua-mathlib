use mlua::prelude::*;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{
    Beta, Binomial, ChiSquared, Distribution, Exp, Gamma, LogNormal, Normal, Poisson, StudentT,
    Uniform,
};

use crate::rng::LuaRng;

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "normal_sample",
        lua.create_function(
            |_, (rng, mean, stddev): (LuaUserDataRef<LuaRng>, f64, f64)| {
                let dist = Normal::new(mean, stddev)
                    .map_err(|e| LuaError::runtime(format!("normal_sample: {e}")))?;
                Ok(dist.sample(&mut *rng.0.borrow_mut()))
            },
        )?,
    )?;

    t.set(
        "beta_sample",
        lua.create_function(
            |_, (rng, alpha, beta): (LuaUserDataRef<LuaRng>, f64, f64)| {
                let dist = Beta::new(alpha, beta)
                    .map_err(|e| LuaError::runtime(format!("beta_sample: {e}")))?;
                Ok(dist.sample(&mut *rng.0.borrow_mut()))
            },
        )?,
    )?;

    t.set(
        "gamma_sample",
        lua.create_function(
            |_, (rng, shape, scale): (LuaUserDataRef<LuaRng>, f64, f64)| {
                let dist = Gamma::new(shape, scale)
                    .map_err(|e| LuaError::runtime(format!("gamma_sample: {e}")))?;
                Ok(dist.sample(&mut *rng.0.borrow_mut()))
            },
        )?,
    )?;

    t.set(
        "exp_sample",
        lua.create_function(|_, (rng, lambda): (LuaUserDataRef<LuaRng>, f64)| {
            let dist =
                Exp::new(lambda).map_err(|e| LuaError::runtime(format!("exp_sample: {e}")))?;
            Ok(dist.sample(&mut *rng.0.borrow_mut()))
        })?,
    )?;

    t.set(
        "poisson_sample",
        lua.create_function(|_, (rng, lambda): (LuaUserDataRef<LuaRng>, f64)| {
            let dist = Poisson::new(lambda)
                .map_err(|e| LuaError::runtime(format!("poisson_sample: {e}")))?;
            let val: f64 = dist.sample(&mut *rng.0.borrow_mut());
            Ok(val.round().max(0.0) as u64)
        })?,
    )?;

    t.set(
        "uniform_sample",
        lua.create_function(|_, (rng, low, high): (LuaUserDataRef<LuaRng>, f64, f64)| {
            let dist = Uniform::new(low, high)
                .map_err(|e| LuaError::runtime(format!("uniform_sample: {e}")))?;
            Ok(dist.sample(&mut *rng.0.borrow_mut()))
        })?,
    )?;

    // ── v0.2 distributions ──────────────────────────────

    t.set(
        "lognormal_sample",
        lua.create_function(|_, (rng, mu, sigma): (LuaUserDataRef<LuaRng>, f64, f64)| {
            let dist = LogNormal::new(mu, sigma)
                .map_err(|e| LuaError::runtime(format!("lognormal_sample: {e}")))?;
            Ok(dist.sample(&mut *rng.0.borrow_mut()))
        })?,
    )?;

    t.set(
        "binomial_sample",
        lua.create_function(|_, (rng, n, p): (LuaUserDataRef<LuaRng>, u64, f64)| {
            let dist = Binomial::new(n, p)
                .map_err(|e| LuaError::runtime(format!("binomial_sample: {e}")))?;
            let val: u64 = dist.sample(&mut *rng.0.borrow_mut());
            Ok(val)
        })?,
    )?;

    t.set(
        "dirichlet_sample",
        lua.create_function(
            |lua, (rng, alphas_table): (LuaUserDataRef<LuaRng>, LuaTable)| {
                let len = alphas_table.raw_len();
                if len < 2 {
                    return Err(LuaError::runtime(
                        "dirichlet_sample: need at least 2 alpha values",
                    ));
                }
                let mut alphas = Vec::with_capacity(len);
                for i in 1..=len {
                    let v: f64 = alphas_table.raw_get(i)?;
                    alphas.push(v);
                }
                // Dirichlet via Gamma sampling (dynamic size, no const generics needed)
                let mut rng_mut = rng.0.borrow_mut();
                let mut samples = Vec::with_capacity(alphas.len());
                let mut sum = 0.0;
                for &a in &alphas {
                    let g = Gamma::new(a, 1.0)
                        .map_err(|e| LuaError::runtime(format!("dirichlet_sample: {e}")))?;
                    let val = g.sample(&mut *rng_mut);
                    samples.push(val);
                    sum += val;
                }
                if sum == 0.0 {
                    return Err(LuaError::runtime(
                        "dirichlet_sample: gamma samples sum to zero (alpha values too small?)",
                    ));
                }
                let out = lua.create_table()?;
                for (i, val) in samples.iter().enumerate() {
                    out.raw_set(i + 1, val / sum)?;
                }
                Ok(out)
            },
        )?,
    )?;

    t.set(
        "categorical_sample",
        lua.create_function(
            |_, (rng, weights_table): (LuaUserDataRef<LuaRng>, LuaTable)| {
                let len = weights_table.raw_len();
                if len == 0 {
                    return Err(LuaError::runtime(
                        "categorical_sample: need at least 1 weight",
                    ));
                }
                let mut weights = Vec::with_capacity(len);
                for i in 1..=len {
                    let v: f64 = weights_table.raw_get(i)?;
                    weights.push(v);
                }
                let dist = WeightedIndex::new(&weights)
                    .map_err(|e| LuaError::runtime(format!("categorical_sample: {e}")))?;
                // Return 1-based index for Lua convention
                let idx = dist.sample(&mut *rng.0.borrow_mut()) + 1;
                Ok(idx)
            },
        )?,
    )?;

    t.set(
        "student_t_sample",
        lua.create_function(|_, (rng, df): (LuaUserDataRef<LuaRng>, f64)| {
            let dist = StudentT::new(df)
                .map_err(|e| LuaError::runtime(format!("student_t_sample: {e}")))?;
            Ok(dist.sample(&mut *rng.0.borrow_mut()))
        })?,
    )?;

    t.set(
        "chi_squared_sample",
        lua.create_function(|_, (rng, df): (LuaUserDataRef<LuaRng>, f64)| {
            let dist = ChiSquared::new(df)
                .map_err(|e| LuaError::runtime(format!("chi_squared_sample: {e}")))?;
            Ok(dist.sample(&mut *rng.0.borrow_mut()))
        })?,
    )?;

    Ok(())
}
