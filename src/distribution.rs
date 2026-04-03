use mlua::prelude::*;
use rand_distr::{Beta, Distribution, Exp, Gamma, Normal, Poisson, Uniform};

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
            Ok(val as u64)
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

    Ok(())
}
