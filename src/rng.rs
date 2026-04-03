use mlua::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;

/// Wrapper around `StdRng` exposed as mlua UserData.
///
/// Uses `RefCell` for interior mutability — Lua is single-threaded so
/// this is safe. `StdRng` uses ChaCha12 which passes all TestU01 suites.
pub(crate) struct LuaRng(pub(crate) RefCell<StdRng>);

impl LuaUserData for LuaRng {
    fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
        methods.add_meta_method(LuaMetaMethod::ToString, |_, _, ()| Ok("LuaRng(StdRng)"));
    }
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "rng_create",
        lua.create_function(|_, seed: u64| Ok(LuaRng(RefCell::new(StdRng::seed_from_u64(seed)))))?,
    )?;

    t.set(
        "rng_float",
        lua.create_function(|_, rng: LuaUserDataRef<LuaRng>| {
            let mut rng_ref = rng
                .0
                .try_borrow_mut()
                .map_err(|_| LuaError::runtime("rng_float: RNG is already borrowed"))?;
            let val: f64 = rng_ref.random();
            Ok(val)
        })?,
    )?;

    t.set(
        "rng_int",
        lua.create_function(|_, (rng, min, max): (LuaUserDataRef<LuaRng>, i64, i64)| {
            if min > max {
                return Err(LuaError::runtime(format!(
                    "rng_int: min ({min}) must be <= max ({max})"
                )));
            }
            let mut rng_ref = rng
                .0
                .try_borrow_mut()
                .map_err(|_| LuaError::runtime("rng_int: RNG is already borrowed"))?;
            let val = rng_ref.random_range(min..=max);
            Ok(val)
        })?,
    )?;

    Ok(())
}
