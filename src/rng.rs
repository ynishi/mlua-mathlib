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

    // shuffle: Fisher-Yates in-place shuffle, returns new table
    t.set(
        "shuffle",
        lua.create_function(|lua, (rng, table): (LuaUserDataRef<LuaRng>, LuaTable)| {
            let len = table.raw_len();
            if len == 0 {
                return lua.create_table();
            }
            // Read into vec
            let mut values: Vec<LuaValue> = Vec::with_capacity(len);
            for i in 1..=len {
                values.push(table.raw_get(i)?);
            }
            // Fisher-Yates
            let mut rng_ref = rng
                .0
                .try_borrow_mut()
                .map_err(|_| LuaError::runtime("shuffle: RNG is already borrowed"))?;
            for i in (1..values.len()).rev() {
                let j = rng_ref.random_range(0..=i);
                values.swap(i, j);
            }
            let out = lua.create_table()?;
            for (i, v) in values.into_iter().enumerate() {
                out.raw_set(i + 1, v)?;
            }
            Ok(out)
        })?,
    )?;

    // sample_with_replacement: draw n samples with replacement
    t.set(
        "sample_with_replacement",
        lua.create_function(
            |lua, (rng, table, n): (LuaUserDataRef<LuaRng>, LuaTable, usize)| {
                let len = table.raw_len();
                if len == 0 {
                    return Err(LuaError::runtime(
                        "sample_with_replacement: input must be non-empty",
                    ));
                }
                if n == 0 {
                    return lua.create_table();
                }
                let mut rng_ref = rng.0.try_borrow_mut().map_err(|_| {
                    LuaError::runtime("sample_with_replacement: RNG is already borrowed")
                })?;
                let out = lua.create_table()?;
                for i in 0..n {
                    let idx = rng_ref.random_range(1..=len);
                    let val: LuaValue = table.raw_get(idx)?;
                    out.raw_set(i + 1, val)?;
                }
                Ok(out)
            },
        )?,
    )?;

    Ok(())
}
