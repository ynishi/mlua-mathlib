#![deny(unsafe_code)]
//! Math library for mlua — RNG, distributions, and descriptive statistics.
//!
//! Provides math functions that are impractical or numerically unstable
//! to implement in pure Lua: distribution sampling with proper algorithms,
//! independent seeded RNG instances, and numerically stable statistics.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use mlua::prelude::*;
//!
//! let lua = Lua::new();
//! let math = mlua_mathlib::module(&lua).unwrap();
//! lua.globals().set("math", math).unwrap();
//!
//! lua.load(r#"
//!     local rng = math.rng_create(42)
//!     print(math.normal_sample(rng, 0.0, 1.0))
//!     print(math.mean({1, 2, 3, 4, 5}))
//! "#).exec().unwrap();
//! ```

mod cdf;
mod distribution;
mod rng;
mod special;
mod stats;

use mlua::prelude::*;

/// Create the math module table with all functions registered.
///
/// # Examples
///
/// ```rust,no_run
/// use mlua::prelude::*;
///
/// let lua = Lua::new();
/// let math = mlua_mathlib::module(&lua).unwrap();
/// lua.globals().set("math", math).unwrap();
///
/// // RNG
/// lua.load("local rng = math.rng_create(42); print(math.rng_float(rng))").exec().unwrap();
///
/// // Statistics
/// let mean: f64 = lua.load("return math.mean({1, 2, 3, 4, 5})").eval().unwrap();
/// assert!((mean - 3.0).abs() < 1e-10);
/// ```
pub fn module(lua: &Lua) -> LuaResult<LuaTable> {
    let t = lua.create_table()?;

    rng::register(lua, &t)?;
    distribution::register(lua, &t)?;
    stats::register(lua, &t)?;
    special::register(lua, &t)?;
    cdf::register(lua, &t)?;

    Ok(t)
}
