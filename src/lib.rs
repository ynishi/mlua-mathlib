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

mod distribution;
mod rng;
mod stats;

use mlua::prelude::*;

/// Create the math module table with all functions registered.
pub fn module(lua: &Lua) -> LuaResult<LuaTable> {
    let t = lua.create_table()?;

    rng::register(lua, &t)?;
    distribution::register(lua, &t)?;
    stats::register(lua, &t)?;

    Ok(t)
}
