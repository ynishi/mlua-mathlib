# mlua-mathlib

Math library for [mlua](https://github.com/mlua-rs/mlua) — RNG, distributions, and descriptive statistics.

Provides math functions that are impractical or numerically unstable to implement in pure Lua: distribution sampling with proper algorithms, independent seeded RNG instances, and numerically stable statistics.

## Features

- **Independent RNG instances** with seed control and reproducibility (ChaCha12 via `rand`)
- **6 distribution samplers** using production-grade algorithms (`rand_distr`)
- **7 descriptive statistics** with numerical stability (Welford variance, interpolated percentiles, stable softmax)

## Quick start

```toml
[dependencies]
mlua-mathlib = "0.1"
mlua = { version = "0.11", features = ["lua54", "vendored"] }
```

```rust
use mlua::prelude::*;

let lua = Lua::new();
let math = mlua_mathlib::module(&lua).unwrap();
lua.globals().set("math", math).unwrap();

lua.load(r#"
    local rng = math.rng_create(42)
    print(math.normal_sample(rng, 0.0, 1.0))
    print(math.mean({1, 2, 3, 4, 5}))
"#).exec().unwrap();
```

## API

### RNG

All sampling functions take an explicit RNG instance as the first argument. No global state.

| Function | Description |
|----------|-------------|
| `rng_create(seed)` | Create an independent RNG instance (ChaCha12) |
| `rng_float(rng)` | Sample uniform float in [0, 1) |
| `rng_int(rng, min, max)` | Sample uniform integer in [min, max] |

### Distribution sampling

| Function | Distribution | Parameters |
|----------|-------------|------------|
| `normal_sample(rng, mean, stddev)` | Normal | mean, standard deviation |
| `beta_sample(rng, alpha, beta)` | Beta | shape parameters |
| `gamma_sample(rng, shape, scale)` | Gamma | shape, scale |
| `exp_sample(rng, lambda)` | Exponential | rate |
| `poisson_sample(rng, lambda)` | Poisson | rate (returns integer) |
| `uniform_sample(rng, low, high)` | Uniform | lower, upper bound |

### Descriptive statistics

All functions take a Lua table (array) of numbers.

| Function | Description |
|----------|-------------|
| `mean(values)` | Arithmetic mean |
| `variance(values)` | Sample variance (Welford's algorithm) |
| `stddev(values)` | Sample standard deviation |
| `median(values)` | Median with linear interpolation |
| `percentile(values, p)` | p-th percentile (0-100) with linear interpolation |
| `iqr(values)` | Interquartile range (Q3 - Q1) |
| `softmax(values)` | Numerically stable softmax (returns table) |

## Why not pure Lua?

| Problem | Pure Lua | mlua-mathlib |
|---------|----------|-------------|
| Beta/Gamma sampling | Complex algorithms (Joehnk, Marsaglia-Tsang), numerical instability | `rand_distr` with production-tested implementations |
| PRNG independence | Single global `math.random`, no instance isolation | Multiple independent seeded RNG instances |
| Variance computation | Naive sum-of-squares suffers catastrophic cancellation | Welford's online algorithm |
| Percentile interpolation | Manual floor-only approximation | Linear interpolation (standard method) |
| Softmax | Overflow on large inputs | Max-subtraction trick for numerical stability |

## Dependencies

| Crate | Purpose |
|-------|---------|
| [rand](https://crates.io/crates/rand) 0.9 | RNG (ChaCha12) |
| [rand_distr](https://crates.io/crates/rand_distr) 0.5 | Distribution sampling |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
