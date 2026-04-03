# mlua-mathlib

Math library for [mlua](https://github.com/mlua-rs/mlua) — RNG, distributions, special functions, and descriptive statistics.

Provides math functions that are impractical or numerically unstable to implement in pure Lua: distribution sampling with proper algorithms, independent seeded RNG instances, special functions (erf, gamma, beta), CDF/PPF, and numerically stable statistics.

## Features

- **Independent RNG instances** with seed control and reproducibility (ChaCha12 via `rand`)
- **12 distribution samplers** using production-grade algorithms (`rand_distr`)
- **Special functions** via `statrs` (erf, gamma, beta, digamma, factorial)
- **CDF/PPF** for Normal, Beta, Gamma, Poisson distributions
- **12 descriptive statistics** with numerical stability (Welford variance, interpolated percentiles, Wilson CI, stable softmax)

## Quick start

```toml
[dependencies]
mlua-mathlib = "0.2"
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
    print(math.normal_cdf(1.96, 0, 1))  -- ≈ 0.975
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
| `lognormal_sample(rng, mu, sigma)` | Log-normal | log-mean, log-stddev |
| `binomial_sample(rng, n, p)` | Binomial | trials, probability (returns integer) |
| `dirichlet_sample(rng, alphas)` | Dirichlet | concentration parameters (returns table) |
| `categorical_sample(rng, weights)` | Categorical | weights (returns 1-based index) |
| `student_t_sample(rng, df)` | Student's t | degrees of freedom |
| `chi_squared_sample(rng, df)` | Chi-squared | degrees of freedom |

### Special functions

| Function | Description |
|----------|-------------|
| `erf(x)` | Error function |
| `erfc(x)` | Complementary error function |
| `lgamma(x)` | Log-gamma function |
| `beta(a, b)` | Beta function |
| `ln_beta(a, b)` | Log-beta function |
| `regularized_incomplete_beta(x, a, b)` | Regularized incomplete beta (for Beta CDF) |
| `regularized_incomplete_gamma(a, x)` | Regularized lower incomplete gamma |
| `digamma(x)` | Digamma (psi) function |
| `factorial(n)` | Factorial (n <= 170) |
| `ln_factorial(n)` | Log-factorial |
| `normal_ppf(p)` | Inverse CDF of N(0,1) |

### CDF / PPF / Distribution utilities

| Function | Description |
|----------|-------------|
| `normal_cdf(x, mu, sigma)` | Normal CDF |
| `normal_ppf_params(p, mu, sigma)` | Normal inverse CDF (parameterized) |
| `beta_cdf(x, alpha, beta)` | Beta CDF |
| `beta_ppf(p, alpha, beta)` | Beta inverse CDF |
| `gamma_cdf(x, shape, rate)` | Gamma CDF |
| `poisson_cdf(k, lambda)` | Poisson CDF |
| `beta_mean(alpha, beta)` | Beta distribution mean |
| `beta_variance(alpha, beta)` | Beta distribution variance |

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
| `covariance(xs, ys)` | Sample covariance |
| `correlation(xs, ys)` | Pearson correlation coefficient |
| `histogram(values, bins)` | Histogram binning (returns `{counts, edges}`) |
| `wilson_ci(successes, total, confidence)` | Wilson score confidence interval (returns `{lower, upper, center}`) |
| `log_normalize(values)` | Logarithmic normalization to [0, 100] |

## Why not pure Lua?

| Problem | Pure Lua | mlua-mathlib |
|---------|----------|-------------|
| Beta/Gamma sampling | Complex algorithms (Joehnk, Marsaglia-Tsang), numerical instability | `rand_distr` with production-tested implementations |
| PRNG independence | Single global `math.random`, no instance isolation | Multiple independent seeded RNG instances |
| Special functions (erf, gamma) | No standard implementation; hand-rolled approximations | `statrs` with validated numerical methods |
| CDF/PPF | Requires special functions as building blocks | Exact implementations via `statrs` |
| Variance computation | Naive sum-of-squares suffers catastrophic cancellation | Welford's online algorithm |
| Wilson CI | Hardcoded z=1.96; no inverse normal function | Arbitrary confidence level via `normal_ppf` |

## Dependencies

| Crate | Purpose |
|-------|---------|
| [rand](https://crates.io/crates/rand) 0.9 | RNG (ChaCha12) |
| [rand_distr](https://crates.io/crates/rand_distr) 0.5 | Distribution sampling |
| [statrs](https://crates.io/crates/statrs) 0.18 | Special functions, CDF/PPF |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
