# mlua-mathlib

Math library for [mlua](https://github.com/mlua-rs/mlua) — RNG, distributions, special functions, and descriptive statistics.

Provides math functions that are impractical or numerically unstable to implement in pure Lua: distribution sampling with proper algorithms, independent seeded RNG instances, special functions (erf, gamma, beta), CDF/PPF, hypothesis testing, information theory, ranking metrics, and numerically stable statistics.

## Features

- **Independent RNG instances** with seed control and reproducibility (ChaCha12 via `rand`)
- **12 distribution samplers** using production-grade algorithms (`rand_distr`)
- **Special functions** via `statrs` (erf, gamma, beta, digamma, factorial)
- **CDF/PPF** for Normal, Beta, Gamma, Poisson distributions
- **16 descriptive & time-series statistics** with numerical stability (Welford variance, interpolated percentiles, Wilson CI, stable softmax, moving average, EWMA, autocorrelation)
- **4 hypothesis tests** (Welch's t, Mann-Whitney U, chi-squared, Kolmogorov-Smirnov)
- **5 ranking & IR metrics** (Spearman, Kendall tau-b, NDCG, MRR, fractional rank)
- **5 information-theoretic functions** (entropy, KL divergence, JS divergence, cross-entropy, total variation distance)

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
| `shuffle(rng, table)` | Fisher-Yates shuffle (returns new table) |
| `sample_with_replacement(rng, table, n)` | Draw n samples with replacement |

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
| `logsumexp(values)` | Numerically stable log-sum-exp |
| `logit(p)` | Log-odds: ln(p/(1-p)) |
| `expit(x)` | Sigmoid / inverse logit (numerically stable) |

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
| `moving_average(values, window)` | Simple moving average |
| `ewma(values, alpha)` | Exponentially weighted moving average |
| `autocorrelation(values, lag)` | Autocorrelation at given lag |
| `permutations(n)` | All n! permutations of {1..n} (n ≤ 8, returns table of tables) |

### Hypothesis testing

All tests return a table with test statistic(s) and p-value.

| Function | Description |
|----------|-------------|
| `welch_t_test(xs, ys)` | Welch's t-test (unequal variances). Returns `{t_stat, df, p_value}` |
| `mann_whitney_u(xs, ys [, opts])` | Mann-Whitney U test. Pass `{tie_correction=true}` as 3rd arg to adjust for ties. Returns `{u_stat, z_score, p_value}` |
| `chi_squared_test(observed, expected)` | Chi-squared goodness-of-fit. Returns `{chi2_stat, df, p_value}` |
| `ks_test(xs, ys)` | Two-sample Kolmogorov-Smirnov test. Returns `{d_stat, p_value}` |

### Ranking & IR metrics

| Function | Description |
|----------|-------------|
| `rank(values)` | Fractional ranks with average tie-breaking (returns table) |
| `spearman_correlation(xs, ys)` | Spearman rank correlation coefficient |
| `kendall_tau(xs, ys)` | Kendall's tau-b (handles ties) |
| `ndcg(relevance, k)` | NDCG@k (linear gain variant: rel/log₂(i+2)) |
| `mrr(rankings)` | Mean Reciprocal Rank (1-based rank positions) |

### Information theory

Input distributions must be valid probability distributions (non-negative, sum to 1).

| Function | Description |
|----------|-------------|
| `entropy(probs)` | Shannon entropy H(p) = -Σ pᵢ ln(pᵢ) |
| `kl_divergence(p, q)` | KL divergence D_KL(p ‖ q) |
| `js_divergence(p, q)` | Jensen-Shannon divergence (symmetric, bounded [0, ln 2]) |
| `cross_entropy(p, q)` | Cross-entropy H(p, q) = -Σ pᵢ ln(qᵢ) |
| `tvd(p, q)` | Total variation distance 0.5 Σ\|pᵢ - qᵢ\| (symmetric, bounded [0, 1]) |

The sum check tolerates `32 × sqrt(n) × 5.96e-8`, which covers the drift a
distribution normalized in f32 carries once widened to f64 — a 50257-entry
softmax lands around `1.2e-4`. Since a sum of `1 ± tol` is accepted, `tvd` can
return marginally more than 1 when both inputs drift upward, and `js_divergence`
marginally more than ln 2. `kl_divergence` and `cross_entropy` return `math.huge`
where `qᵢ = 0` while `pᵢ > 0` rather than raising, since the definition is
infinite there. Errors name the element that failed, 1-based as in the Lua table
(`p[3] is negative: -0.1` for a pairwise call, `probs[3] ...` for `entropy`).

## Why not pure Lua?

| Problem | Pure Lua | mlua-mathlib |
|---------|----------|-------------|
| Beta/Gamma sampling | Complex algorithms (Joehnk, Marsaglia-Tsang), numerical instability | `rand_distr` with production-tested implementations |
| PRNG independence | Single global `math.random`, no instance isolation | Multiple independent seeded RNG instances |
| Special functions (erf, gamma) | No standard implementation; hand-rolled approximations | `statrs` with validated numerical methods |
| CDF/PPF | Requires special functions as building blocks | Exact implementations via `statrs` |
| Variance computation | Naive sum-of-squares suffers catastrophic cancellation | Welford's online algorithm |
| Wilson CI | Hardcoded z=1.96; no inverse normal function | Arbitrary confidence level via `normal_ppf` |
| Hypothesis tests | Requires CDF tables or lookup; manual formula implementation | Exact p-values via `statrs` distributions |
| KL/JS divergence | Numerical instability with small probabilities | Proper log-domain computation with validation |

## Dependencies

| Crate | Purpose |
|-------|---------|
| [rand](https://crates.io/crates/rand) 0.9 | RNG (ChaCha12) |
| [rand_distr](https://crates.io/crates/rand_distr) 0.5 | Distribution sampling |
| [statrs](https://crates.io/crates/statrs) 0.18 | Special functions, CDF/PPF |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
