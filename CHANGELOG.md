# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-04

### Added

- **Distributions**: 6 additional sampling functions (dependency: `statrs`)
  - `lognormal_sample(rng, mu, sigma)` — Log-normal distribution
  - `binomial_sample(rng, n, p)` — Binomial distribution (returns integer)
  - `dirichlet_sample(rng, alphas)` — Dirichlet distribution (dynamic size)
  - `categorical_sample(rng, weights)` — Weighted random choice (1-based index)
  - `student_t_sample(rng, df)` — Student's t-distribution
  - `chi_squared_sample(rng, df)` — Chi-squared distribution
- **Special functions**: 10 functions via `statrs`
  - `erf(x)`, `erfc(x)` — error function and complement
  - `lgamma(x)` — log-gamma
  - `beta(a, b)`, `ln_beta(a, b)` — beta function
  - `regularized_incomplete_beta(x, a, b)` — for Beta CDF
  - `regularized_incomplete_gamma(a, x)` — for Gamma/Poisson CDF
  - `digamma(x)` — psi function
  - `factorial(n)`, `ln_factorial(n)` — factorial (n <= 170)
  - `normal_ppf(p)` — inverse CDF of N(0,1)
- **CDF/PPF**: distribution cumulative/inverse functions
  - `normal_cdf(x, mu, sigma)`, `normal_ppf_params(p, mu, sigma)`
  - `beta_cdf(x, alpha, beta)`, `beta_ppf(p, alpha, beta)`
  - `gamma_cdf(x, shape, rate)`
  - `poisson_cdf(k, lambda)`
  - `beta_mean(alpha, beta)`, `beta_variance(alpha, beta)`
- **Statistics**: 5 additional functions
  - `covariance(xs, ys)` — sample covariance
  - `correlation(xs, ys)` — Pearson correlation coefficient
  - `histogram(values, bins)` — returns `{counts, edges}`
  - `wilson_ci(successes, total, confidence)` — Wilson score confidence interval
  - `log_normalize(values)` — logarithmic normalization to [0, 100]

### Changed

- New dependency: `statrs` 0.18

## [0.1.0] - 2026-04-04

### Added

- **RNG**: `rng_create(seed)`, `rng_float(rng)`, `rng_int(rng, min, max)` — independent seeded RNG instances using ChaCha12
- **Distributions**: 6 sampling functions
  - `normal_sample(rng, mean, stddev)` — Normal distribution (ziggurat algorithm)
  - `beta_sample(rng, alpha, beta)` — Beta distribution
  - `gamma_sample(rng, shape, scale)` — Gamma distribution
  - `exp_sample(rng, lambda)` — Exponential distribution
  - `poisson_sample(rng, lambda)` — Poisson distribution (returns integer)
  - `uniform_sample(rng, low, high)` — Continuous uniform distribution
- **Statistics**: 7 descriptive statistics functions
  - `mean(values)` — arithmetic mean
  - `variance(values)` — sample variance (Welford's algorithm)
  - `stddev(values)` — sample standard deviation
  - `median(values)` — median with linear interpolation
  - `percentile(values, p)` — p-th percentile with linear interpolation
  - `iqr(values)` — interquartile range
  - `softmax(values)` — numerically stable softmax

[0.2.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.2.0
[0.1.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.1.0
