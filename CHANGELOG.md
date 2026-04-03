# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.1.0
