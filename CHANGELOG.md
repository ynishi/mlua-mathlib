# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-09

### Added

- **Hypothesis testing**: 4 statistical tests
  - `welch_t_test(xs, ys)` — Welch's t-test (unequal variances)
  - `mann_whitney_u(xs, ys [, {tie_correction=true}])` — Mann-Whitney U test with optional tie correction
  - `chi_squared_test(observed, expected)` — Chi-squared goodness-of-fit test
  - `ks_test(xs, ys)` — Two-sample Kolmogorov-Smirnov test (asymptotic p-value with Stephens correction)
- **Ranking & IR metrics**: 5 functions
  - `rank(values)` — Fractional ranks with average tie-breaking
  - `spearman_correlation(xs, ys)` — Spearman rank correlation
  - `kendall_tau(xs, ys)` — Kendall's tau-b (handles ties)
  - `ndcg(relevance, k)` — NDCG@k (linear gain variant)
  - `mrr(rankings)` — Mean Reciprocal Rank
- **Information theory**: 4 functions
  - `entropy(probs)` — Shannon entropy (natural log)
  - `kl_divergence(p, q)` — KL divergence
  - `js_divergence(p, q)` — Jensen-Shannon divergence
  - `cross_entropy(p, q)` — Cross-entropy
- **Special functions**: 3 additions
  - `logsumexp(values)` — Numerically stable log-sum-exp
  - `logit(p)` — Log-odds
  - `expit(x)` — Sigmoid / inverse logit (numerically stable)
- **RNG**: 2 additions
  - `shuffle(rng, table)` — Fisher-Yates shuffle (returns new table)
  - `sample_with_replacement(rng, table, n)` — Draw n samples with replacement
- **Time series / combinatorics**: 4 additions
  - `moving_average(values, window)` — Simple moving average
  - `ewma(values, alpha)` — Exponentially weighted moving average
  - `autocorrelation(values, lag)` — Autocorrelation at given lag
  - `permutations(n)` — All n! permutations via Heap's algorithm (n ≤ 8)

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

[0.3.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.3.0
[0.2.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.2.0
[0.1.0]: https://github.com/ynishi/mlua-mathlib/releases/tag/v0.1.0
