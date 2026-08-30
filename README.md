# mlua-mathlib

Math library for [mlua](https://github.com/mlua-rs/mlua) — RNG, distributions, special functions, and descriptive statistics.

Provides math functions that are impractical or numerically unstable to implement in pure Lua: distribution sampling with proper algorithms, independent seeded RNG instances, special functions (erf, gamma, beta), CDF/PPF, hypothesis testing, information theory, ranking metrics, and numerically stable statistics.

## Features

- **Independent RNG instances** with seed control and reproducibility (ChaCha12 via `rand`)
- **12 distribution samplers** using production-grade algorithms (`rand_distr`)
- **Special functions** via `statrs` (erf, gamma, beta, digamma, factorial)
- **CDF/PPF** for Normal, Beta, Gamma, Poisson distributions
- **16 descriptive & time-series statistics** with numerical stability (Welford variance, interpolated percentiles, Wilson CI, stable softmax, moving average, EWMA, autocorrelation)
- **5 hypothesis tests** (Welch's t, Mann-Whitney U, chi-squared, Kolmogorov-Smirnov, permutation)
- **5 bootstrap intervals** (mean / paired difference, and the cluster family: mean / difference / ratio)
- **2 multiple-comparison adjustments** (Holm-Bonferroni, Benjamini-Hochberg)
- **2 effect sizes** (Cohen's d, Cliff's delta)
- **2 calibration measures** (expected/maximum calibration error, Brier score)
- **5 ranking & IR metrics** (Spearman, Kendall tau-b, NDCG, MRR, fractional rank)
- **8 information-theoretic functions** (entropy, KL/JS divergence, cross-entropy, total variation, Hellinger, 1D Wasserstein, mutual information)

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
| `permutation_test(xs, ys, draws, seed [, opts])` | Permutation test on the difference in means. Pass `{alternative="greater"\|"less"}` for one-sided. Returns `{observed, p_value, extreme_draws, draws}` |

The first four tests each assume a shape — normality for Welch's t, continuity
for Mann-Whitney and Kolmogorov-Smirnov, expected counts for chi-squared.
`permutation_test` assumes only that the labels are exchangeable under the
null, at the cost of `draws` reshuffles.

Its p-value is `(1 + extreme) / (1 + draws)`, so the floor is `1/(1+draws)`
rather than zero: the observed arrangement is itself one of the permutations
under the null, and a p-value of exactly 0 claims more than any finite number
of draws can support.

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
| `hellinger(p, q)` | Hellinger distance sqrt(1 - Σ sqrt(pᵢqᵢ)) — a true metric, bounded [0, 1] |
| `wasserstein_1d(p, q [, support])` | Area between the CDFs; the only distance here that reads the support's order |
| `mutual_information(joint)` | I(X;Y) from a joint distribution matrix |

`mutual_information` takes the **joint** distribution — `joint[i][j] = P(X=i, Y=j)`,
row-major, summing to 1 — and derives the marginals from it. Zero exactly under
independence, bounded above by min(H(X), H(Y)). For the joint entropy on its own,
pass the flattened matrix to `entropy`.

Choosing among the distances: `tvd` and `hellinger` are both bounded metrics,
but `tvd` reads the *difference* of two probabilities while `hellinger` reads
their *ratio* — the latter separates two small probabilities differing by a
large factor where the former sees only a small gap. `js_divergence` is bounded
and symmetric but not a metric. `kl_divergence` is neither, and is the one to
reach for when the asymmetry is the point (a reference distribution against a
candidate).

`wasserstein_1d` is different in kind: every other distance here compares `pᵢ`
against `qᵢ` and is unchanged by permuting the bins. Wasserstein is not — it
measures how far the mass has to travel, so it is the right choice over ordered
outcomes and the wrong one over unordered categories, where the bin order is
arbitrary and so would be the answer.

The sum check tolerates `32 × sqrt(n) × 5.96e-8`, which covers the drift a
distribution normalized in f32 carries once widened to f64 — a 50257-entry
softmax lands around `1.2e-4`. Since a sum of `1 ± tol` is accepted, `tvd` can
return marginally more than 1 when both inputs drift upward, `js_divergence`
marginally more than ln 2, and `hellinger` marginally more than 1.
`kl_divergence` and `cross_entropy` return `math.huge` where `qᵢ = 0` while
`pᵢ > 0` rather than raising, since the definition is infinite there.
`wasserstein_1d` drops the final interval, where both CDFs have reached their
total and the difference is zero up to that same tolerance. Errors name the
element that failed, 1-based as in the Lua table (`p[3] is negative: -0.1` for
a pairwise call, `probs[3] ...` for `entropy`).

### Calibration

Whether a model's stated confidence matches how often it turns out right.

| Function | Description |
|----------|-------------|
| `calibration_error(confidences, outcomes, bins)` | Expected and maximum calibration error. Returns `{ece, mce, bins, bins_used}` |
| `brier_score(confidences, outcomes)` | Mean squared error of a probabilistic prediction |

`confidences` are probabilities in [0, 1] and `outcomes` are 0 or 1. The
returned `bins` array carries `{count, confidence, accuracy}` per bin, which is
what a reliability diagram is drawn from.

```lua
local r = math.calibration_error({0.9, 0.8, 0.6, 0.55}, {1, 1, 0, 1}, 10)
-- r.ece, r.mce, r.bins_used, r.bins[i].{count, confidence, accuracy}
```

The partition is equal-width over [0, 1] and that is not configurable. Equal-width
and equal-frequency binning give different numbers for the same predictions, as
does a different bin count, so an ECE is only comparable against one computed the
same way — a flag would make it easy to compare two numbers that are not
comparable. Empty bins contribute nothing rather than counting as a zero gap.

**An ECE of zero does not mean calibrated.** It means the confidences and the
outcomes averaged out within each bin of that partition, and errors in opposite
directions inside one bin cancel. A model that says 0.4 and is always right, and
says 0.6 and is always wrong, scores ECE 0 at two bins — and 0.55 at ten. Read a
low ECE alongside the bin count it was computed on and `bins_used`.

Read the two measures together. Brier is a **proper scoring rule** and ECE is
not, so it cannot be zeroed that way: the model above scores 0.36, worse than a
coin. The other direction is a model predicting the base rate for everything —
perfectly calibrated and useless, ECE 0 and Brier 0.25 at a 50% base rate. ECE
says whether the confidences are honest; Brier says whether they are also
informative.

The bin edges are left-closed, `[m/M, (m+1)/M)`, with `1.0` placed in the last
bin. Guo et al. define them right-closed; the numbers differ only for a
confidence landing exactly on an edge.

### Resampling

Bootstrap percentile intervals. Which function applies depends on how the
measurements arrive.

| Function | Description |
|----------|-------------|
| `bootstrap_mean(values, draws, seed [, conf])` | Independent observations, one per resampling unit |
| `paired_bootstrap_diff(xs, ys, draws, seed [, conf])` | `xs[i]` and `ys[i]` measure the same item; the difference is taken per pair first |
| `cluster_bootstrap_mean(by_cluster, draws, seed [, conf])` | Percentile interval on the mean |
| `cluster_bootstrap_diff(a_by_cluster, b_by_cluster, draws, seed [, conf])` | Interval on the difference, both sides from the same draw |
| `cluster_bootstrap_ratio(num_by_cluster, den_by_cluster, draws, seed [, conf])` | Interval on the ratio of the summed sides |

The **cluster** family resamples the group rather than the observation, for
measurements that arrive correlated (repeated samples from one prompt, several
positions from one game). Resampling observations independently there would
understate the spread. The first two functions are the flat-series case, where
each observation is its own unit; they report `observations` in place of
`clusters`.

Pairing and clustering answer different questions and compose: `xs[i]` vs
`ys[i]` is *the same item measured twice*, while a cluster is *one group of
correlated measurements*. Use `paired_bootstrap_diff` for two models on one
prompt set, `cluster_bootstrap_diff` when each of those prompts also carries
several measurements.

`by_cluster` is a sequence of sequences — one inner table per cluster, holding
that cluster's observations. An empty cluster is allowed and still takes part
in the resampling. At least 2 clusters are required (one cluster gives the same
draw every time), though a trustworthy interval wants dozens.

**`diff` and `ratio` are paired**: one draw is applied to both sides, so
`a_by_cluster[i]` and `b_by_cluster[i]` must measure the **same** cluster — two
models evaluated on the same prompt set, indexed the same way. Only the count
can be checked; passing two independent groups that happen to be the same length
manufactures a correlation that is not there and reports an interval far too
narrow. For genuinely independent groups, bootstrap each side separately.

```lua
local by_game = {{0.3, 0.5}, {0.2}, {}, {0.8, 0.1, 0.4}}
local ci = math.cluster_bootstrap_mean(by_game, 2000, 42)
-- {point, lower, upper, draws_used, undefined_draws, clusters}
```

`undefined_draws` counts the resamples on which the statistic had no value —
no observations for a mean, a zero or sign-flipped denominator for a ratio. A
large share is the signal that the statistic rests on too few clusters to
resample. `conf` defaults to 0.95, and endpoints are taken with the same
interpolated `percentile` used elsewhere in this crate. A given seed reproduces
a given interval.

Those draws are dropped rather than replaced, so the interval is over the
resample distribution **conditioned on the statistic being defined** —
coverage arguments for the unconditional bootstrap do not carry over unchanged.
`undefined_draws` is what makes that visible.

A percentile interval need not contain the point estimate. For a mean it
effectively always does; for a skewed statistic such as a ratio it can sit
entirely to one side. That is the method reporting the shape of the resample
distribution, and unlike `wilson_ci` — where the endpoints are clamped because
exact arithmetic already contains `p̂` — there is nothing here to repair.

`diff` and `ratio` are separate functions rather than something you compose
from `mean`: two separately bootstrapped quantities carry no joint
distribution, so their intervals cannot be combined after the fact. Both sides
have to be measured inside the same draw.

`ratio` requires the denominator to keep one sign. A draw whose denominator sum
crosses zero relative to the whole sample is counted as undefined, since the
ratio there is a different quantity rather than a perturbation of the estimate.

### Multiple comparison & effect size

| Function | Description |
|----------|-------------|
| `holm(p_values)` | Holm-Bonferroni step-down; controls the family-wise error rate |
| `benjamini_hochberg(p_values)` | Step-up; controls the false discovery rate |
| `cohens_d(xs, ys)` | Difference in means, in pooled standard deviations |
| `cliffs_delta(xs, ys)` | `P(x > y) - P(x < y)`, in [-1, 1] |

The adjustments return a table of the same length, in the input order; compare
each against the uncorrected level. Holm gives the stronger guarantee (no false
rejection anywhere in the family), Benjamini-Hochberg the weaker one (a bounded
share of the rejections made) in exchange for power.

Effect sizes answer "by how much", which a p-value does not — any difference
reaches any significance level given enough observations. `cohens_d` assumes
comparable spread in the two groups; `cliffs_delta` reads only the direction of
each pairwise comparison, so a heavy tail does not distort it.

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
