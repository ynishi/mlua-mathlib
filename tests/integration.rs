use mlua::prelude::*;

fn setup() -> Lua {
    let lua = Lua::new();
    let math = mlua_mathlib::module(&lua).unwrap();
    lua.globals().set("math", math).unwrap();
    lua
}

// ── RNG ──────────────────────────────────────────────────

#[test]
fn rng_create_and_float() {
    let lua = setup();
    let val: f64 = lua
        .load("local rng = math.rng_create(42); return math.rng_float(rng)")
        .eval()
        .unwrap();
    assert!((0.0..1.0).contains(&val));
}

#[test]
fn rng_deterministic() {
    let lua = setup();
    let code = r#"
        local rng1 = math.rng_create(123)
        local rng2 = math.rng_create(123)
        local a = math.rng_float(rng1)
        local b = math.rng_float(rng2)
        return a == b
    "#;
    let same: bool = lua.load(code).eval().unwrap();
    assert!(same, "same seed must produce same sequence");
}

#[test]
fn rng_int_range() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(99)
        local results = {}
        for i = 1, 100 do
            local v = math.rng_int(rng, 1, 6)
            if v < 1 or v > 6 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "rng_int values must be in [1, 6]");
}

#[test]
fn rng_int_min_gt_max_errors() {
    let lua = setup();
    let result: LuaResult<i64> = lua
        .load("local rng = math.rng_create(1); return math.rng_int(rng, 10, 5)")
        .eval();
    assert!(result.is_err());
}

// ── Distributions ────────────────────────────────────────

#[test]
fn normal_sample_basic() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        local sum = 0
        local n = 10000
        for i = 1, n do sum = sum + math.normal_sample(rng, 0.0, 1.0) end
        return sum / n
    "#;
    let mean: f64 = lua.load(code).eval().unwrap();
    assert!(
        mean.abs() < 0.1,
        "N(0,1) mean of 10k samples should be near 0, got {mean}"
    );
}

#[test]
fn beta_sample_in_unit_interval() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.beta_sample(rng, 2.0, 5.0)
            if v < 0 or v > 1 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Beta samples must be in [0, 1]");
}

#[test]
fn gamma_sample_positive() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.gamma_sample(rng, 2.0, 1.0)
            if v <= 0 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Gamma samples must be positive");
}

#[test]
fn exp_sample_positive() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.exp_sample(rng, 1.5)
            if v <= 0 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Exp samples must be positive");
}

#[test]
fn poisson_sample_non_negative() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.poisson_sample(rng, 5.0)
            if v < 0 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Poisson samples must be non-negative");
}

#[test]
fn uniform_sample_in_range() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.uniform_sample(rng, 10.0, 20.0)
            if v < 10 or v >= 20 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Uniform samples must be in [10, 20)");
}

#[test]
fn normal_invalid_stddev_errors() {
    let lua = setup();
    // NaN triggers rand_distr error
    let result: LuaResult<f64> = lua
        .load("local rng = math.rng_create(1); return math.normal_sample(rng, 0, 0/0)")
        .eval();
    assert!(result.is_err());
}

// ── Statistics ───────────────────────────────────────────

#[test]
fn mean_basic() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.mean({1, 2, 3, 4, 5})")
        .eval()
        .unwrap();
    assert!((val - 3.0).abs() < 1e-10);
}

#[test]
fn variance_basic() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.variance({2, 4, 4, 4, 5, 5, 7, 9})")
        .eval()
        .unwrap();
    // sample variance = 4.571428...
    assert!((val - 4.571428571428571).abs() < 1e-10);
}

#[test]
fn stddev_basic() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.stddev({2, 4, 4, 4, 5, 5, 7, 9})")
        .eval()
        .unwrap();
    let expected = 4.571428571428571_f64.sqrt();
    assert!((val - expected).abs() < 1e-10);
}

#[test]
fn median_odd() {
    let lua = setup();
    let val: f64 = lua.load("return math.median({3, 1, 2})").eval().unwrap();
    assert!((val - 2.0).abs() < 1e-10);
}

#[test]
fn median_even() {
    let lua = setup();
    let val: f64 = lua.load("return math.median({1, 2, 3, 4})").eval().unwrap();
    assert!((val - 2.5).abs() < 1e-10);
}

#[test]
fn percentile_basic() {
    let lua = setup();
    // 25th percentile of {1,2,3,4,5,6,7,8,9,10}
    let val: f64 = lua
        .load("return math.percentile({1,2,3,4,5,6,7,8,9,10}, 25)")
        .eval()
        .unwrap();
    // rank = 0.25 * 9 = 2.25 → lerp(3, 4, 0.25) = 3.25
    assert!((val - 3.25).abs() < 1e-10);
}

#[test]
fn percentile_out_of_range_errors() {
    let lua = setup();
    let result: LuaResult<f64> = lua.load("return math.percentile({1,2,3}, 101)").eval();
    assert!(result.is_err());
}

#[test]
fn iqr_basic() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.iqr({1,2,3,4,5,6,7,8,9,10})")
        .eval()
        .unwrap();
    // Q3 - Q1 = 7.75 - 3.25 = 4.5
    let q1 = 3.25;
    let q3 = 7.75;
    assert!((val - (q3 - q1)).abs() < 1e-10);
}

#[test]
fn softmax_basic() {
    let lua = setup();
    let code = r#"
        local result = math.softmax({1, 2, 3})
        local sum = 0
        for _, v in ipairs(result) do sum = sum + v end
        return sum
    "#;
    let sum: f64 = lua.load(code).eval().unwrap();
    assert!((sum - 1.0).abs() < 1e-10, "softmax should sum to 1");
}

#[test]
fn softmax_ordering() {
    let lua = setup();
    let code = r#"
        local result = math.softmax({1, 2, 3})
        return result[1] < result[2] and result[2] < result[3]
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "softmax should preserve ordering");
}

#[test]
fn mean_empty_errors() {
    let lua = setup();
    let result: LuaResult<f64> = lua.load("return math.mean({})").eval();
    assert!(result.is_err());
}

#[test]
fn variance_single_element() {
    let lua = setup();
    let val: f64 = lua.load("return math.variance({42})").eval().unwrap();
    assert!((val - 0.0).abs() < 1e-10);
}

#[test]
fn nan_rejected_in_stats() {
    let lua = setup();
    let result: LuaResult<f64> = lua.load("return math.mean({1, 0/0, 3})").eval();
    assert!(result.is_err(), "NaN should be rejected");
}

#[test]
fn infinity_rejected_in_stats() {
    let lua = setup();
    let result: LuaResult<f64> = lua.load("return math.mean({1, 1/0, 3})").eval();
    assert!(result.is_err(), "Infinity should be rejected");
}

// ── v0.2 Distributions ──────────────────────────────────

#[test]
fn lognormal_sample_positive() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            if math.lognormal_sample(rng, 0.0, 1.0) <= 0 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "LogNormal samples must be positive");
}

#[test]
fn binomial_sample_range() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local v = math.binomial_sample(rng, 10, 0.5)
            if v < 0 or v > 10 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Binomial(10, 0.5) samples must be in [0, 10]");
}

#[test]
fn dirichlet_sample_sums_to_one() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        local result = math.dirichlet_sample(rng, {1.0, 2.0, 3.0})
        local sum = 0
        for _, v in ipairs(result) do
            if v < 0 then return -1 end
            sum = sum + v
        end
        return sum
    "#;
    let sum: f64 = lua.load(code).eval().unwrap();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Dirichlet must sum to 1, got {sum}"
    );
}

#[test]
fn categorical_sample_valid_index() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            local idx = math.categorical_sample(rng, {0.1, 0.3, 0.6})
            if idx < 1 or idx > 3 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "categorical must return 1-based index");
}

#[test]
fn student_t_sample_basic() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        local sum = 0
        for i = 1, 10000 do sum = sum + math.student_t_sample(rng, 30) end
        return sum / 10000
    "#;
    let mean: f64 = lua.load(code).eval().unwrap();
    assert!(mean.abs() < 0.1, "StudentT(30) mean ~ 0, got {mean}");
}

#[test]
fn chi_squared_sample_positive() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        for i = 1, 1000 do
            if math.chi_squared_sample(rng, 5.0) <= 0 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Chi-squared samples must be positive");
}

// ── v0.2 Special Functions ──────────────────────────────

#[test]
fn erf_known_values() {
    let lua = setup();
    let val: f64 = lua.load("return math.erf(0)").eval().unwrap();
    assert!(val.abs() < 1e-15);
    let val: f64 = lua.load("return math.erf(1)").eval().unwrap();
    assert!((val - 0.8427007929497149).abs() < 1e-10);
}

#[test]
fn erfc_complement() {
    let lua = setup();
    let code = "return math.erf(1.5) + math.erfc(1.5)";
    let val: f64 = lua.load(code).eval().unwrap();
    assert!((val - 1.0).abs() < 1e-15);
}

#[test]
fn lgamma_known() {
    let lua = setup();
    // lgamma(1) = 0, lgamma(5) = ln(24) = 3.178...
    let val: f64 = lua.load("return math.lgamma(1)").eval().unwrap();
    assert!(val.abs() < 1e-10);
    let val: f64 = lua.load("return math.lgamma(5)").eval().unwrap();
    assert!((val - 24.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn beta_function_known() {
    let lua = setup();
    // B(1,1) = 1
    let val: f64 = lua.load("return math.beta(1, 1)").eval().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn regularized_incomplete_beta_boundaries() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.regularized_incomplete_beta(0, 2, 3)")
        .eval()
        .unwrap();
    assert!(val.abs() < 1e-10);
    let val: f64 = lua
        .load("return math.regularized_incomplete_beta(1, 2, 3)")
        .eval()
        .unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn digamma_known() {
    let lua = setup();
    // digamma(1) = -euler_mascheroni ≈ -0.5772
    let val: f64 = lua.load("return math.digamma(1)").eval().unwrap();
    assert!((val - (-0.5772156649015329)).abs() < 1e-8);
}

#[test]
fn factorial_basic() {
    let lua = setup();
    let val: f64 = lua.load("return math.factorial(5)").eval().unwrap();
    assert!((val - 120.0).abs() < 1e-10);
    let val: f64 = lua.load("return math.factorial(0)").eval().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn factorial_overflow_errors() {
    let lua = setup();
    let result: LuaResult<f64> = lua.load("return math.factorial(171)").eval();
    assert!(result.is_err());
}

#[test]
fn normal_ppf_known() {
    let lua = setup();
    // ppf(0.5) = 0 (median of N(0,1))
    let val: f64 = lua.load("return math.normal_ppf(0.5)").eval().unwrap();
    assert!(val.abs() < 1e-10);
    // ppf(0.975) ≈ 1.96
    let val: f64 = lua.load("return math.normal_ppf(0.975)").eval().unwrap();
    assert!((val - 1.959963984540054).abs() < 1e-6);
}

// ── v0.2 CDF / PPF ─────────────────────────────────────

#[test]
fn normal_cdf_known() {
    let lua = setup();
    let val: f64 = lua.load("return math.normal_cdf(0, 0, 1)").eval().unwrap();
    assert!((val - 0.5).abs() < 1e-10);
}

#[test]
fn beta_cdf_boundaries() {
    let lua = setup();
    let val: f64 = lua.load("return math.beta_cdf(0, 2, 5)").eval().unwrap();
    assert!(val.abs() < 1e-10);
    let val: f64 = lua.load("return math.beta_cdf(1, 2, 5)").eval().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn gamma_cdf_scale_consistent() {
    let lua = setup();
    // Gamma(shape=1, scale=1) is Exp(1). CDF at x=1 = 1 - e^(-1) ≈ 0.6321
    let val: f64 = lua
        .load("return math.gamma_cdf(1.0, 1.0, 1.0)")
        .eval()
        .unwrap();
    let expected = 1.0 - (-1.0_f64).exp();
    assert!(
        (val - expected).abs() < 1e-6,
        "gamma_cdf(1, shape=1, scale=1) should be ~0.6321, got {val}"
    );
}

#[test]
fn gamma_cdf_positive() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.gamma_cdf(1.0, 2.0, 1.0)")
        .eval()
        .unwrap();
    assert!(val > 0.0 && val < 1.0);
}

#[test]
fn poisson_cdf_basic() {
    let lua = setup();
    // P(X <= 0) for Poisson(1) = e^(-1) ≈ 0.3679
    let val: f64 = lua.load("return math.poisson_cdf(0, 1.0)").eval().unwrap();
    assert!((val - (-1.0_f64).exp()).abs() < 1e-6);
}

#[test]
fn beta_ppf_roundtrip() {
    let lua = setup();
    let code = r#"
        local p = 0.7
        local x = math.beta_ppf(p, 2.0, 5.0)
        local p2 = math.beta_cdf(x, 2.0, 5.0)
        local diff = p - p2
        if diff < 0 then diff = -diff end
        return diff < 1e-6
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "beta_ppf/beta_cdf roundtrip");
}

#[test]
fn beta_mean_known() {
    let lua = setup();
    let val: f64 = lua.load("return math.beta_mean(2, 5)").eval().unwrap();
    assert!((val - 2.0 / 7.0).abs() < 1e-10);
}

#[test]
fn beta_variance_known() {
    let lua = setup();
    let val: f64 = lua.load("return math.beta_variance(2, 5)").eval().unwrap();
    // (2*5) / (7^2 * 8) = 10/392 ≈ 0.02551
    let expected = (2.0 * 5.0) / (49.0 * 8.0);
    assert!((val - expected).abs() < 1e-10);
}

// ── v0.2 Statistics ─────────────────────────────────────

#[test]
fn covariance_perfect_positive() {
    let lua = setup();
    let code = "return math.covariance({1,2,3,4,5}, {2,4,6,8,10})";
    let val: f64 = lua.load(code).eval().unwrap();
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn correlation_perfect() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.correlation({1,2,3,4,5}, {2,4,6,8,10})")
        .eval()
        .unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn correlation_negative() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.correlation({1,2,3,4,5}, {10,8,6,4,2})")
        .eval()
        .unwrap();
    assert!((val - (-1.0)).abs() < 1e-10);
}

#[test]
fn histogram_basic() {
    let lua = setup();
    let code = r#"
        local h = math.histogram({1,2,3,4,5,6,7,8,9,10}, 5)
        local total = 0
        for _, c in ipairs(h.counts) do total = total + c end
        return total == 10 and #h.counts == 5 and #h.edges == 6
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "histogram: 10 values, 5 bins, 6 edges expected");
}

#[test]
fn wilson_ci_basic() {
    let lua = setup();
    let code = r#"
        local ci = math.wilson_ci(80, 100, 0.95)
        return ci.lower > 0.7 and ci.upper < 0.9 and ci.lower < ci.center and ci.center < ci.upper
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "Wilson CI for 80/100 at 95% should be ~(0.71, 0.87)");
}

#[test]
fn wilson_ci_contains_the_estimate_at_the_endpoints_via_lua() {
    let lua = setup();
    // In exact arithmetic the interval touches p̂ at 0 and 1; in floating point
    // the residue lands a few ulp off, which without the clamp hands the caller
    // a bound excluding its own estimate. n=10 and n=2000 are two of the
    // lengths where the upper end fell short.
    let code = r#"
        for _, n in ipairs({1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000}) do
            local hi = math.wilson_ci(n, n, 0.95)
            if hi.upper < 1.0 then return "upper < 1 at n=" .. n end
            if hi.lower > 1.0 then return "lower > p_hat at n=" .. n end

            local lo = math.wilson_ci(0, n, 0.95)
            if lo.lower > 0.0 then return "lower > 0 at n=" .. n end
            if lo.upper < 0.0 then return "upper < p_hat at n=" .. n end
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn wilson_ci_still_brackets_interior_estimates_via_lua() {
    let lua = setup();
    // The clamp must not collapse an interior interval onto p̂.
    let code = r#"
        for k = 1, 99 do
            local ci = math.wilson_ci(k, 100, 0.95)
            local p = k / 100
            if not (ci.lower <= p and p <= ci.upper) then
                return "does not contain p_hat at k=" .. k
            end
            if not (ci.lower < ci.upper) then
                return "collapsed to a point at k=" .. k
            end
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn log_normalize_basic() {
    let lua = setup();
    let code = r#"
        local result = math.log_normalize({1, 10, 100})
        return result[1] < result[2] and result[2] < result[3]
            and result[3] > 99.9 and result[3] <= 100
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "log_normalize should preserve ordering, max near 100");
}

#[test]
fn log_normalize_negative_errors() {
    let lua = setup();
    let result: LuaResult<LuaTable> = lua.load("return math.log_normalize({1, -2, 3})").eval();
    assert!(result.is_err(), "negative values should be rejected");
}

// ── Missing function tests ─────────────────────────────

#[test]
fn ln_beta_known() {
    let lua = setup();
    // ln_beta(1,1) = ln(B(1,1)) = ln(1) = 0
    let val: f64 = lua.load("return math.ln_beta(1, 1)").eval().unwrap();
    assert!(val.abs() < 1e-10);
}

#[test]
fn regularized_incomplete_gamma_known() {
    let lua = setup();
    // regularized_incomplete_gamma(1, 1) = 1 - e^(-1) ≈ 0.6321
    let val: f64 = lua
        .load("return math.regularized_incomplete_gamma(1, 1)")
        .eval()
        .unwrap();
    let expected = 1.0 - (-1.0_f64).exp();
    assert!(
        (val - expected).abs() < 1e-6,
        "reg_inc_gamma(1,1) should be ~0.6321, got {val}"
    );
}

#[test]
fn ln_factorial_known() {
    let lua = setup();
    // ln_factorial(5) = ln(120) ≈ 4.7875
    let val: f64 = lua.load("return math.ln_factorial(5)").eval().unwrap();
    assert!((val - 120.0_f64.ln()).abs() < 1e-10);
    // ln_factorial(0) = ln(1) = 0
    let val: f64 = lua.load("return math.ln_factorial(0)").eval().unwrap();
    assert!(val.abs() < 1e-10);
}

#[test]
fn normal_inverse_cdf_known() {
    let lua = setup();
    // normal_inverse_cdf(0.5, 0, 1) = 0
    let val: f64 = lua
        .load("return math.normal_inverse_cdf(0.5, 0, 1)")
        .eval()
        .unwrap();
    assert!(val.abs() < 1e-10);
    // normal_inverse_cdf(0.5, 10, 2) = 10 (median = mean)
    let val: f64 = lua
        .load("return math.normal_inverse_cdf(0.5, 10, 2)")
        .eval()
        .unwrap();
    assert!((val - 10.0).abs() < 1e-10);
}

// ── v0.3 Ranking ────────────────────────────────────────

#[test]
fn rank_basic() {
    let lua = setup();
    let code = r#"
        local r = math.rank({3, 1, 2})
        return r[1] == 3 and r[2] == 1 and r[3] == 2
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "rank should assign 1-based ranks in original order");
}

#[test]
fn rank_ties() {
    let lua = setup();
    let code = r#"
        local r = math.rank({3, 1, 3})
        return r[1] == 2.5 and r[2] == 1 and r[3] == 2.5
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "tied values should get average rank");
}

#[test]
fn spearman_perfect_via_lua() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.spearman_correlation({1,2,3,4,5}, {2,4,6,8,10})")
        .eval()
        .unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn kendall_tau_via_lua() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.kendall_tau({1,2,3}, {3,2,1})")
        .eval()
        .unwrap();
    assert!((val - (-1.0)).abs() < 1e-10);
}

#[test]
fn ndcg_perfect_via_lua() {
    let lua = setup();
    let val: f64 = lua.load("return math.ndcg({3, 2, 1}, 3)").eval().unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn mrr_via_lua() {
    let lua = setup();
    let val: f64 = lua.load("return math.mrr({1, 2, 5})").eval().unwrap();
    let expected = (1.0 + 0.5 + 0.2) / 3.0;
    assert!((val - expected).abs() < 1e-10);
}

// ── v0.3 Hypothesis Testing ─────────────────────────────

#[test]
fn welch_t_test_via_lua() {
    let lua = setup();
    let code = r#"
        local r = math.welch_t_test({1,2,3,4,5}, {100,200,300,400,500})
        return r.p_value < 0.05 and r.df > 0
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "very different groups should yield significant p-value");
}

#[test]
fn mann_whitney_u_via_lua() {
    let lua = setup();
    let code = r#"
        local r = math.mann_whitney_u({1,2,3}, {1,2,3})
        return r.u_stat >= 0 and r.p_value >= 0 and r.p_value <= 1
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "mann_whitney_u should return valid results");
}

#[test]
fn chi_squared_test_via_lua() {
    let lua = setup();
    let code = r#"
        local r = math.chi_squared_test({25,25,25,25}, {25,25,25,25})
        return r.chi2_stat < 1e-10 and r.df == 3 and r.p_value > 0.99
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "perfect fit should yield chi2≈0, p≈1");
}

#[test]
fn ks_test_via_lua() {
    let lua = setup();
    let code = r#"
        local xs = {}; local ys = {}
        for i = 1, 50 do xs[i] = i; ys[i] = i + 100 end
        local r = math.ks_test(xs, ys)
        return r.d_stat > 0.9 and r.p_value >= 0 and r.p_value <= 1
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "completely separated samples should have d≈1");
}

// ── v0.3 Information Theory ─────────────────────────────

#[test]
fn entropy_via_lua() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.entropy({0.25, 0.25, 0.25, 0.25})")
        .eval()
        .unwrap();
    let expected = 4.0_f64.ln();
    assert!((val - expected).abs() < 1e-10);
}

#[test]
fn kl_divergence_via_lua() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.kl_divergence({0.5, 0.5}, {0.5, 0.5})")
        .eval()
        .unwrap();
    assert!(val.abs() < 1e-10, "KL(p||p) should be 0");
}

#[test]
fn js_divergence_symmetric_via_lua() {
    let lua = setup();
    let code = r#"
        local a = math.js_divergence({0.9, 0.1}, {0.1, 0.9})
        local b = math.js_divergence({0.1, 0.9}, {0.9, 0.1})
        local diff = a - b
        if diff < 0 then diff = -diff end
        return diff < 1e-10
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "JS divergence should be symmetric");
}

#[test]
fn cross_entropy_via_lua() {
    let lua = setup();
    let code = r#"
        local p = {0.25, 0.25, 0.25, 0.25}
        local ce = math.cross_entropy(p, p)
        local h = math.entropy(p)
        local diff = ce - h
        if diff < 0 then diff = -diff end
        return diff < 1e-10
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "H(p,p) should equal H(p)");
}

#[test]
fn tvd_via_lua() {
    let lua = setup();
    let code = r#"
        local same = math.tvd({0.25, 0.25, 0.5}, {0.25, 0.25, 0.5})
        local disjoint = math.tvd({1.0, 0.0}, {0.0, 1.0})
        return same < 1e-10 and disjoint > 1.0 - 1e-10
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(
        ok,
        "TVD should be 0 on identical and 1 on disjoint supports"
    );
}

#[test]
fn off_support_divergence_is_huge_via_lua() {
    let lua = setup();
    // `setup` replaces the stdlib `math` table, so `math.huge` is not in scope.
    let code = r#"
        local inf = 1 / 0
        local kl = math.kl_divergence({0.5, 0.5}, {1.0, 0.0})
        local ce = math.cross_entropy({0.5, 0.5}, {1.0, 0.0})
        return kl == inf and ce == inf
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "a support gap should read as infinity, not raise");
}

#[test]
fn long_distribution_tolerates_f32_drift_via_lua() {
    let lua = setup();
    // 1.2e-4 is the drift an f32-normalized 50257-entry softmax carries; the old
    // fixed 1e-6 rejected it. Added by hand here — Lua numbers are f64, so the
    // f32 rounding that produces it cannot be reproduced from this side.
    let code = r#"
        local n = 50257
        local p = {}
        local each = 1.0 / n
        for i = 1, n do p[i] = each end
        p[1] = p[1] + 1.2e-4
        return math.entropy(p)
    "#;
    let h: f64 = lua.load(code).eval().unwrap();
    assert!(
        h > 0.0,
        "a drifted long distribution should still be scored"
    );
}

#[test]
fn error_names_the_offending_element_via_lua() {
    let lua = setup();
    let code = r#"
        local ok, err = pcall(math.entropy, {0.5, -0.1, 0.6})
        return tostring(err)
    "#;
    let err: String = lua.load(code).eval().unwrap();
    // 1-based, matching the Lua table: -0.1 is probs[2], not probs[1].
    assert!(
        err.contains("probs[2] is negative"),
        "error should name the element, got: {err}"
    );
}

// ── v0.5 Resampling / adjustment / effect size ──────────

#[test]
fn cluster_bootstrap_mean_via_lua() {
    let lua = setup();
    let code = r#"
        -- one observation per cluster; the point estimate is their mean
        local by_cluster = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}
        local ci = math.cluster_bootstrap_mean(by_cluster, 2000, 42)
        local ok = ci.point > 4.49 and ci.point < 4.51
            and ci.lower <= ci.point and ci.point <= ci.upper
            and ci.lower < ci.upper
            and ci.clusters == 8
            and ci.draws_used == 2000
            and ci.undefined_draws == 0
        if not ok then
            return string.format("point=%f lower=%f upper=%f used=%d undef=%d clusters=%d",
                ci.point, ci.lower, ci.upper, ci.draws_used, ci.undefined_draws, ci.clusters)
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn cluster_bootstrap_is_reproducible_via_lua() {
    let lua = setup();
    let code = r#"
        local c = {{3}, {1}, {4}, {1}, {5}, {9}, {2}, {6}}
        local a = math.cluster_bootstrap_mean(c, 500, 7)
        local b = math.cluster_bootstrap_mean(c, 500, 7)
        local d = math.cluster_bootstrap_mean(c, 500, 8)
        return a.lower == b.lower and a.upper == b.upper
            and not (a.lower == d.lower and a.upper == d.upper)
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "same seed must reproduce, a different seed must not");
}

#[test]
fn cluster_bootstrap_diff_pairs_within_a_draw_via_lua() {
    let lua = setup();
    // b tracks a with a constant offset, so the difference has no spread —
    // which only holds because both sides are measured on the same draw.
    let code = r#"
        local a = {{1}, {5}, {9}, {13}, {17}}
        local b = {{3}, {7}, {11}, {15}, {19}}
        local ci = math.cluster_bootstrap_diff(a, b, 2000, 3)
        local width = ci.upper - ci.lower
        if width < 0 then width = -width end
        local point_off = ci.point + 2.0
        if point_off < 0 then point_off = -point_off end
        return width < 1e-9 and point_off < 1e-9
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(
        ok,
        "a constant offset must give a zero-width interval at -2"
    );
}

#[test]
fn cluster_bootstrap_ratio_via_lua() {
    let lua = setup();
    let code = r#"
        -- ratio of sums: (1+3) / (1+9) = 0.4, not the mean of 1/1 and 3/9
        local num = {{1}, {3}}
        local den = {{1}, {9}}
        local ci = math.cluster_bootstrap_ratio(num, den, 500, 2)
        local off = ci.point - 0.4
        if off < 0 then off = -off end
        return off < 1e-9
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "ratio must be of the summed sides");
}

#[test]
fn cluster_bootstrap_reports_undefined_draws_via_lua() {
    let lua = setup();
    // Cluster 2 is empty; a draw naming only it has no observations to average.
    let code = r#"
        local c = {{4}, {}, {6}}
        local ci = math.cluster_bootstrap_mean(c, 2000, 5)
        return ci.clusters == 3
            and ci.undefined_draws > 0
            and ci.draws_used + ci.undefined_draws == 2000
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "empty clusters must surface as undefined draws");
}

#[test]
fn cluster_bootstrap_mismatched_sides_error_via_lua() {
    let lua = setup();
    let code = r#"
        local ok, err = pcall(math.cluster_bootstrap_diff, {{1}, {2}}, {{1}}, 100, 1)
        return tostring(err)
    "#;
    let err: String = lua.load(code).eval().unwrap();
    assert!(
        err.contains("same clusters") && err.contains("a_by_cluster"),
        "error should name the mismatch and the parameters, got: {err}"
    );
}

#[test]
fn cluster_bootstrap_refuses_a_degenerate_or_implausible_call_via_lua() {
    let lua = setup();
    let code = r#"
        -- one cluster: every draw is the same draw
        local _, single = pcall(math.cluster_bootstrap_mean, {{1, 2, 3}}, 2000, 42)
        -- draws and seed swapped: a timestamp as a draw count
        local _, swapped = pcall(math.cluster_bootstrap_mean, {{1}, {2}}, 1755000000, 42)
        -- a non-numeric observation, which must name its position
        local _, badval = pcall(math.cluster_bootstrap_mean, {{1}, {2, "x"}}, 100, 1)
        return tostring(single) .. " | " .. tostring(swapped) .. " | " .. tostring(badval)
    "#;
    let msg: String = lua.load(code).eval().unwrap();
    assert!(msg.contains("at least 2 clusters"), "got: {msg}");
    assert!(msg.contains("argument order"), "got: {msg}");
    assert!(msg.contains("by_cluster[2][2]"), "got: {msg}");
}

#[test]
fn cluster_bootstrap_ratio_guards_a_sign_flipping_denominator_via_lua() {
    let lua = setup();
    // Whole-sample denominator 3 + 3 - 5 = 1 > 0. Draws naming the negative
    // cluster twice go negative, and those ratios are a different quantity.
    let code = r#"
        local num = {{1}, {1}, {1}}
        local den = {{3}, {3}, {-5}}
        local ci = math.cluster_bootstrap_ratio(num, den, 2000, 12)
        return ci.undefined_draws > 0 and ci.lower > 0
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(
        ok,
        "sign-flipping draws must be undefined, keeping the interval positive"
    );
}

#[test]
fn multiple_comparison_adjustments_via_lua() {
    let lua = setup();
    let code = r#"
        local raw = {0.01, 0.04, 0.03}
        local holm = math.holm(raw)
        local bh = math.benjamini_hochberg(raw)
        for i = 1, #raw do
            if holm[i] < raw[i] then return "holm lowered p[" .. i .. "]" end
            if bh[i] > holm[i] then return "bh exceeded holm at " .. i end
            if holm[i] > 1.0 or bh[i] > 1.0 then return "above 1 at " .. i end
        end
        -- worked example: holm(0.01) = 3 * 0.01
        local off = holm[1] - 0.03
        if off < 0 then off = -off end
        if off > 1e-12 then return "holm[1]=" .. holm[1] end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn effect_sizes_via_lua() {
    let lua = setup();
    let code = r#"
        local low = {1, 2, 3}
        local high = {4, 5, 6}
        local delta = math.cliffs_delta(high, low)
        local d = math.cohens_d(high, low)
        local same = math.cliffs_delta(low, low)
        if delta < 0.999 then return "cliffs_delta=" .. delta end
        if same > 1e-12 or same < -1e-12 then return "self delta=" .. same end
        if d <= 0 then return "cohens_d should be positive, got " .. d end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

// ── v0.6 Permutation / paired bootstrap / mutual information ──

#[test]
fn bootstrap_mean_via_lua() {
    let lua = setup();
    let code = r#"
        local ci = math.bootstrap_mean({1, 2, 3, 4, 5, 6, 7, 8}, 2000, 1)
        local ok = ci.point > 4.49 and ci.point < 4.51
            and ci.lower < ci.point and ci.point < ci.upper
            and ci.observations == 8
            and ci.draws_used == 2000
        if not ok then
            return string.format("point=%f lower=%f upper=%f obs=%d",
                ci.point, ci.lower, ci.upper, ci.observations)
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn paired_bootstrap_diff_cancels_shared_variation_via_lua() {
    let lua = setup();
    // Both series carry a large per-item effect and differ by a constant 2.
    // Pairing removes what they share; the unpaired side stays wide.
    let code = r#"
        local xs = {10, 50, 90, 130, 170}
        local ys = {8, 48, 88, 128, 168}
        local paired = math.paired_bootstrap_diff(xs, ys, 2000, 4)
        local side = math.bootstrap_mean(xs, 2000, 4)
        local width = paired.upper - paired.lower
        local off = paired.point - 2.0
        if off < 0 then off = -off end
        return width < 1e-9 and off < 1e-9 and (side.upper - side.lower) > 20
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "pairing must collapse a constant difference");
}

#[test]
fn paired_bootstrap_diff_reports_a_real_width_via_lua() {
    let lua = setup();
    // The difference varies per pair, so the interval has to have width — the
    // constant-offset case collapses to zero and would hide an interval that
    // never widens.
    let code = r#"
        local xs = {10, 12, 14, 16, 18, 20, 22, 24}
        local ys = { 9, 13, 12, 18, 15, 22, 19, 26}
        local ci = math.paired_bootstrap_diff(xs, ys, 2000, 11)
        local width = ci.upper - ci.lower
        -- mean difference is +0.5, and the per-pair differences straddle zero
        if width < 0.5 then return "interval too narrow: " .. width end
        if ci.lower > ci.point or ci.point > ci.upper then return "does not bracket the point" end
        if ci.observations ~= 8 then return "observations=" .. ci.observations end
        -- the sign is genuinely undecided here
        if not (ci.lower < 0 and ci.upper > 0) then
            return string.format("expected to straddle zero: [%f, %f]", ci.lower, ci.upper)
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn bootstrap_mean_matches_the_cluster_path_via_lua() {
    let lua = setup();
    // The Lua-facing pair, not just the internals: one observation per cluster
    // is the same resampling, so the same seed must give the same interval.
    let code = r#"
        local values = {3, 1, 4, 1, 5, 9}
        local by_cluster = {{3}, {1}, {4}, {1}, {5}, {9}}
        local flat = math.bootstrap_mean(values, 500, 21)
        local clustered = math.cluster_bootstrap_mean(by_cluster, 500, 21)
        return flat.lower == clustered.lower
            and flat.upper == clustered.upper
            and flat.point == clustered.point
            and flat.observations == clustered.clusters
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "the flat and cluster paths must agree at the same seed");
}

#[test]
fn resampling_errors_name_the_right_signature_via_lua() {
    let lua = setup();
    // A caller who swapped draws and seed gets told which signature to check,
    // and it has to be the one they actually called.
    let code = r#"
        local _, flat = pcall(math.bootstrap_mean, {1, 2, 3}, 1e10, 1)
        local _, paired = pcall(math.paired_bootstrap_diff, {1, 2}, {3, 4}, 1e10, 1)
        local _, clustered = pcall(math.cluster_bootstrap_mean, {{1}, {2}}, 1e10, 1)
        local _, single = pcall(math.bootstrap_mean, {1}, 100, 1)
        return tostring(flat) .. " | " .. tostring(paired) .. " | "
            .. tostring(clustered) .. " | " .. tostring(single)
    "#;
    let msg: String = lua.load(code).eval().unwrap();
    assert!(msg.contains("(values, draws, seed)"), "got: {msg}");
    assert!(msg.contains("(xs, ys, draws, seed)"), "got: {msg}");
    assert!(msg.contains("(by_cluster, draws, seed)"), "got: {msg}");
    assert!(
        msg.contains("at least 2 observations"),
        "a flat series must not be told about clusters: {msg}"
    );
}

#[test]
fn paired_bootstrap_diff_requires_equal_lengths_via_lua() {
    let lua = setup();
    let code = r#"
        local _, err = pcall(math.paired_bootstrap_diff, {1, 2, 3}, {1, 2}, 100, 1)
        return tostring(err)
    "#;
    let err: String = lua.load(code).eval().unwrap();
    assert!(err.contains("paired element by element"), "got: {err}");
}

#[test]
fn permutation_test_via_lua() {
    let lua = setup();
    let code = r#"
        local low = {1, 2, 3, 4, 5}
        local high = {11, 12, 13, 14, 15}
        local sep = math.permutation_test(high, low, 2000, 1)
        if sep.p_value >= 0.01 then return "separated p=" .. sep.p_value end
        if sep.observed < 9.99 or sep.observed > 10.01 then return "observed=" .. sep.observed end
        if sep.p_value <= 0 then return "p must never reach zero" end

        local a = {1, 3, 5, 7, 9}
        local b = {2, 4, 6, 8, 10}
        local mixed = math.permutation_test(a, b, 2000, 2)
        if mixed.p_value <= 0.2 then return "interleaved p=" .. mixed.p_value end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn permutation_test_alternative_option_via_lua() {
    let lua = setup();
    let code = r#"
        local low = {1, 2, 3, 4, 5}
        local high = {6, 7, 8, 9, 10}
        local g = math.permutation_test(high, low, 2000, 4, {alternative = "greater"})
        local l = math.permutation_test(high, low, 2000, 4, {alternative = "less"})
        local _, err = pcall(math.permutation_test, high, low, 100, 1, {alternative = "sideways"})
        return g.p_value < 0.05 and l.p_value > 0.95
            and string.find(tostring(err), "two_sided", 1, true) ~= nil
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(
        ok,
        "one-sided alternatives and the rejection of an unknown one"
    );
}

#[test]
fn permutation_test_rejects_a_misspelled_option_via_lua() {
    let lua = setup();
    // A typo'd key used to leave `alternative` unset, so a caller asking for a
    // one-sided test silently received a two-sided p-value.
    let code = r#"
        local low = {1, 2, 3, 4, 5}
        local high = {6, 7, 8, 9, 10}
        local _, typo = pcall(math.permutation_test, high, low, 100, 1,
            {alternatve = "greater"})
        local _, wrong_type = pcall(math.permutation_test, high, low, 100, 1,
            {alternative = 42})
        return tostring(typo) .. " | " .. tostring(wrong_type)
    "#;
    let msg: String = lua.load(code).eval().unwrap();
    assert!(
        msg.contains("unknown option"),
        "typo must be refused: {msg}"
    );
    assert!(msg.contains("alternatve"), "the typo must be quoted: {msg}");
    assert!(
        msg.matches("permutation_test").count() >= 2,
        "both calls must fail: {msg}"
    );
}

#[test]
fn permutation_test_counts_floating_point_ties_via_lua() {
    let lua = setup();
    // Identical multisets: every arrangement ties, so p is exactly 1. Summing
    // 0.1 + 0.2 + 0.3 in different orders differs by ulps, and an exact
    // comparison dropped those ties and reported p ≈ 0.80.
    let code = r#"
        local r = math.permutation_test({0.1, 0.2, 0.3}, {0.2, 0.3, 0.1}, 2000, 1)
        if r.extreme_draws ~= r.draws then
            return "dropped ties: " .. r.extreme_draws .. "/" .. r.draws
        end
        local off = r.p_value - 1.0
        if off < 0 then off = -off end
        if off > 1e-12 then return "p=" .. r.p_value end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

#[test]
fn mutual_information_via_lua() {
    let lua = setup();
    let code = r#"
        -- independent: p(x,y) = p(x)q(y)
        local indep = math.mutual_information({{0.12, 0.18}, {0.28, 0.42}})
        if indep > 1e-9 then return "independent mi=" .. indep end

        -- a perfect copy: I(X;Y) = H(X)
        local dep = math.mutual_information({{0.25, 0.0}, {0.0, 0.75}})
        local hx = math.entropy({0.25, 0.75})
        local off = dep - hx
        if off < 0 then off = -off end
        if off > 1e-9 then return "dep=" .. dep .. " hx=" .. hx end

        -- a ragged matrix names the offending row, 1-based
        local _, err = pcall(math.mutual_information, {{0.5, 0.5}, {0.0}})
        if string.find(tostring(err), "joint[2] has 1 columns", 1, true) == nil then
            return "ragged: " .. tostring(err)
        end
        return "ok"
    "#;
    let res: String = lua.load(code).eval().unwrap();
    assert_eq!(res, "ok");
}

// ── v0.3 Special (logsumexp, logit, expit) ──────────────

#[test]
fn logsumexp_via_lua() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.logsumexp({1000, 1001, 1002})")
        .eval()
        .unwrap();
    assert!(
        val > 1001.0 && val < 1003.0,
        "logsumexp should be numerically stable"
    );
}

#[test]
fn logit_expit_roundtrip() {
    let lua = setup();
    let code = r#"
        local p = 0.73
        local x = math.logit(p)
        local p2 = math.expit(x)
        local diff = p - p2
        if diff < 0 then diff = -diff end
        return diff < 1e-10
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "expit(logit(p)) should equal p");
}

// ── v0.3 Stats (moving_average, ewma, autocorrelation, permutations, shuffle, sample_with_replacement) ──

#[test]
fn moving_average_via_lua() {
    let lua = setup();
    let code = r#"
        local ma = math.moving_average({1,2,3,4,5}, 3)
        return #ma == 3 and ma[1] == 2 and ma[2] == 3 and ma[3] == 4
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "SMA(3) of [1,2,3,4,5] should be [2,3,4]");
}

#[test]
fn ewma_via_lua() {
    let lua = setup();
    let code = r#"
        local e = math.ewma({10, 20, 30}, 1.0)
        return e[1] == 10 and e[2] == 20 and e[3] == 30
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "ewma with alpha=1 should equal raw values");
}

#[test]
fn autocorrelation_lag0() {
    let lua = setup();
    let val: f64 = lua
        .load("return math.autocorrelation({1,2,3,4,5}, 0)")
        .eval()
        .unwrap();
    assert!(
        (val - 1.0).abs() < 1e-10,
        "autocorrelation at lag 0 should be 1"
    );
}

#[test]
fn permutations_via_lua() {
    let lua = setup();
    let code = r#"
        local p = math.permutations(3)
        return #p == 6
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "permutations(3) should yield 3! = 6 permutations");
}

#[test]
fn shuffle_via_lua() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        local s = math.shuffle(rng, {1,2,3,4,5})
        if #s ~= 5 then return false end
        local sum = 0
        for _, v in ipairs(s) do sum = sum + v end
        return sum == 15
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(ok, "shuffle should preserve all elements");
}

#[test]
fn sample_with_replacement_via_lua() {
    let lua = setup();
    let code = r#"
        local rng = math.rng_create(42)
        local s = math.sample_with_replacement(rng, {10, 20, 30}, 100)
        if #s ~= 100 then return false end
        for _, v in ipairs(s) do
            if v ~= 10 and v ~= 20 and v ~= 30 then return false end
        end
        return true
    "#;
    let ok: bool = lua.load(code).eval().unwrap();
    assert!(
        ok,
        "sample_with_replacement should only return elements from input"
    );
}
