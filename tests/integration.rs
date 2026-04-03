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
    assert!(ok);
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
    assert!(ok);
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
    assert!(ok);
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
    assert!(ok);
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
    assert!(ok);
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
    assert!(ok);
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
    assert!(ok);
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
