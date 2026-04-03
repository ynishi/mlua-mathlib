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
