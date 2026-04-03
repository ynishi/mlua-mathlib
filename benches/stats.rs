use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mlua::prelude::*;

fn setup() -> Lua {
    let lua = Lua::new();
    let math = mlua_mathlib::module(&lua).unwrap();
    lua.globals().set("math", math).unwrap();
    lua
}

fn bench_mean(c: &mut Criterion) {
    let lua = setup();
    c.bench_function("mean_100", |b| {
        b.iter(|| {
            let _: f64 = lua
                .load(black_box(
                    "local t = {} for i=1,100 do t[i]=i end return math.mean(t)",
                ))
                .eval()
                .unwrap();
        });
    });
}

fn bench_variance(c: &mut Criterion) {
    let lua = setup();
    c.bench_function("variance_100", |b| {
        b.iter(|| {
            let _: f64 = lua
                .load(black_box(
                    "local t = {} for i=1,100 do t[i]=i end return math.variance(t)",
                ))
                .eval()
                .unwrap();
        });
    });
}

fn bench_normal_sample(c: &mut Criterion) {
    let lua = setup();
    lua.load("__bench_rng = math.rng_create(42)")
        .exec()
        .unwrap();
    c.bench_function("normal_sample", |b| {
        b.iter(|| {
            let _: f64 = lua
                .load(black_box(
                    "return math.normal_sample(__bench_rng, 0.0, 1.0)",
                ))
                .eval()
                .unwrap();
        });
    });
}

fn bench_softmax(c: &mut Criterion) {
    let lua = setup();
    c.bench_function("softmax_10", |b| {
        b.iter(|| {
            let _: LuaTable = lua
                .load(black_box("return math.softmax({1,2,3,4,5,6,7,8,9,10})"))
                .eval()
                .unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_mean,
    bench_variance,
    bench_normal_sample,
    bench_softmax
);
criterion_main!(benches);
