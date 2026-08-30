use mlua::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::stats::percentile_impl;

/// Confidence level used when the caller does not name one.
const DEFAULT_CONFIDENCE: f64 = 0.95;

/// Smallest cluster count a resample can say anything with.
///
/// At one cluster every draw is the same draw, so the interval collapses onto
/// the point estimate and reads as infinite precision. Two is the smallest
/// count that produces distinct draws; it is still far too few to trust — the
/// usual guidance is dozens — but past that the caller has `clusters` and
/// `draws_used` to judge with, and the library should not pick the threshold.
const MIN_CLUSTERS: usize = 2;

/// Ceiling on `draws`, so a swapped argument fails loudly.
///
/// `draws` and `seed` are adjacent bare numbers and Lua has no named
/// arguments; passing a timestamp as `draws` would otherwise reserve tens of
/// gigabytes before doing anything. Well past any useful resample count.
const MAX_DRAWS: usize = 10_000_000;

/// A percentile interval and the accounting of the draws behind it.
#[derive(Debug)]
struct Interval {
    /// The statistic on the sample as walked — every cluster once.
    point: f64,
    lower: f64,
    upper: f64,
    /// Draws the statistic was defined on. The interval rests on these.
    draws_used: usize,
    /// Draws the statistic was undefined on, and which the interval therefore
    /// does not rest on. A large share means the statistic sits on few enough
    /// clusters that resampling destroys it.
    undefined_draws: usize,
    clusters: usize,
}

/// A sum and a count, so a statistic over a draw can be accumulated by
/// combining per-cluster subtotals instead of walking observations again.
#[derive(Clone, Copy, Default)]
struct Tally {
    sum: f64,
    n: usize,
}

impl Tally {
    /// `None` on an empty tally: a mean over no observations is undefined, and
    /// that is a fact about the draw the caller should see rather than a zero
    /// to average in.
    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.sum / self.n as f64)
        }
    }
}

/// Reduce each cluster to a subtotal, once.
///
/// A draw then costs `O(clusters)` rather than `O(observations)`, which is
/// what keeps a 2000-draw resample over a long sample affordable.
fn cluster_tallies(clusters: &[Vec<f64>]) -> Vec<Tally> {
    clusters
        .iter()
        .map(|c| Tally {
            sum: c.iter().sum(),
            n: c.len(),
        })
        .collect()
}

/// Combine the subtotals of the clusters a draw names, with repetition.
fn tally_over(tallies: &[Tally], draw: &[usize]) -> Tally {
    let mut total = Tally::default();
    for &c in draw {
        total.sum += tallies[c].sum;
        total.n += tallies[c].n;
    }
    total
}

/// Resample `clusters` clusters `draws` times and bound `stat`.
///
/// `stat` receives one draw — cluster indices with repetition — and returns its
/// value on exactly those clusters, or `None` where the draw leaves it
/// undefined. Every term of a difference or ratio must be computed from that
/// same list: two separately bootstrapped quantities carry no joint
/// distribution, so their intervals cannot be combined after the fact.
///
/// # What the returned interval is conditioned on
///
/// Draws the statistic is undefined on are dropped, and the percentiles are
/// taken over what remains — there is no value to put in their place. The
/// interval is therefore over the distribution *conditioned on the statistic
/// being defined*, and coverage arguments for the unconditional bootstrap do
/// not carry over unchanged. `undefined_draws` is what makes that visible.
///
/// # Skew
///
/// A percentile interval need not contain the point estimate. For a mean it
/// effectively always does; for a skewed statistic such as a ratio it can sit
/// entirely to one side. That is the method reporting the shape of the
/// resample distribution, not a defect to clamp away.
fn cluster_bootstrap(
    clusters: usize,
    draws: usize,
    seed: u64,
    confidence: f64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<Interval, String> {
    if clusters < MIN_CLUSTERS {
        return Err(format!(
            "needs at least {MIN_CLUSTERS} clusters to resample, got {clusters}; a single cluster \
             gives the same draw every time and would report a zero-width interval"
        ));
    }
    if draws == 0 {
        return Err("needs at least one draw".into());
    }
    if draws > MAX_DRAWS {
        return Err(format!(
            "draws is capped at {MAX_DRAWS}, got {draws}; check the argument order — it is \
             (by_cluster, draws, seed)"
        ));
    }
    if !(0.0..=1.0).contains(&confidence) {
        return Err(format!("confidence must be in [0, 1], got {confidence}"));
    }

    let whole: Vec<usize> = (0..clusters).collect();
    let point = stat(&whole).filter(|v| v.is_finite()).ok_or(
        "the statistic is undefined on the sample as walked, so there is nothing to bound",
    )?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut draw = vec![0usize; clusters];
    let mut values: Vec<f64> = Vec::new();
    let mut undefined_draws = 0usize;
    for _ in 0..draws {
        for slot in draw.iter_mut() {
            *slot = rng.random_range(0..clusters);
        }
        match stat(&draw).filter(|v| v.is_finite()) {
            Some(v) => values.push(v),
            None => undefined_draws += 1,
        }
    }
    if values.is_empty() {
        return Err(format!(
            "all {draws} draws were undefined even though the statistic exists on the whole \
             sample; it rests on too few clusters to resample"
        ));
    }

    values.sort_by(f64::total_cmp);
    let tail = (1.0 - confidence) / 2.0;
    Ok(Interval {
        point,
        lower: percentile_impl(&values, tail * 100.0),
        upper: percentile_impl(&values, (1.0 - tail) * 100.0),
        draws_used: values.len(),
        undefined_draws,
        clusters,
    })
}

/// Read a Lua sequence of sequences as per-cluster observations.
///
/// An empty cluster is allowed — a game in which the measured position never
/// arose is a real cluster carrying no observations, and dropping it would
/// change what the resampling is over.
fn table_to_clusters(table: &LuaTable, side: &str) -> LuaResult<Vec<Vec<f64>>> {
    let n = table.raw_len();
    if n == 0 {
        return Err(LuaError::runtime(format!("{side} holds no clusters")));
    }
    let mut clusters = Vec::with_capacity(n);
    for i in 1..=n {
        let inner: LuaTable = table.raw_get(i).map_err(|_| {
            LuaError::runtime(format!("{side}[{i}] is not a table of observations"))
        })?;
        let m = inner.raw_len();
        let mut values = Vec::with_capacity(m);
        for j in 1..=m {
            // Name the position on a conversion failure too, not only on the
            // finiteness check below.
            let v: f64 = inner
                .raw_get(j)
                .map_err(|_| LuaError::runtime(format!("{side}[{i}][{j}] is not a number")))?;
            if !v.is_finite() {
                return Err(LuaError::runtime(format!(
                    "{side}[{i}][{j}] is {v} (NaN/Infinity not allowed)"
                )));
            }
            values.push(v);
        }
        clusters.push(values);
    }
    Ok(clusters)
}

/// Both sides of a paired statistic must be indexed by the same clusters.
///
/// Only the count can be checked here. That `a[i]` and `b[i]` measure the
/// *same* cluster is a contract the caller keeps: one draw is applied to both
/// sides, so passing two independent groups of equal length would manufacture
/// a correlation that is not there and report an interval far too narrow.
fn assert_same_clusters(a: &[Vec<f64>], b: &[Vec<f64>], names: (&str, &str)) -> LuaResult<()> {
    if a.len() != b.len() {
        return Err(LuaError::runtime(format!(
            "{} and {} must be indexed by the same clusters: {} vs {}",
            names.0,
            names.1,
            a.len(),
            b.len()
        )));
    }
    Ok(())
}

fn interval_to_table(lua: &Lua, iv: &Interval) -> LuaResult<LuaTable> {
    let t = lua.create_table()?;
    t.set("point", iv.point)?;
    t.set("lower", iv.lower)?;
    t.set("upper", iv.upper)?;
    t.set("draws_used", iv.draws_used)?;
    t.set("undefined_draws", iv.undefined_draws)?;
    t.set("clusters", iv.clusters)?;
    Ok(t)
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    // cluster_bootstrap_mean(by_cluster, draws, seed [, confidence])
    t.set(
        "cluster_bootstrap_mean",
        lua.create_function(
            |lua, (by_cluster, draws, seed, confidence): (LuaTable, usize, u64, Option<f64>)| {
                let clusters = table_to_clusters(&by_cluster, "by_cluster")?;
                let tallies = cluster_tallies(&clusters);
                let iv = cluster_bootstrap(
                    tallies.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| tally_over(&tallies, draw).mean(),
                )
                .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_mean: {e}")))?;
                interval_to_table(lua, &iv)
            },
        )?,
    )?;

    // cluster_bootstrap_diff(a_by_cluster, b_by_cluster, draws, seed [, confidence])
    //
    // Both means come from the same draw, so the interval is on the difference
    // rather than on two quantities compared after the fact. This requires
    // a_by_cluster[i] and b_by_cluster[i] to measure the same cluster.
    t.set(
        "cluster_bootstrap_diff",
        lua.create_function(
            |lua,
             (a_t, b_t, draws, seed, confidence): (
                LuaTable,
                LuaTable,
                usize,
                u64,
                Option<f64>,
            )| {
                let a = table_to_clusters(&a_t, "a_by_cluster")?;
                let b = table_to_clusters(&b_t, "b_by_cluster")?;
                assert_same_clusters(&a, &b, ("a_by_cluster", "b_by_cluster"))
                    .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_diff: {e}")))?;
                let (ta, tb) = (cluster_tallies(&a), cluster_tallies(&b));
                let iv = cluster_bootstrap(
                    ta.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| Some(tally_over(&ta, draw).mean()? - tally_over(&tb, draw).mean()?),
                )
                .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_diff: {e}")))?;
                interval_to_table(lua, &iv)
            },
        )?,
    )?;

    // cluster_bootstrap_ratio(num_by_cluster, den_by_cluster, draws, seed [, confidence])
    //
    // The ratio of the summed numerator to the summed denominator over the draw
    // (a ratio estimator), not the mean of per-cluster ratios. The denominator
    // must keep one sign: a draw whose denominator crosses zero is undefined,
    // since the ratio there is not a perturbation of the estimate but a
    // different quantity.
    t.set(
        "cluster_bootstrap_ratio",
        lua.create_function(
            |lua,
             (num_t, den_t, draws, seed, confidence): (
                LuaTable,
                LuaTable,
                usize,
                u64,
                Option<f64>,
            )| {
                let num = table_to_clusters(&num_t, "num_by_cluster")?;
                let den = table_to_clusters(&den_t, "den_by_cluster")?;
                assert_same_clusters(&num, &den, ("num_by_cluster", "den_by_cluster"))
                    .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_ratio: {e}")))?;
                let (tn, td) = (cluster_tallies(&num), cluster_tallies(&den));
                let whole_den: f64 = td.iter().map(|t| t.sum).sum();
                let iv = cluster_bootstrap(
                    tn.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| {
                        let d = tally_over(&td, draw).sum;
                        if d == 0.0 || (d > 0.0) != (whole_den > 0.0) {
                            return None;
                        }
                        Some(tally_over(&tn, draw).sum / d)
                    },
                )
                .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_ratio: {e}")))?;
                interval_to_table(lua, &iv)
            },
        )?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_per_cluster(values: &[f64]) -> Vec<Vec<f64>> {
        values.iter().map(|&v| vec![v]).collect()
    }

    /// The mean over a draw, from precomputed subtotals.
    fn mean_of(clusters: &[Vec<f64>]) -> impl Fn(&[usize]) -> Option<f64> + use<> {
        let tallies = cluster_tallies(clusters);
        move |draw| tally_over(&tallies, draw).mean()
    }

    #[test]
    fn point_estimate_is_the_sample_as_walked() {
        let iv = cluster_bootstrap(
            4,
            200,
            7,
            0.95,
            mean_of(&one_per_cluster(&[1.0, 2.0, 3.0, 4.0])),
        )
        .unwrap();
        assert!((iv.point - 2.5).abs() < 1e-12);
    }

    #[test]
    fn a_symmetric_sample_gives_an_interval_around_the_point_estimate() {
        // Holds for a mean over a symmetric sample. It is not a guarantee of
        // the method — see the skew note on `cluster_bootstrap`.
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let iv = cluster_bootstrap(8, 2000, 1, 0.95, mean_of(&c)).unwrap();
        assert!(iv.lower <= iv.point && iv.point <= iv.upper);
        assert!(iv.lower < iv.upper);
    }

    #[test]
    fn same_seed_gives_the_same_interval() {
        let c = one_per_cluster(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let a = cluster_bootstrap(6, 500, 42, 0.95, mean_of(&c)).unwrap();
        let b = cluster_bootstrap(6, 500, 42, 0.95, mean_of(&c)).unwrap();
        let other = cluster_bootstrap(6, 500, 43, 0.95, mean_of(&c)).unwrap();
        assert_eq!(a.lower.to_bits(), b.lower.to_bits());
        assert_eq!(a.upper.to_bits(), b.upper.to_bits());
        assert!(a.lower != other.lower || a.upper != other.upper);
    }

    #[test]
    fn a_wider_confidence_gives_a_wider_interval() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let narrow = cluster_bootstrap(8, 2000, 3, 0.50, mean_of(&c)).unwrap();
        let wide = cluster_bootstrap(8, 2000, 3, 0.99, mean_of(&c)).unwrap();
        assert!(wide.upper - wide.lower > narrow.upper - narrow.lower);
    }

    #[test]
    fn confidence_of_one_spans_the_draws_and_of_zero_collapses() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let full = cluster_bootstrap(6, 500, 8, 1.0, mean_of(&c)).unwrap();
        let none = cluster_bootstrap(6, 500, 8, 0.0, mean_of(&c)).unwrap();
        assert!(full.lower < none.lower && none.upper < full.upper);
        assert!(
            (none.upper - none.lower).abs() < 1e-12,
            "0 gives the median"
        );
    }

    #[test]
    fn a_single_draw_is_allowed_and_gives_that_draw() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0]);
        let iv = cluster_bootstrap(4, 1, 5, 0.95, mean_of(&c)).unwrap();
        assert_eq!(iv.draws_used, 1);
        assert!((iv.upper - iv.lower).abs() < 1e-12);
    }

    #[test]
    fn a_constant_sample_gives_a_zero_width_interval() {
        let c = one_per_cluster(&[2.0; 5]);
        let iv = cluster_bootstrap(5, 300, 11, 0.95, mean_of(&c)).unwrap();
        assert!((iv.upper - iv.lower).abs() < 1e-12);
        assert!((iv.point - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empty_clusters_are_counted_but_contribute_nothing() {
        // Cluster 2 holds no observations. The mean over a draw that names only
        // it is undefined; over a draw that names others it is their mean.
        let c = vec![vec![4.0], vec![], vec![6.0]];
        let iv = cluster_bootstrap(3, 2000, 5, 0.95, mean_of(&c)).unwrap();
        assert_eq!(iv.clusters, 3);
        assert!((iv.point - 5.0).abs() < 1e-12);
        assert!(iv.undefined_draws > 0, "expected some undefined draws");
        assert_eq!(iv.draws_used + iv.undefined_draws, 2000);
    }

    #[test]
    fn undefined_on_the_whole_sample_is_an_error() {
        let c: Vec<Vec<f64>> = vec![vec![], vec![]];
        let err = cluster_bootstrap(2, 100, 1, 0.95, mean_of(&c)).unwrap_err();
        assert!(
            err.contains("undefined on the sample as walked"),
            "got: {err}"
        );
    }

    #[test]
    fn a_single_cluster_is_refused() {
        // One cluster means one possible draw, and an interval of width zero
        // that would read as infinite precision.
        let c = one_per_cluster(&[1.0]);
        let err = cluster_bootstrap(1, 100, 1, 0.95, mean_of(&c)).unwrap_err();
        assert!(err.contains("at least 2 clusters"), "got: {err}");
    }

    #[test]
    fn zero_draws_and_an_implausible_draw_count_are_errors() {
        let c = one_per_cluster(&[1.0, 2.0]);
        assert!(cluster_bootstrap(2, 0, 1, 0.95, mean_of(&c)).is_err());
        let err = cluster_bootstrap(2, MAX_DRAWS + 1, 1, 0.95, mean_of(&c)).unwrap_err();
        assert!(err.contains("argument order"), "got: {err}");
    }

    #[test]
    fn confidence_outside_the_unit_interval_is_an_error() {
        let c = one_per_cluster(&[1.0, 2.0]);
        assert!(cluster_bootstrap(2, 10, 1, 1.5, mean_of(&c)).is_err());
    }

    #[test]
    fn a_difference_of_identical_sides_brackets_zero() {
        let a = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let (ta, tb) = (cluster_tallies(&a), cluster_tallies(&a));
        let iv = cluster_bootstrap(5, 2000, 9, 0.95, |d| {
            Some(tally_over(&ta, d).mean()? - tally_over(&tb, d).mean()?)
        })
        .unwrap();
        assert!(iv.point.abs() < 1e-12);
        assert!(iv.lower <= 0.0 && 0.0 <= iv.upper);
    }

    #[test]
    fn pairing_within_a_draw_is_tighter_than_combining_separate_intervals() {
        // Two sides that move together: b is a plus a constant. Measured inside
        // one draw the difference is exactly that constant. Bootstrapping the
        // sides separately and subtracting the intervals keeps both sampling
        // distributions and reports a spread that is not in the difference —
        // which is why diff cannot be composed from two calls to mean.
        let a = one_per_cluster(&[1.0, 5.0, 9.0, 13.0, 17.0]);
        let b: Vec<Vec<f64>> = a.iter().map(|c| vec![c[0] + 2.0]).collect();
        let (ta, tb) = (cluster_tallies(&a), cluster_tallies(&b));

        let paired = cluster_bootstrap(5, 2000, 4, 0.95, |d| {
            Some(tally_over(&ta, d).mean()? - tally_over(&tb, d).mean()?)
        })
        .unwrap();
        assert!((paired.point + 2.0).abs() < 1e-12);

        let sep_a = cluster_bootstrap(5, 2000, 4, 0.95, mean_of(&a)).unwrap();
        let sep_b = cluster_bootstrap(5, 2000, 77, 0.95, mean_of(&b)).unwrap();
        // The interval a naive caller would build from two separate runs.
        let naive_width = (sep_a.upper - sep_b.lower) - (sep_a.lower - sep_b.upper);

        assert!(
            naive_width > paired.upper - paired.lower,
            "naive {naive_width} should exceed paired {}",
            paired.upper - paired.lower
        );
        assert!(
            naive_width > 1.0,
            "the naive width is substantial, not a rounding artifact"
        );
    }

    #[test]
    fn a_ratio_is_of_the_sums_not_of_the_per_cluster_ratios() {
        // Per-cluster ratios are 1/1 and 3/9; their mean is 0.666..., while the
        // ratio of sums is 4/10 = 0.4. The documented estimator is the latter.
        let (tn, td) = (
            cluster_tallies(&[vec![1.0], vec![3.0]]),
            cluster_tallies(&[vec![1.0], vec![9.0]]),
        );
        let iv = cluster_bootstrap(2, 500, 2, 0.95, |d| {
            let s = tally_over(&td, d).sum;
            if s == 0.0 {
                return None;
            }
            Some(tally_over(&tn, d).sum / s)
        })
        .unwrap();
        assert!((iv.point - 0.4).abs() < 1e-12);
    }

    #[test]
    fn a_zero_denominator_draw_is_undefined_not_infinite() {
        let (tn, td) = (
            cluster_tallies(&[vec![1.0], vec![2.0]]),
            cluster_tallies(&[vec![0.0], vec![5.0]]),
        );
        let whole: f64 = td.iter().map(|t| t.sum).sum();
        let iv = cluster_bootstrap(2, 2000, 6, 0.95, |d| {
            let s = tally_over(&td, d).sum;
            if s == 0.0 || (s > 0.0) != (whole > 0.0) {
                return None;
            }
            Some(tally_over(&tn, d).sum / s)
        })
        .unwrap();
        // The draw naming cluster 1 twice has a zero denominator.
        assert!(iv.undefined_draws > 0);
        assert!(iv.lower.is_finite() && iv.upper.is_finite());
    }

    #[test]
    fn a_denominator_that_flips_sign_within_a_draw_is_undefined() {
        // Whole-sample denominator is 3 + 3 - 5 = 1 > 0, but draws that name
        // the negative cluster twice go negative. Those ratios are not
        // perturbations of the estimate — without this guard they enter the
        // percentiles and drag the lower end below zero.
        let (tn, td) = (
            cluster_tallies(&[vec![1.0], vec![1.0], vec![1.0]]),
            cluster_tallies(&[vec![3.0], vec![3.0], vec![-5.0]]),
        );
        let whole: f64 = td.iter().map(|t| t.sum).sum();
        assert!((whole - 1.0).abs() < 1e-12);

        let guarded = cluster_bootstrap(3, 2000, 12, 0.95, |d| {
            let s = tally_over(&td, d).sum;
            if s == 0.0 || (s > 0.0) != (whole > 0.0) {
                return None;
            }
            Some(tally_over(&tn, d).sum / s)
        })
        .unwrap();
        assert!(
            guarded.undefined_draws > 0,
            "sign-flipping draws exist here"
        );
        assert!(
            guarded.lower > 0.0,
            "a positive-denominator ratio stays positive"
        );

        let unguarded = cluster_bootstrap(3, 2000, 12, 0.95, |d| {
            let s = tally_over(&td, d).sum;
            if s == 0.0 {
                return None;
            }
            Some(tally_over(&tn, d).sum / s)
        })
        .unwrap();
        assert!(
            unguarded.lower < 0.0,
            "without the guard the interval crosses zero: {unguarded:?}"
        );
    }
}
