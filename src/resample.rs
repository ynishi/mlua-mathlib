use mlua::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::stats::percentile_impl;

/// Confidence level used when the caller does not name one.
const DEFAULT_CONFIDENCE: f64 = 0.95;

/// A percentile interval and the accounting of the draws behind it.
#[derive(Debug)]
struct Interval {
    /// The statistic on the sample as walked — every cluster once.
    point: f64,
    lower: f64,
    upper: f64,
    /// Draws the statistic was defined on. The interval rests on these.
    draws_used: usize,
    /// Draws the statistic was undefined on, and which the interval
    /// therefore does not rest on. A large share means the statistic sits on
    /// few enough clusters that resampling destroys it.
    undefined_draws: usize,
    clusters: usize,
}

/// A running sum and count, so a statistic over a draw can be accumulated
/// without materializing the resampled observations.
#[derive(Clone, Copy, Default)]
struct Tally {
    sum: f64,
    n: usize,
}

impl Tally {
    /// `None` on an empty tally: a mean over no observations is undefined,
    /// and that is a fact about the draw the caller should see rather than a
    /// zero to average in.
    fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.sum / self.n as f64)
        }
    }
}

/// Accumulate the observations of the clusters a draw names, with repetition.
fn tally_over(clusters: &[Vec<f64>], draw: &[usize]) -> Tally {
    let mut tally = Tally::default();
    for &c in draw {
        for &v in &clusters[c] {
            tally.sum += v;
            tally.n += 1;
        }
    }
    tally
}

/// Resample `clusters` clusters `draws` times and bound `stat`.
///
/// `stat` receives one draw — cluster indices with repetition — and returns
/// its value on exactly those clusters, or `None` where the draw leaves it
/// undefined. Every term of a difference or ratio must be computed from that
/// same list: two separately bootstrapped quantities carry no joint
/// distribution, so their intervals cannot be combined after the fact.
fn cluster_bootstrap(
    clusters: usize,
    draws: usize,
    seed: u64,
    confidence: f64,
    stat: impl Fn(&[usize]) -> Option<f64>,
) -> Result<Interval, String> {
    if clusters == 0 {
        return Err("needs at least one cluster to resample".into());
    }
    if draws == 0 {
        return Err("needs at least one draw".into());
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
    let mut values: Vec<f64> = Vec::with_capacity(draws);
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
            let v: f64 = inner.raw_get(j)?;
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
fn assert_same_clusters(a: &[Vec<f64>], b: &[Vec<f64>]) -> LuaResult<()> {
    if a.len() != b.len() {
        return Err(LuaError::runtime(format!(
            "the two sides must be indexed by the same clusters: {} vs {}",
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
                let iv = cluster_bootstrap(
                    clusters.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| tally_over(&clusters, draw).mean(),
                )
                .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_mean: {e}")))?;
                interval_to_table(lua, &iv)
            },
        )?,
    )?;

    // cluster_bootstrap_diff(a_by_cluster, b_by_cluster, draws, seed [, confidence])
    // Both means come from the same draw, so the interval is on the difference
    // rather than on two quantities compared after the fact.
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
                let a = table_to_clusters(&a_t, "a")?;
                let b = table_to_clusters(&b_t, "b")?;
                assert_same_clusters(&a, &b)
                    .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_diff: {e}")))?;
                let iv = cluster_bootstrap(
                    a.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| {
                        let ma = tally_over(&a, draw).mean()?;
                        let mb = tally_over(&b, draw).mean()?;
                        Some(ma - mb)
                    },
                )
                .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_diff: {e}")))?;
                interval_to_table(lua, &iv)
            },
        )?,
    )?;

    // cluster_bootstrap_ratio(num_by_cluster, den_by_cluster, draws, seed [, confidence])
    // The ratio of the summed numerator to the summed denominator over the
    // draw (a ratio estimator), not the mean of per-cluster ratios.
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
                let num = table_to_clusters(&num_t, "numerator")?;
                let den = table_to_clusters(&den_t, "denominator")?;
                assert_same_clusters(&num, &den)
                    .map_err(|e| LuaError::runtime(format!("cluster_bootstrap_ratio: {e}")))?;
                let iv = cluster_bootstrap(
                    num.len(),
                    draws,
                    seed,
                    confidence.unwrap_or(DEFAULT_CONFIDENCE),
                    |draw| {
                        let d = tally_over(&den, draw).sum;
                        if d == 0.0 {
                            return None;
                        }
                        Some(tally_over(&num, draw).sum / d)
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

    #[test]
    fn point_estimate_is_the_sample_as_walked() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0]);
        let iv = cluster_bootstrap(4, 200, 7, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        assert!((iv.point - 2.5).abs() < 1e-12);
    }

    #[test]
    fn interval_brackets_the_point_estimate() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let iv = cluster_bootstrap(8, 2000, 1, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        assert!(iv.lower <= iv.point && iv.point <= iv.upper);
        assert!(iv.lower < iv.upper);
    }

    #[test]
    fn same_seed_gives_the_same_interval() {
        let c = one_per_cluster(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let a = cluster_bootstrap(6, 500, 42, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        let b = cluster_bootstrap(6, 500, 42, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        assert_eq!(a.lower.to_bits(), b.lower.to_bits());
        assert_eq!(a.upper.to_bits(), b.upper.to_bits());
    }

    #[test]
    fn a_wider_confidence_gives_a_wider_interval() {
        let c = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let narrow = cluster_bootstrap(8, 2000, 3, 0.50, |d| tally_over(&c, d).mean()).unwrap();
        let wide = cluster_bootstrap(8, 2000, 3, 0.99, |d| tally_over(&c, d).mean()).unwrap();
        assert!(wide.upper - wide.lower > narrow.upper - narrow.lower);
    }

    #[test]
    fn a_constant_sample_gives_a_zero_width_interval() {
        let c = one_per_cluster(&[2.0; 5]);
        let iv = cluster_bootstrap(5, 300, 11, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        assert!((iv.upper - iv.lower).abs() < 1e-12);
        assert!((iv.point - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empty_clusters_are_counted_but_contribute_nothing() {
        // Cluster 2 holds no observations. The mean over a draw that names only
        // it is undefined; over a draw that names others it is their mean.
        let c = vec![vec![4.0], vec![], vec![6.0]];
        let iv = cluster_bootstrap(3, 2000, 5, 0.95, |d| tally_over(&c, d).mean()).unwrap();
        assert_eq!(iv.clusters, 3);
        assert!((iv.point - 5.0).abs() < 1e-12);
        // Every-cluster-is-the-empty-one draws exist at 3 clusters over 2000 draws.
        assert!(iv.undefined_draws > 0, "expected some undefined draws");
        assert_eq!(iv.draws_used + iv.undefined_draws, 2000);
    }

    #[test]
    fn undefined_on_the_whole_sample_is_an_error() {
        let c: Vec<Vec<f64>> = vec![vec![], vec![]];
        let err = cluster_bootstrap(2, 100, 1, 0.95, |d| tally_over(&c, d).mean()).unwrap_err();
        assert!(
            err.contains("undefined on the sample as walked"),
            "got: {err}"
        );
    }

    #[test]
    fn zero_clusters_and_zero_draws_are_errors() {
        let c = one_per_cluster(&[1.0]);
        assert!(cluster_bootstrap(0, 10, 1, 0.95, |d| tally_over(&c, d).mean()).is_err());
        assert!(cluster_bootstrap(1, 0, 1, 0.95, |d| tally_over(&c, d).mean()).is_err());
    }

    #[test]
    fn confidence_outside_the_unit_interval_is_an_error() {
        let c = one_per_cluster(&[1.0, 2.0]);
        assert!(cluster_bootstrap(2, 10, 1, 1.5, |d| tally_over(&c, d).mean()).is_err());
    }

    #[test]
    fn a_difference_of_identical_sides_brackets_zero() {
        let a = one_per_cluster(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = a.clone();
        let iv = cluster_bootstrap(5, 2000, 9, 0.95, |d| {
            Some(tally_over(&a, d).mean()? - tally_over(&b, d).mean()?)
        })
        .unwrap();
        assert!((iv.point).abs() < 1e-12);
        assert!(iv.lower <= 0.0 && 0.0 <= iv.upper);
    }

    #[test]
    fn a_difference_from_the_same_draw_is_tighter_than_from_separate_ones() {
        // b tracks a with a constant offset, so within one draw the difference
        // is exactly that offset and its interval collapses. Bootstrapping the
        // two sides separately would keep both sampling distributions and
        // report a spread that is not there — this is why diff exists at all.
        let a = one_per_cluster(&[1.0, 5.0, 9.0, 13.0, 17.0]);
        let b: Vec<Vec<f64>> = a.iter().map(|c| vec![c[0] + 2.0]).collect();
        let paired = cluster_bootstrap(5, 2000, 4, 0.95, |d| {
            Some(tally_over(&a, d).mean()? - tally_over(&b, d).mean()?)
        })
        .unwrap();
        assert!((paired.upper - paired.lower).abs() < 1e-12);
        assert!((paired.point + 2.0).abs() < 1e-12);

        let sep_a = cluster_bootstrap(5, 2000, 4, 0.95, |d| tally_over(&a, d).mean()).unwrap();
        assert!(
            sep_a.upper - sep_a.lower > 1.0,
            "the individual side does vary across draws"
        );
    }

    #[test]
    fn a_ratio_is_of_the_sums_not_of_the_per_cluster_ratios() {
        // Per-cluster ratios are 1/1 and 3/9; their mean is 0.666..., while the
        // ratio of sums is 4/10 = 0.4. The documented estimator is the latter.
        let num = vec![vec![1.0], vec![3.0]];
        let den = vec![vec![1.0], vec![9.0]];
        let iv = cluster_bootstrap(2, 500, 2, 0.95, |d| {
            let s = tally_over(&den, d).sum;
            if s == 0.0 {
                return None;
            }
            Some(tally_over(&num, d).sum / s)
        })
        .unwrap();
        assert!((iv.point - 0.4).abs() < 1e-12);
    }

    #[test]
    fn a_zero_denominator_draw_is_undefined_not_infinite() {
        let num = vec![vec![1.0], vec![2.0]];
        let den = vec![vec![0.0], vec![5.0]];
        let iv = cluster_bootstrap(2, 2000, 6, 0.95, |d| {
            let s = tally_over(&den, d).sum;
            if s == 0.0 {
                return None;
            }
            Some(tally_over(&num, d).sum / s)
        })
        .unwrap();
        // The draw naming cluster 1 twice has a zero denominator.
        assert!(iv.undefined_draws > 0);
        assert!(iv.lower.is_finite() && iv.upper.is_finite());
    }
}
