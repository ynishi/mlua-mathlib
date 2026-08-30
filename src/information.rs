use mlua::prelude::*;

use crate::stats::table_to_vec;

/// Unit roundoff of `f32`: half an ulp at 1.0.
const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// Safety factor over the statistical growth of the rounding error.
const TOL_SLACK: f64 = 32.0;

/// Absolute tolerance for the "sums to 1" check on a distribution of `n` elements.
///
/// Callers normalize in f32 (a softmax output) and widen to f64, so the check
/// has to admit the drift that carries: `1.2e-5` at `n = 4096` and `1.2e-4` at
/// `n = 50257`, against the `1e-6` this used to compare against.
///
/// The bound grows like `√n * u` (`u` = f32 unit roundoff) because the per-term
/// rounding errors cancel rather than accumulate. The deterministic worst case
/// `(n-1) * u` is ~26x looser than anything observed, and at a 50257-entry
/// vocabulary it would let 0.3% of the mass go missing unnoticed — a top-k mask
/// that forgets to renormalize lands exactly there. [`TOL_SLACK`] keeps a factor
/// of ~4 over the observed maximum while holding that leak to 0.04%.
///
/// One consequence: at short lengths this is looser than the fixed `1e-6` it
/// replaced (`1.9e-6` at `n = 1`, `3.8e-6` at `n = 4`). A floor would be dead
/// weight — `√n` growth passes `1e-6` before `n` reaches 1 — so the bound is a
/// single expression rather than a floor that never applies.
fn norm_tol(n: usize) -> f64 {
    TOL_SLACK * (n as f64).sqrt() * F32_UNIT_ROUNDOFF
}

/// Why a slice is not a probability distribution.
///
/// Each variant names the offending side and index so the caller can find the
/// element rather than re-deriving it from a failed sum. Indices are **1-based**,
/// matching the Lua tables these come from and the diagnostics in
/// [`crate::stats::table_to_vec`].
#[derive(Debug, PartialEq)]
enum DistError {
    /// Zero-length input: probability over an empty support is undefined.
    ///
    /// Unreachable from Lua — `table_to_vec` refuses an empty table first. Kept
    /// as the internal contract for the `_impl` functions.
    Empty { side: &'static str },
    /// A pairwise call received two distributions over different supports.
    LengthMismatch { p: usize, q: usize },
    /// A joint distribution matrix had a row of the wrong width. Named
    /// separately from [`Self::LengthMismatch`] so the row can be pointed at —
    /// `p` / `q` are the pairwise vocabulary and say nothing here.
    RaggedRow {
        row: usize,
        expected: usize,
        found: usize,
    },
    /// An element was `NaN` or `±inf`.
    ///
    /// Unreachable from Lua for the same reason as [`Self::Empty`].
    NonFinite {
        side: &'static str,
        index: usize,
        value: f64,
    },
    /// An element was strictly negative. Zero is allowed.
    Negative {
        side: &'static str,
        index: usize,
        value: f64,
    },
    /// The sum lies outside `1 ± tol`, with `tol` from [`norm_tol`].
    NotNormalized {
        side: &'static str,
        sum: f64,
        tol: f64,
    },
    /// A support passed to [`wasserstein_1d_impl`] was not strictly increasing.
    /// The distance is the area between two CDFs, which needs the positions in
    /// order to mean anything.
    UnorderedSupport {
        index: usize,
        previous: f64,
        value: f64,
    },
    /// A support with one position per bin was expected. Named separately from
    /// [`Self::LengthMismatch`], whose `p` / `q` would put the blame on a
    /// distribution whose length is fine.
    SupportLengthMismatch { distribution: usize, support: usize },
}

impl std::fmt::Display for DistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { side } => write!(f, "{side} is empty"),
            Self::LengthMismatch { p, q } => {
                write!(f, "length mismatch: p has {p}, q has {q}")
            }
            Self::RaggedRow {
                row,
                expected,
                found,
            } => {
                write!(
                    f,
                    "joint[{row}] has {found} columns, expected {expected} to match joint[1]"
                )
            }
            Self::NonFinite { side, index, value } => {
                write!(f, "{side}[{index}] is not finite: {value}")
            }
            Self::Negative { side, index, value } => {
                write!(f, "{side}[{index}] is negative: {value}")
            }
            Self::NotNormalized { side, sum, tol } => {
                write!(f, "{side} sums to {sum}, expected 1 ± {tol:e}")
            }
            Self::SupportLengthMismatch {
                distribution,
                support,
            } => {
                write!(
                    f,
                    "support needs one position per bin: {distribution} bins, {support} positions"
                )
            }
            Self::UnorderedSupport {
                index,
                previous,
                value,
            } => {
                write!(
                    f,
                    "support must be strictly increasing: support[{index}] = {value} \
                     does not exceed {previous}"
                )
            }
        }
    }
}

/// Check that `dist` is non-empty, finite, non-negative and sums to 1.
///
/// Per-index diagnostics win over the aggregate sum verdict: a slice holding a
/// `NaN` fails both, and pointing at the element is the more useful answer.
fn validate(dist: &[f64], side: &'static str) -> Result<(), DistError> {
    if dist.is_empty() {
        return Err(DistError::Empty { side });
    }
    let mut sum = 0.0;
    for (i, &value) in dist.iter().enumerate() {
        // 1-based: these indices are quoted back to Lua, where the table starts at 1.
        let index = i + 1;
        if !value.is_finite() {
            return Err(DistError::NonFinite { side, index, value });
        }
        if value < 0.0 {
            return Err(DistError::Negative { side, index, value });
        }
        sum += value;
    }
    let tol = norm_tol(dist.len());
    if (sum - 1.0).abs() > tol {
        return Err(DistError::NotNormalized { side, sum, tol });
    }
    Ok(())
}

/// [`validate`] both sides after checking they share a support.
fn validate_pair(p: &[f64], q: &[f64]) -> Result<(), DistError> {
    if p.len() != q.len() {
        return Err(DistError::LengthMismatch {
            p: p.len(),
            q: q.len(),
        });
    }
    validate(p, "p")?;
    validate(q, "q")?;
    Ok(())
}

/// Shannon entropy: H(p) = -Σ p_i * ln(p_i)
/// Zero terms are dropped (0·log 0 := 0), so hard zeros are fine.
fn entropy_impl(probs: &[f64]) -> Result<f64, DistError> {
    validate(probs, "probs")?;
    Ok(probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum())
}

/// KL divergence: D_KL(p || q) = Σ p_i * ln(p_i / q_i)
/// Returns `+inf` where q_i = 0 while p_i > 0: the definition is genuinely
/// infinite there, and a caller sweeping a sequence of distributions needs a
/// number rather than an abort.
fn kl_divergence_impl(p: &[f64], q: &[f64]) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let mut acc = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue;
        }
        if qi == 0.0 {
            return Ok(f64::INFINITY);
        }
        acc += pi * (pi / qi).ln();
    }
    Ok(acc)
}

/// Jensen-Shannon divergence: JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
/// where m = 0.5 * (p + q). Always finite, symmetric, bounded [0, ln(2)].
fn js_divergence_impl(p: &[f64], q: &[f64]) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let m: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| 0.5 * (pi + qi))
        .collect();
    let kl_pm = kl_divergence_impl(p, &m)?;
    let kl_qm = kl_divergence_impl(q, &m)?;
    Ok(0.5 * kl_pm + 0.5 * kl_qm)
}

/// Cross-entropy: H(p, q) = -Σ p_i * ln(q_i)
/// Returns `+inf` where q_i = 0 while p_i > 0, as [`kl_divergence_impl`] does.
fn cross_entropy_impl(p: &[f64], q: &[f64]) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let mut acc = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue;
        }
        if qi == 0.0 {
            return Ok(f64::INFINITY);
        }
        acc -= pi * qi.ln();
    }
    Ok(acc)
}

/// Hellinger distance: `H(p,q) = sqrt(Σ (sqrt(p_i) - sqrt(q_i))² / 2)`.
///
/// A true metric on distributions — symmetric, and it satisfies the triangle
/// inequality, which neither KL (asymmetric) nor its square root does. Bounded
/// `[0, 1]`: zero when the distributions are identical, one when their supports
/// are disjoint.
///
/// Sits between [`tvd_impl`] and [`js_divergence_impl`] in practice. Like TVD
/// it is bounded and metric; unlike TVD it responds to the *ratio* of the
/// probabilities rather than their difference, so it separates two small
/// probabilities that differ by a factor where TVD sees only a small gap.
///
/// # Summed from the differences, not from `1 - BC`
///
/// `Σ(√p - √q)² = 2 - 2·BC` makes the two forms algebraically identical but not
/// numerically. As the distributions converge the Bhattacharyya coefficient
/// approaches 1 and `1 - BC` cancels: that route returns 15x the true distance
/// at `q = p ± 1e-9` and exactly zero at `1e-8`. Summing the per-element
/// differences keeps each term at its own scale, and being a sum of squares it
/// needs no clamp to stay non-negative.
fn hellinger_impl(p: &[f64], q: &[f64]) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let sq: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&a, &b)| {
            let d = a.sqrt() - b.sqrt();
            d * d
        })
        .sum();
    Ok((sq / 2.0).sqrt())
}

/// 1-Wasserstein distance between two distributions over an **ordered** support.
///
/// `W₁(p,q) = Σ |F_p(i) - F_q(i)| * (x_{i+1} - x_i)`, the area between the two
/// cumulative distributions — the earth-mover's distance in one dimension.
///
/// # This one reads the order of the support
///
/// Every other distance in this module is invariant to permuting the bins: KL,
/// JS, TVD and Hellinger compare `p_i` against `q_i` and never against `p_j`.
/// Wasserstein does not. Moving mass one bin to the left costs less than moving
/// it ten bins, which is what makes it the right measure over ordered outcomes
/// (scores, ranks, token positions) and the wrong one over unordered categories
/// — there the bin order is arbitrary and so would be the answer.
///
/// `support` gives the position of each bin and must be strictly increasing;
/// omitted, the positions are `0, 1, 2, …` and the result is in bins.
///
/// Unlike TVD it is unbounded above: it scales with how far the mass has to
/// travel, so the units are those of the support.
fn wasserstein_1d_impl(p: &[f64], q: &[f64], support: Option<&[f64]>) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let widths: Vec<f64> = match support {
        None => vec![1.0; p.len().saturating_sub(1)],
        Some(x) => {
            if x.len() != p.len() {
                return Err(DistError::SupportLengthMismatch {
                    distribution: p.len(),
                    support: x.len(),
                });
            }
            for (i, w) in x.windows(2).enumerate() {
                // partial_cmp rather than `<=` so a NaN position is refused
                // too, rather than comparing false and slipping through.
                if w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater) {
                    return Err(DistError::UnorderedSupport {
                        index: i + 2,
                        previous: w[0],
                        value: w[1],
                    });
                }
            }
            x.windows(2).map(|w| w[1] - w[0]).collect()
        }
    };

    let mut cp = 0.0;
    let mut cq = 0.0;
    let mut acc = 0.0;
    for (i, width) in widths.iter().enumerate() {
        cp += p[i];
        cq += q[i];
        acc += (cp - cq).abs() * width;
    }
    Ok(acc)
}

/// Mutual information `I(X;Y) = Σ p(x,y) * ln(p(x,y) / (p(x) * p(y)))`, in nats.
///
/// Takes the **joint** distribution as a row-major matrix — `joint[i][j]` is
/// `P(X = i, Y = j)` and the whole matrix sums to 1. The marginals are derived
/// from it, which is the point: everything the other functions here take is a
/// single distribution, so nothing could express a relationship between two
/// variables.
///
/// Zero exactly when the variables are independent, and bounded above by
/// `min(H(X), H(Y))`. Cells with `p(x,y) = 0` contribute nothing (the
/// `0·log 0 := 0` convention), and a marginal of zero can only arise where the
/// whole row or column is zero, so no term divides by zero.
///
/// The joint entropy, if the caller wants it separately, is `entropy` over the
/// flattened matrix.
fn mutual_information_impl(joint: &[Vec<f64>]) -> Result<f64, DistError> {
    if joint.is_empty() {
        return Err(DistError::Empty { side: "joint" });
    }
    let cols = joint[0].len();
    if cols == 0 {
        return Err(DistError::Empty { side: "joint" });
    }
    for (i, row) in joint.iter().enumerate() {
        if row.len() != cols {
            return Err(DistError::RaggedRow {
                row: i + 1,
                expected: cols,
                found: row.len(),
            });
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(DistError::NonFinite {
                    side: "joint",
                    index: i * cols + j + 1,
                    value: v,
                });
            }
            if v < 0.0 {
                return Err(DistError::Negative {
                    side: "joint",
                    index: i * cols + j + 1,
                    value: v,
                });
            }
        }
    }

    let cells = joint.len() * cols;
    let sum: f64 = joint.iter().flat_map(|r| r.iter()).sum();
    let tol = norm_tol(cells);
    if (sum - 1.0).abs() > tol {
        return Err(DistError::NotNormalized {
            side: "joint",
            sum,
            tol,
        });
    }

    // Divide through by the observed sum rather than trusting it to be 1.
    // A joint summing to `S` would otherwise yield `S·I - S·ln S ≈ I - (S-1)`,
    // biasing the result by up to the tolerance itself — and in one direction:
    // a sum below 1 reports a positive mutual information for variables that
    // are exactly independent. The tolerance admits drift by design (callers
    // normalize in f32), so that is the common case, not the rare one.
    let row_marginals: Vec<f64> = joint.iter().map(|r| r.iter().sum::<f64>() / sum).collect();
    let col_marginals: Vec<f64> = (0..cols)
        .map(|j| joint.iter().map(|r| r[j]).sum::<f64>() / sum)
        .collect();

    let mut acc = 0.0;
    for (i, row) in joint.iter().enumerate() {
        for (j, &p_xy) in row.iter().enumerate() {
            if p_xy > 0.0 {
                let p = p_xy / sum;
                // Both marginals are >= p > 0 here, so neither divides by zero.
                acc += p * (p / (row_marginals[i] * col_marginals[j])).ln();
            }
        }
    }
    // Cancellation can leave a tiny negative where the truth is exactly zero.
    Ok(acc.max(0.0))
}

/// Total variation distance: TVD(p, q) = 0.5 * Σ|p_i - q_i|
/// Symmetric: the share of probability mass the two distributions disagree on.
/// Bounded [0, 1] for exactly normalized inputs; since [`validate`] admits a sum
/// of `1 ± norm_tol(n)`, a caller feeding two drifted distributions can see up to
/// `1 + norm_tol(n)` back.
fn tvd_impl(p: &[f64], q: &[f64]) -> Result<f64, DistError> {
    validate_pair(p, q)?;
    let l1: f64 = p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).sum();
    Ok(0.5 * l1)
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    t.set(
        "entropy",
        lua.create_function(|_, table: LuaTable| {
            let v = table_to_vec(&table)?;
            entropy_impl(&v).map_err(|e| LuaError::runtime(format!("entropy: {e}")))
        })?,
    )?;

    t.set(
        "kl_divergence",
        lua.create_function(|_, (p_t, q_t): (LuaTable, LuaTable)| {
            let p = table_to_vec(&p_t)?;
            let q = table_to_vec(&q_t)?;
            kl_divergence_impl(&p, &q).map_err(|e| LuaError::runtime(format!("kl_divergence: {e}")))
        })?,
    )?;

    t.set(
        "js_divergence",
        lua.create_function(|_, (p_t, q_t): (LuaTable, LuaTable)| {
            let p = table_to_vec(&p_t)?;
            let q = table_to_vec(&q_t)?;
            js_divergence_impl(&p, &q).map_err(|e| LuaError::runtime(format!("js_divergence: {e}")))
        })?,
    )?;

    t.set(
        "cross_entropy",
        lua.create_function(|_, (p_t, q_t): (LuaTable, LuaTable)| {
            let p = table_to_vec(&p_t)?;
            let q = table_to_vec(&q_t)?;
            cross_entropy_impl(&p, &q).map_err(|e| LuaError::runtime(format!("cross_entropy: {e}")))
        })?,
    )?;

    t.set(
        "hellinger",
        lua.create_function(|_, (p_t, q_t): (LuaTable, LuaTable)| {
            let p = table_to_vec(&p_t)?;
            let q = table_to_vec(&q_t)?;
            hellinger_impl(&p, &q).map_err(|e| LuaError::runtime(format!("hellinger: {e}")))
        })?,
    )?;

    // wasserstein_1d(p, q)            -- positions are 0, 1, 2, ...
    // wasserstein_1d(p, q, support)   -- explicit, strictly increasing
    t.set(
        "wasserstein_1d",
        lua.create_function(
            |_, (p_t, q_t, support_t): (LuaTable, LuaTable, Option<LuaTable>)| {
                let p = table_to_vec(&p_t)?;
                let q = table_to_vec(&q_t)?;
                let support = match support_t {
                    Some(t) => Some(table_to_vec(&t)?),
                    None => None,
                };
                wasserstein_1d_impl(&p, &q, support.as_deref())
                    .map_err(|e| LuaError::runtime(format!("wasserstein_1d: {e}")))
            },
        )?,
    )?;

    t.set(
        "mutual_information",
        lua.create_function(|_, joint_t: LuaTable| {
            let rows = joint_t.raw_len();
            if rows == 0 {
                return Err(LuaError::runtime("mutual_information: joint is empty"));
            }
            let mut joint = Vec::with_capacity(rows);
            for i in 1..=rows {
                let row_t: LuaTable = joint_t.raw_get(i).map_err(|_| {
                    LuaError::runtime(format!("mutual_information: joint[{i}] is not a table"))
                })?;
                let cols = row_t.raw_len();
                let mut row = Vec::with_capacity(cols);
                for j in 1..=cols {
                    let v: f64 = row_t.raw_get(j).map_err(|_| {
                        LuaError::runtime(format!(
                            "mutual_information: joint[{i}][{j}] is not a number"
                        ))
                    })?;
                    row.push(v);
                }
                joint.push(row);
            }
            mutual_information_impl(&joint)
                .map_err(|e| LuaError::runtime(format!("mutual_information: {e}")))
        })?,
    )?;

    t.set(
        "tvd",
        lua.create_function(|_, (p_t, q_t): (LuaTable, LuaTable)| {
            let p = table_to_vec(&p_t)?;
            let q = table_to_vec(&q_t)?;
            tvd_impl(&p, &q).map_err(|e| LuaError::runtime(format!("tvd: {e}")))
        })?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A length-`n` distribution normalized the way a caller's f32 softmax is:
    /// accumulate and divide in f32, then widen.
    fn f32_normalized(n: usize) -> Vec<f64> {
        let raw: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 + 1.0).sqrt()).collect();
        let total: f32 = raw.iter().sum();
        raw.iter().map(|&x| (x / total) as f64).collect()
    }

    #[test]
    fn entropy_uniform() {
        // H(uniform(4)) = ln(4)
        let h = entropy_impl(&[0.25, 0.25, 0.25, 0.25]).unwrap();
        assert!((h - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn entropy_degenerate() {
        let h = entropy_impl(&[1.0, 0.0, 0.0]).unwrap();
        assert!((h - 0.0).abs() < 1e-10);
    }

    #[test]
    fn kl_divergence_same() {
        let kl = kl_divergence_impl(&[0.5, 0.5], &[0.5, 0.5]).unwrap();
        assert!((kl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn kl_divergence_positive() {
        let kl = kl_divergence_impl(&[0.9, 0.1], &[0.5, 0.5]).unwrap();
        assert!(kl > 0.0);
    }

    #[test]
    fn js_divergence_symmetric() {
        let p = [0.9, 0.1];
        let q = [0.1, 0.9];
        let js_pq = js_divergence_impl(&p, &q).unwrap();
        let js_qp = js_divergence_impl(&q, &p).unwrap();
        assert!((js_pq - js_qp).abs() < 1e-10);
    }

    #[test]
    fn js_divergence_bounded() {
        let js = js_divergence_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(js <= 2.0_f64.ln() + 1e-10);
    }

    #[test]
    fn cross_entropy_equals_entropy_when_same() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let ce = cross_entropy_impl(&p, &p).unwrap();
        let h = entropy_impl(&p).unwrap();
        assert!((ce - h).abs() < 1e-10);
    }

    #[test]
    fn kl_divergence_infinite_off_support() {
        let kl = kl_divergence_impl(&[0.5, 0.5], &[1.0, 0.0]).unwrap();
        assert!(kl.is_infinite() && kl.is_sign_positive());
    }

    #[test]
    fn cross_entropy_infinite_off_support() {
        let ce = cross_entropy_impl(&[0.5, 0.5], &[1.0, 0.0]).unwrap();
        assert!(ce.is_infinite() && ce.is_sign_positive());
    }

    #[test]
    fn kl_divergence_finite_when_p_is_zero_off_support() {
        // p_i = 0 where q_i = 0 contributes nothing, so the result stays finite.
        let kl = kl_divergence_impl(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!((kl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn hellinger_spans_zero_to_one() {
        let p = [0.25, 0.25, 0.5];
        assert!(hellinger_impl(&p, &p).unwrap().abs() < 1e-12);
        let disjoint = hellinger_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((disjoint - 1.0).abs() < 1e-12, "got {disjoint}");
    }

    #[test]
    fn hellinger_is_symmetric_and_obeys_the_triangle_inequality() {
        // A metric, which neither KL nor its square root is.
        let p = [0.7, 0.2, 0.1];
        let q = [0.1, 0.3, 0.6];
        let r = [0.4, 0.4, 0.2];
        let pq = hellinger_impl(&p, &q).unwrap();
        assert!(
            (pq - hellinger_impl(&q, &p).unwrap()).abs() < 1e-12,
            "symmetric"
        );
        let pr = hellinger_impl(&p, &r).unwrap();
        let rq = hellinger_impl(&r, &q).unwrap();
        assert!(pq <= pr + rq + 1e-12, "{pq} > {pr} + {rq}");
    }

    #[test]
    fn hellinger_reads_ratios_where_tvd_reads_differences() {
        // Two pairs with the same total variation. Hellinger separates the one
        // whose small probabilities differ by a large factor.
        let base = [0.001, 0.499, 0.5];
        let ratio_shift = [0.011, 0.489, 0.5]; // 11x on the small bin
        let bulk_shift = [0.001, 0.509, 0.49]; // same 0.01 moved in the bulk

        let tvd_ratio = tvd_impl(&base, &ratio_shift).unwrap();
        let tvd_bulk = tvd_impl(&base, &bulk_shift).unwrap();
        assert!((tvd_ratio - tvd_bulk).abs() < 1e-12, "TVD sees them alike");

        let h_ratio = hellinger_impl(&base, &ratio_shift).unwrap();
        let h_bulk = hellinger_impl(&base, &bulk_shift).unwrap();
        assert!(h_ratio > h_bulk * 2.0, "h_ratio={h_ratio} h_bulk={h_bulk}");
    }

    #[test]
    fn wasserstein_measures_how_far_the_mass_moved() {
        // One unit of mass shifted by k bins costs k.
        for k in 1..5 {
            let mut p = vec![0.0; 6];
            let mut q = vec![0.0; 6];
            p[0] = 1.0;
            q[k] = 1.0;
            let w = wasserstein_1d_impl(&p, &q, None).unwrap();
            assert!((w - k as f64).abs() < 1e-12, "k={k} gave {w}");
        }
    }

    #[test]
    fn wasserstein_reads_the_order_where_the_others_do_not() {
        // Same mass moved one bin versus five. TVD and Hellinger report the
        // same number for both; Wasserstein does not.
        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let near = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let far = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        assert!((tvd_impl(&a, &near).unwrap() - tvd_impl(&a, &far).unwrap()).abs() < 1e-12);
        assert!(
            (hellinger_impl(&a, &near).unwrap() - hellinger_impl(&a, &far).unwrap()).abs() < 1e-12
        );

        let w_near = wasserstein_1d_impl(&a, &near, None).unwrap();
        let w_far = wasserstein_1d_impl(&a, &far, None).unwrap();
        assert!((w_near - 1.0).abs() < 1e-12, "got {w_near}");
        assert!((w_far - 5.0).abs() < 1e-12, "got {w_far}");
    }

    #[test]
    fn wasserstein_scales_with_the_support() {
        // The same shift measured on a support ten times as wide costs ten
        // times as much: the units are the support's.
        let p = [1.0, 0.0, 0.0];
        let q = [0.0, 0.0, 1.0];
        let unit = wasserstein_1d_impl(&p, &q, Some(&[0.0, 1.0, 2.0])).unwrap();
        let wide = wasserstein_1d_impl(&p, &q, Some(&[0.0, 10.0, 20.0])).unwrap();
        assert!((unit - 2.0).abs() < 1e-12, "got {unit}");
        assert!((wide - 20.0).abs() < 1e-12, "got {wide}");
        // Omitting the support is the unit-spaced case.
        assert!((wasserstein_1d_impl(&p, &q, None).unwrap() - unit).abs() < 1e-12);
    }

    #[test]
    fn wasserstein_is_symmetric_and_zero_on_identity() {
        let p = [0.2, 0.3, 0.5];
        let q = [0.5, 0.1, 0.4];
        assert!(wasserstein_1d_impl(&p, &p, None).unwrap().abs() < 1e-12);
        let pq = wasserstein_1d_impl(&p, &q, None).unwrap();
        let qp = wasserstein_1d_impl(&q, &p, None).unwrap();
        assert!((pq - qp).abs() < 1e-12, "{pq} vs {qp}");
    }

    #[test]
    fn wasserstein_refuses_an_unordered_or_mismatched_support() {
        let p = [0.5, 0.5];
        let q = [0.5, 0.5];
        let err = wasserstein_1d_impl(&p, &q, Some(&[1.0, 1.0])).unwrap_err();
        assert_eq!(
            err,
            DistError::UnorderedSupport {
                index: 2,
                previous: 1.0,
                value: 1.0
            }
        );
        assert!(err.to_string().contains("strictly increasing"));
        // The support's length is blamed on the support, not on `q` — whose
        // length is fine.
        let len_err = wasserstein_1d_impl(&p, &q, Some(&[0.0, 1.0, 2.0])).unwrap_err();
        assert_eq!(
            len_err,
            DistError::SupportLengthMismatch {
                distribution: 2,
                support: 3
            }
        );
        assert!(len_err.to_string().contains("one position per bin"));
    }

    #[test]
    fn wasserstein_uses_each_interval_width_not_the_first() {
        // Unequal spacing: an implementation reusing x[1]-x[0] for every gap
        // would report 2 here instead of 10.
        let p = [1.0, 0.0, 0.0];
        let q = [0.0, 0.0, 1.0];
        let w = wasserstein_1d_impl(&p, &q, Some(&[0.0, 1.0, 10.0])).unwrap();
        assert!((w - 10.0).abs() < 1e-12, "got {w}");

        // And the mirror: wide first, narrow second.
        let w2 = wasserstein_1d_impl(&p, &q, Some(&[0.0, 9.0, 10.0])).unwrap();
        assert!((w2 - 10.0).abs() < 1e-12, "got {w2}");

        // Mass moved only across the narrow gap costs only that gap.
        let mid = [0.0, 1.0, 0.0];
        let narrow = wasserstein_1d_impl(&mid, &q, Some(&[0.0, 9.0, 10.0])).unwrap();
        assert!((narrow - 1.0).abs() < 1e-12, "got {narrow}");
    }

    #[test]
    fn hellinger_stays_accurate_as_the_distributions_converge() {
        // The `1 - BC` route cancels here: it returns 15x the true value at
        // 1e-9 and exactly zero at 1e-8. Summing the per-element differences
        // keeps each term at its own scale.
        for k in 6..=10 {
            let e = 10f64.powi(-k);
            let p = [0.5, 0.5];
            let q = [0.5 + e, 0.5 - e];
            let h = hellinger_impl(&p, &q).unwrap();
            // For p = [1/2, 1/2] and a shift of e the exact distance is e/sqrt(2)
            // to first order.
            let expected = e / 2f64.sqrt();
            let rel = (h - expected).abs() / expected;
            assert!(rel < 1e-6, "e=1e-{k}: got {h:e}, expected {expected:e}");
        }
    }

    #[test]
    fn mutual_information_is_zero_under_independence() {
        // p(x,y) = p(x)q(y) exactly, so knowing one says nothing about the other.
        let px = [0.3, 0.7];
        let qy = [0.4, 0.6];
        let joint: Vec<Vec<f64>> = px
            .iter()
            .map(|&a| qy.iter().map(|&b| a * b).collect())
            .collect();
        let mi = mutual_information_impl(&joint).unwrap();
        assert!(mi.abs() < 1e-12, "got {mi}");
    }

    #[test]
    fn mutual_information_of_a_perfect_dependence_is_the_marginal_entropy() {
        // Y is a copy of X: the diagonal carries everything, so I(X;Y) = H(X).
        let joint = vec![vec![0.25, 0.0], vec![0.0, 0.75]];
        let mi = mutual_information_impl(&joint).unwrap();
        let h_x = entropy_impl(&[0.25, 0.75]).unwrap();
        assert!((mi - h_x).abs() < 1e-12, "mi={mi}, H(X)={h_x}");
    }

    #[test]
    fn mutual_information_is_bounded_by_the_smaller_marginal_entropy() {
        let joint = vec![vec![0.1, 0.2, 0.05], vec![0.15, 0.3, 0.2]];
        let mi = mutual_information_impl(&joint).unwrap();
        let h_rows = entropy_impl(&[0.35, 0.65]).unwrap();
        let h_cols = entropy_impl(&[0.25, 0.5, 0.25]).unwrap();
        assert!(mi >= 0.0, "never negative: {mi}");
        assert!(
            mi <= h_rows.min(h_cols) + 1e-12,
            "mi={mi} exceeds the bound"
        );
    }

    #[test]
    fn mutual_information_is_unbiased_by_a_drifted_sum() {
        // An independent joint whose sum sits at the edge of the tolerance.
        // Without dividing through by the observed sum, the result would carry
        // a bias of roughly (1 - sum) — positive mutual information for
        // variables that are exactly independent.
        for cells in [4usize, 4096, 50176] {
            let rows = if cells == 4 { 2 } else { 64 };
            let cols = cells / rows;
            let tol = norm_tol(cells);
            // What remains is f64 accumulation over the cells, not the
            // normalization bias — which reached 2.1e-4 here before the sum
            // was divided out.
            let bound = 16.0 * cells as f64 * f64::EPSILON;
            for direction in [-0.5, 0.5] {
                let each = (1.0 + direction * tol) / cells as f64;
                let joint: Vec<Vec<f64>> = (0..rows).map(|_| vec![each; cols]).collect();
                let mi = mutual_information_impl(&joint).unwrap();
                assert!(
                    mi < bound,
                    "cells={cells} direction={direction}: independent joint reported \
                     mi={mi:e}, bound {bound:e}"
                );
            }
        }
    }

    #[test]
    fn mutual_information_of_a_degenerate_shape_is_zero() {
        // A single row or column means one variable is constant, so it carries
        // no information about the other.
        assert!(
            mutual_information_impl(&[vec![0.25, 0.25, 0.5]])
                .unwrap()
                .abs()
                < 1e-12
        );
        assert!(
            mutual_information_impl(&[vec![0.25], vec![0.25], vec![0.5]])
                .unwrap()
                .abs()
                < 1e-12
        );
        // 1x1: the whole mass in one cell.
        assert!(mutual_information_impl(&[vec![1.0]]).unwrap().abs() < 1e-12);
    }

    #[test]
    fn mutual_information_tolerates_an_all_zero_row_or_column() {
        // An outcome that never occurs has a zero marginal, which would divide
        // by zero if its cells were not already skipped as zero.
        let zero_row = vec![vec![0.5, 0.5], vec![0.0, 0.0]];
        let mi_row = mutual_information_impl(&zero_row).unwrap();
        assert!(mi_row.is_finite() && mi_row.abs() < 1e-12, "got {mi_row}");

        let zero_col = vec![vec![0.5, 0.0], vec![0.5, 0.0]];
        let mi_col = mutual_information_impl(&zero_col).unwrap();
        assert!(mi_col.is_finite() && mi_col.abs() < 1e-12, "got {mi_col}");
    }

    #[test]
    fn mutual_information_is_symmetric_under_transpose() {
        let joint = vec![vec![0.1, 0.2, 0.05], vec![0.15, 0.3, 0.2]];
        let transposed: Vec<Vec<f64>> = (0..3)
            .map(|j| (0..2).map(|i| joint[i][j]).collect())
            .collect();
        let a = mutual_information_impl(&joint).unwrap();
        let b = mutual_information_impl(&transposed).unwrap();
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn mutual_information_rejects_a_ragged_or_unnormalized_matrix() {
        let ragged = vec![vec![0.5, 0.5], vec![0.0]];
        let err = mutual_information_impl(&ragged).unwrap_err();
        assert_eq!(
            err,
            DistError::RaggedRow {
                row: 2,
                expected: 2,
                found: 1
            }
        );
        // The row is named, 1-based, as everything else in this module is.
        assert_eq!(
            err.to_string(),
            "joint[2] has 1 columns, expected 2 to match joint[1]"
        );
        let short = vec![vec![0.2, 0.2], vec![0.2, 0.2]];
        assert!(matches!(
            mutual_information_impl(&short).unwrap_err(),
            DistError::NotNormalized { side: "joint", .. }
        ));
        assert!(mutual_information_impl(&[]).is_err());
    }

    #[test]
    fn tvd_identical_is_zero() {
        let tvd = tvd_impl(&[0.25, 0.25, 0.5], &[0.25, 0.25, 0.5]).unwrap();
        assert!((tvd - 0.0).abs() < 1e-10);
    }

    #[test]
    fn tvd_disjoint_is_one() {
        let tvd = tvd_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((tvd - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tvd_symmetric_and_bounded() {
        let p = [0.7, 0.2, 0.1];
        let q = [0.1, 0.3, 0.6];
        let pq = tvd_impl(&p, &q).unwrap();
        let qp = tvd_impl(&q, &p).unwrap();
        assert!((pq - qp).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&pq));
    }

    #[test]
    fn tolerance_admits_f32_normalized_vocabulary() {
        // The fixed 1e-6 rejected these; the length-dependent bound admits them.
        for n in [4096usize, 32000, 50257] {
            let p = f32_normalized(n);
            let drift = (p.iter().sum::<f64>() - 1.0).abs();
            assert!(
                drift > 1e-6,
                "length {n} should drift past the old fixed tolerance, got {drift:e}"
            );
            entropy_impl(&p).unwrap_or_else(|e| panic!("length {n} rejected: {e}"));
        }
    }

    #[test]
    fn tolerance_is_tight_on_both_sides() {
        // What pins the bound: a sum short by twice the tolerance must fail and
        // one short by half of it must pass. A 5%-off distribution would be
        // caught by any tolerance and proves nothing about this one.
        for n in [4usize, 256, 4096, 50257] {
            let tol = norm_tol(n);
            let each = 1.0 / n as f64;
            // Scaling the whole vector is the shape a forgotten renormalization
            // takes; subtracting from one element goes negative once the
            // tolerance exceeds 1/n.
            let short_by = |missing: f64| -> Vec<f64> { vec![each * (1.0 - missing); n] };

            entropy_impl(&short_by(tol * 0.5))
                .unwrap_or_else(|e| panic!("n={n}: inside tol rejected: {e}"));

            assert!(
                entropy_impl(&short_by(tol * 2.0)).is_err(),
                "n={n}: a sum short by 2x the tolerance slipped through"
            );
        }
    }

    #[test]
    fn tolerance_scales_with_sqrt_of_length() {
        // Quadrupling the length doubles the bound.
        assert!((norm_tol(1024) / norm_tol(256) - 2.0).abs() < 1e-9);
        // And it stays far from letting a tenth of a percent go missing.
        assert!(norm_tol(50257) < 5e-4, "got {:e}", norm_tol(50257));
    }

    #[test]
    fn tvd_exceeds_one_when_both_inputs_drift() {
        // Documented consequence of admitting a sum of 1 ± tol: two one-hots that
        // each drift upward disagree on slightly more than the whole mass.
        let n = 4096;
        let tol = norm_tol(n);
        let mut p = vec![0.0; n];
        let mut q = vec![0.0; n];
        p[0] = 1.0 + tol * 0.5;
        q[1] = 1.0 + tol * 0.5;
        let v = tvd_impl(&p, &q).unwrap();
        assert!(v > 1.0 && v <= 1.0 + tol, "expected just above 1, got {v}");
    }

    #[test]
    fn error_names_the_offending_element() {
        // Index is 1-based: -0.1 is the second element of the Lua table.
        let err = entropy_impl(&[0.5, -0.1, 0.6]).unwrap_err();
        assert_eq!(
            err,
            DistError::Negative {
                side: "probs",
                index: 2,
                value: -0.1
            }
        );
        assert_eq!(err.to_string(), "probs[2] is negative: -0.1");
    }

    #[test]
    fn error_distinguishes_non_finite_from_a_bad_sum() {
        let err = entropy_impl(&[0.5, f64::NAN, 0.5]).unwrap_err();
        assert!(matches!(
            err,
            DistError::NonFinite {
                side: "probs",
                index: 2,
                ..
            }
        ));
    }

    #[test]
    fn error_names_the_side_of_a_pair() {
        let err = kl_divergence_impl(&[0.5, 0.5], &[0.5, 0.4]).unwrap_err();
        assert!(matches!(err, DistError::NotNormalized { side: "q", .. }));
    }

    #[test]
    fn error_reports_a_length_mismatch_before_contents() {
        let err = kl_divergence_impl(&[1.0], &[0.5, 0.5]).unwrap_err();
        assert_eq!(err, DistError::LengthMismatch { p: 1, q: 2 });
    }

    #[test]
    fn error_rejects_an_empty_distribution() {
        assert_eq!(
            entropy_impl(&[]).unwrap_err(),
            DistError::Empty { side: "probs" }
        );
    }
}
