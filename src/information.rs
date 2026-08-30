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
}

impl std::fmt::Display for DistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty { side } => write!(f, "{side} is empty"),
            Self::LengthMismatch { p, q } => {
                write!(f, "length mismatch: p has {p}, q has {q}")
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
            return Err(DistError::LengthMismatch {
                p: cols,
                q: row.len(),
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

    let row_marginals: Vec<f64> = joint.iter().map(|r| r.iter().sum()).collect();
    let col_marginals: Vec<f64> = (0..cols)
        .map(|j| joint.iter().map(|r| r[j]).sum())
        .collect();

    let mut acc = 0.0;
    for (i, row) in joint.iter().enumerate() {
        for (j, &p_xy) in row.iter().enumerate() {
            if p_xy > 0.0 {
                // Both marginals are >= p_xy > 0 here, so neither divides by zero.
                acc += p_xy * (p_xy / (row_marginals[i] * col_marginals[j])).ln();
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
        assert!(matches!(
            mutual_information_impl(&ragged).unwrap_err(),
            DistError::LengthMismatch { .. }
        ));
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
