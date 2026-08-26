use mlua::prelude::*;

use crate::stats::table_to_vec;

/// Unit roundoff of `f32`: half an ulp at 1.0.
const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// Tolerance floor, kept so short distributions behave as they always have.
const TOL_FLOOR: f64 = 1e-6;

/// Absolute tolerance for the "sums to 1" check on a distribution of `n` elements.
///
/// Naive summation of `n` floats carries a forward error bound of
/// `(n-1) * u * Σ|x_i|`; with `Σ|x_i| = 1` and `u` the f32 unit roundoff that is
/// `(n-1) * 5.96e-8`. Callers normalize in f32 (a softmax output) and widen to
/// f64, so the check has to admit that drift — a 50257-entry vocabulary reaches
/// `1.3e-4` in practice, well past the `1e-6` this used to compare against.
fn norm_tol(n: usize) -> f64 {
    TOL_FLOOR.max(n.saturating_sub(1) as f64 * F32_UNIT_ROUNDOFF)
}

/// Why a slice is not a probability distribution.
///
/// Each variant names the offending side and index so the caller can find the
/// row rather than re-deriving it from a failed sum.
#[derive(Debug, PartialEq)]
enum DistError {
    /// Zero-length input: probability over an empty support is undefined.
    Empty { side: &'static str },
    /// A pairwise call received two distributions over different supports.
    LengthMismatch { p: usize, q: usize },
    /// An element was `NaN` or `±inf`.
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
    for (index, &value) in dist.iter().enumerate() {
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

/// Total variation distance: TVD(p, q) = 0.5 * Σ|p_i - q_i|
/// Symmetric and bounded [0, 1]: the share of probability mass the two
/// distributions disagree on.
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
                drift > TOL_FLOOR,
                "length {n} should drift past the old fixed tolerance, got {drift:e}"
            );
            entropy_impl(&p).unwrap_or_else(|e| panic!("length {n} rejected: {e}"));
        }
    }

    #[test]
    fn tolerance_still_catches_a_missing_renormalization() {
        // Masking out 5% of the mass and forgetting to renormalize stays an error
        // at every length.
        for n in [4usize, 4096, 50257] {
            let scaled: Vec<f64> = f32_normalized(n).iter().map(|x| x * 0.95).collect();
            assert!(entropy_impl(&scaled).is_err(), "length {n} slipped through");
        }
    }

    #[test]
    fn tolerance_floor_holds_for_short_vectors() {
        assert_eq!(norm_tol(1), TOL_FLOOR);
        assert_eq!(norm_tol(2), TOL_FLOOR);
        assert!(norm_tol(50257) > 1e-3);
    }

    #[test]
    fn error_names_the_offending_element() {
        let err = entropy_impl(&[0.5, -0.1, 0.6]).unwrap_err();
        assert_eq!(
            err,
            DistError::Negative {
                side: "probs",
                index: 1,
                value: -0.1
            }
        );
        assert_eq!(err.to_string(), "probs[1] is negative: -0.1");
    }

    #[test]
    fn error_distinguishes_non_finite_from_a_bad_sum() {
        let err = entropy_impl(&[0.5, f64::NAN, 0.5]).unwrap_err();
        assert!(matches!(
            err,
            DistError::NonFinite {
                side: "probs",
                index: 1,
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
