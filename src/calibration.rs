use mlua::prelude::*;

use crate::stats::table_to_vec;

/// Why a set of predictions cannot be scored for calibration.
///
/// Indices are 1-based, matching the Lua tables they come from.
#[derive(Debug, PartialEq)]
enum CalibrationError {
    Empty,
    LengthMismatch {
        confidences: usize,
        outcomes: usize,
    },
    /// A confidence outside `[0, 1]` — it is a probability, not a score.
    OutOfUnitRange {
        index: usize,
        value: f64,
    },
    /// An outcome that is neither 0 nor 1. Calibration is scored against what
    /// actually happened, so a partial outcome has no meaning here.
    NotBinary {
        index: usize,
        value: f64,
    },
    NoBins,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "expected at least one prediction"),
            Self::LengthMismatch {
                confidences,
                outcomes,
            } => write!(
                f,
                "confidences and outcomes must be paired: {confidences} vs {outcomes}"
            ),
            Self::OutOfUnitRange { index, value } => {
                write!(f, "confidences[{index}] is outside [0, 1]: {value}")
            }
            Self::NotBinary { index, value } => {
                write!(f, "outcomes[{index}] is neither 0 nor 1: {value}")
            }
            Self::NoBins => write!(f, "needs at least one bin"),
        }
    }
}

/// Check the shape both calibration measures require.
fn validate(confidences: &[f64], outcomes: &[f64]) -> Result<(), CalibrationError> {
    if confidences.len() != outcomes.len() {
        return Err(CalibrationError::LengthMismatch {
            confidences: confidences.len(),
            outcomes: outcomes.len(),
        });
    }
    if confidences.is_empty() {
        return Err(CalibrationError::Empty);
    }
    for (i, &c) in confidences.iter().enumerate() {
        if !c.is_finite() || !(0.0..=1.0).contains(&c) {
            return Err(CalibrationError::OutOfUnitRange {
                index: i + 1,
                value: c,
            });
        }
    }
    for (i, &o) in outcomes.iter().enumerate() {
        if o != 0.0 && o != 1.0 {
            return Err(CalibrationError::NotBinary {
                index: i + 1,
                value: o,
            });
        }
    }
    Ok(())
}

/// One equal-width confidence bin.
#[derive(Debug)]
struct Bin {
    count: usize,
    /// Mean confidence of the predictions that landed here.
    confidence: f64,
    /// Share of them that turned out right.
    accuracy: f64,
}

/// Expected and maximum calibration error over equal-width bins.
///
/// A model is calibrated when the predictions it makes at confidence `c` come
/// true about `c` of the time. ECE is the gap between those two, averaged over
/// the bins and weighted by how many predictions each holds; MCE is the widest
/// single gap. Both are in `[0, 1]` and zero only for perfect calibration.
///
/// # Why the bins are equal-width, and fixed
///
/// The number is only comparable against another number computed the same way:
/// equal-width and equal-frequency binning give different values for the same
/// predictions, as does a different bin count. This takes `bins` from the
/// caller but fixes the partition to equal width over `[0, 1]` — the standard
/// form [Guo et al. 2017]. An equal-frequency variant would be a separate
/// function rather than a flag, so a reported ECE always names its method.
///
/// Empty bins contribute nothing; a bin with no predictions has no gap to
/// measure rather than a gap of zero. `bins_used` reports how many were
/// actually occupied, which is what says whether the partition suited the data.
fn calibration_error_impl(
    confidences: &[f64],
    outcomes: &[f64],
    bins: usize,
) -> Result<(f64, f64, Vec<Bin>), CalibrationError> {
    validate(confidences, outcomes)?;
    if bins == 0 {
        return Err(CalibrationError::NoBins);
    }

    let mut counts = vec![0usize; bins];
    let mut conf_sums = vec![0.0; bins];
    let mut hits = vec![0.0; bins];
    for (&c, &o) in confidences.iter().zip(outcomes.iter()) {
        // The top edge belongs to the last bin: c = 1.0 would otherwise index
        // one past the end.
        let idx = ((c * bins as f64) as usize).min(bins - 1);
        counts[idx] += 1;
        conf_sums[idx] += c;
        hits[idx] += o;
    }

    let n = confidences.len() as f64;
    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    let mut out = Vec::with_capacity(bins);
    for b in 0..bins {
        let count = counts[b];
        if count == 0 {
            out.push(Bin {
                count: 0,
                confidence: 0.0,
                accuracy: 0.0,
            });
            continue;
        }
        let cf = count as f64;
        let confidence = conf_sums[b] / cf;
        let accuracy = hits[b] / cf;
        let gap = (accuracy - confidence).abs();
        ece += cf / n * gap;
        mce = mce.max(gap);
        out.push(Bin {
            count,
            confidence,
            accuracy,
        });
    }
    Ok((ece, mce, out))
}

/// Brier score: the mean squared error of a probabilistic prediction.
///
/// `(1/N) Σ (p_i - o_i)²`, in `[0, 1]`, lower being better. Unlike ECE it is a
/// *proper scoring rule* — it is minimized only by predicting the true
/// probability, so it cannot be gamed by a model that hedges toward the base
/// rate. It also does not need binning, and so does not inherit the arbitrary
/// choice ECE makes.
///
/// The two measure different things and are worth reading together: a model
/// can be well calibrated (low ECE) while barely discriminating between cases
/// (high Brier), if it predicts the base rate for everything.
///
/// Binary outcomes only. The multi-class form sums over classes and belongs to
/// a separate function.
fn brier_score_impl(confidences: &[f64], outcomes: &[f64]) -> Result<f64, CalibrationError> {
    validate(confidences, outcomes)?;
    let n = confidences.len() as f64;
    let sum: f64 = confidences
        .iter()
        .zip(outcomes.iter())
        .map(|(&p, &o)| (p - o) * (p - o))
        .sum();
    Ok(sum / n)
}

pub(crate) fn register(lua: &Lua, t: &LuaTable) -> LuaResult<()> {
    // calibration_error(confidences, outcomes, bins)
    t.set(
        "calibration_error",
        lua.create_function(|lua, (conf_t, out_t, bins): (LuaTable, LuaTable, usize)| {
            let confidences = table_to_vec(&conf_t)?;
            let outcomes = table_to_vec(&out_t)?;
            let (ece, mce, bin_list) = calibration_error_impl(&confidences, &outcomes, bins)
                .map_err(|e| LuaError::runtime(format!("calibration_error: {e}")))?;

            let result = lua.create_table()?;
            result.set("ece", ece)?;
            result.set("mce", mce)?;
            let bins_t = lua.create_table()?;
            let mut occupied = 0usize;
            for (i, b) in bin_list.iter().enumerate() {
                let entry = lua.create_table()?;
                entry.set("count", b.count)?;
                entry.set("confidence", b.confidence)?;
                entry.set("accuracy", b.accuracy)?;
                bins_t.set(i + 1, entry)?;
                if b.count > 0 {
                    occupied += 1;
                }
            }
            result.set("bins", bins_t)?;
            result.set("bins_used", occupied)?;
            Ok(result)
        })?,
    )?;

    t.set(
        "brier_score",
        lua.create_function(|_, (conf_t, out_t): (LuaTable, LuaTable)| {
            let confidences = table_to_vec(&conf_t)?;
            let outcomes = table_to_vec(&out_t)?;
            brier_score_impl(&confidences, &outcomes)
                .map_err(|e| LuaError::runtime(format!("brier_score: {e}")))
        })?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_perfectly_calibrated_model_scores_zero() {
        // Every prediction at 0.0 fails and every prediction at 1.0 succeeds:
        // confidence matches accuracy in each occupied bin.
        let conf = [0.0, 0.0, 1.0, 1.0];
        let out = [0.0, 0.0, 1.0, 1.0];
        let (ece, mce, _) = calibration_error_impl(&conf, &out, 10).unwrap();
        assert!(ece.abs() < 1e-12, "ece={ece}");
        assert!(mce.abs() < 1e-12, "mce={mce}");
    }

    #[test]
    fn a_maximally_overconfident_model_scores_one() {
        // Certain every time, wrong every time.
        let conf = [1.0, 1.0, 1.0, 1.0];
        let out = [0.0, 0.0, 0.0, 0.0];
        let (ece, mce, _) = calibration_error_impl(&conf, &out, 10).unwrap();
        assert!((ece - 1.0).abs() < 1e-12, "ece={ece}");
        assert!((mce - 1.0).abs() < 1e-12, "mce={mce}");
    }

    #[test]
    fn ece_weights_bins_by_population_and_mce_does_not() {
        // Bin A: 98 predictions at 0.5, half right -> gap 0.
        // Bin B: 2 predictions at 0.95, both wrong -> gap 0.95.
        // ECE is dominated by the crowded bin; MCE reports the worst one.
        let mut conf = vec![0.5; 98];
        let mut out: Vec<f64> = (0..98).map(|i| (i % 2) as f64).collect();
        conf.extend([0.95, 0.95]);
        out.extend([0.0, 0.0]);

        let (ece, mce, _) = calibration_error_impl(&conf, &out, 10).unwrap();
        assert!(ece < 0.03, "the small bin barely moves ECE: {ece}");
        assert!((mce - 0.95).abs() < 1e-12, "MCE sees it in full: {mce}");
    }

    #[test]
    fn the_top_edge_lands_in_the_last_bin() {
        // c = 1.0 * bins would index one past the end without the clamp.
        let (_, _, bins) = calibration_error_impl(&[1.0], &[1.0], 4).unwrap();
        assert_eq!(bins[3].count, 1, "1.0 belongs to the last bin");
        assert_eq!(bins[0].count + bins[1].count + bins[2].count, 0);
    }

    #[test]
    fn empty_bins_contribute_nothing() {
        // Two predictions at opposite ends of 10 bins: 8 bins stay empty and
        // must not be read as perfectly calibrated.
        let (ece, _, bins) = calibration_error_impl(&[0.05, 0.95], &[0.0, 1.0], 10).unwrap();
        let occupied = bins.iter().filter(|b| b.count > 0).count();
        assert_eq!(occupied, 2);
        // |0 - 0.05| and |1 - 0.95|, each weighted 1/2.
        assert!((ece - 0.05).abs() < 1e-12, "ece={ece}");
    }

    #[test]
    fn bin_count_changes_the_number() {
        // The same predictions scored over different partitions do not agree —
        // which is why the bin count is part of what an ECE reports.
        let conf: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let out: Vec<f64> = (0..100).map(|i| if i > 50 { 1.0 } else { 0.0 }).collect();
        let coarse = calibration_error_impl(&conf, &out, 2).unwrap().0;
        let fine = calibration_error_impl(&conf, &out, 50).unwrap().0;
        assert!(
            (coarse - fine).abs() > 1e-6,
            "coarse={coarse} fine={fine} — binning is not neutral"
        );
    }

    #[test]
    fn calibration_refuses_a_malformed_input() {
        assert_eq!(
            calibration_error_impl(&[0.5, 1.5], &[0.0, 1.0], 10).unwrap_err(),
            CalibrationError::OutOfUnitRange {
                index: 2,
                value: 1.5
            }
        );
        assert_eq!(
            calibration_error_impl(&[0.5, 0.5], &[0.0, 0.5], 10).unwrap_err(),
            CalibrationError::NotBinary {
                index: 2,
                value: 0.5
            }
        );
        assert!(matches!(
            calibration_error_impl(&[0.5], &[0.0, 1.0], 10).unwrap_err(),
            CalibrationError::LengthMismatch { .. }
        ));
        assert_eq!(
            calibration_error_impl(&[0.5], &[1.0], 0).unwrap_err(),
            CalibrationError::NoBins
        );
        assert_eq!(
            calibration_error_impl(&[], &[], 10).unwrap_err(),
            CalibrationError::Empty
        );
    }

    #[test]
    fn brier_is_zero_for_certain_and_correct_predictions() {
        assert!(
            brier_score_impl(&[1.0, 0.0, 1.0], &[1.0, 0.0, 1.0])
                .unwrap()
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn brier_is_one_for_certain_and_wrong_predictions() {
        let b = brier_score_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((b - 1.0).abs() < 1e-12, "got {b}");
    }

    #[test]
    fn brier_punishes_confidence_more_than_hedging() {
        // Same direction of error, different certainty. The squared term is
        // what makes it a proper scoring rule.
        let hedged = brier_score_impl(&[0.6, 0.6], &[0.0, 0.0]).unwrap();
        let certain = brier_score_impl(&[0.9, 0.9], &[0.0, 0.0]).unwrap();
        assert!(certain > hedged, "hedged={hedged} certain={certain}");
        // Predicting 0.5 throughout scores 0.25 regardless of outcome.
        let coin = brier_score_impl(&[0.5, 0.5], &[0.0, 1.0]).unwrap();
        assert!((coin - 0.25).abs() < 1e-12, "got {coin}");
    }

    #[test]
    fn calibration_and_brier_disagree_on_a_base_rate_predictor() {
        // A model that predicts the base rate for everything is perfectly
        // calibrated and useless. ECE sees nothing wrong; Brier does.
        let conf = vec![0.5; 100];
        let out: Vec<f64> = (0..100).map(|i| (i % 2) as f64).collect();
        let (ece, _, _) = calibration_error_impl(&conf, &out, 10).unwrap();
        let brier = brier_score_impl(&conf, &out).unwrap();
        assert!(ece.abs() < 1e-12, "calibrated: {ece}");
        assert!((brier - 0.25).abs() < 1e-12, "but uninformative: {brier}");
    }
}
