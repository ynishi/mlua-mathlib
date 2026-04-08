use mlua::prelude::*;

use crate::stats::table_to_vec;

/// Shannon entropy: H(p) = -Σ p_i * ln(p_i)
/// Input must be a valid probability distribution (non-negative, sums to ~1).
fn entropy_impl(probs: &[f64]) -> Result<f64, &'static str> {
    let sum: f64 = probs.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err("probabilities must sum to 1");
    }
    for &p in probs {
        if p < 0.0 {
            return Err("probabilities must be non-negative");
        }
    }
    let h: f64 = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    Ok(h)
}

/// KL divergence: D_KL(p || q) = Σ p_i * ln(p_i / q_i)
fn kl_divergence_impl(p: &[f64], q: &[f64]) -> Result<f64, &'static str> {
    if p.len() != q.len() {
        return Err("distributions must have equal length");
    }
    let sum_p: f64 = p.iter().sum();
    if (sum_p - 1.0).abs() > 1e-6 {
        return Err("p must sum to 1");
    }
    let sum_q: f64 = q.iter().sum();
    if (sum_q - 1.0).abs() > 1e-6 {
        return Err("q must sum to 1");
    }
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi < 0.0 || qi < 0.0 {
            return Err("probabilities must be non-negative");
        }
        if pi > 0.0 && qi == 0.0 {
            return Err("q must be > 0 wherever p > 0 (absolute continuity)");
        }
    }
    let kl: f64 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum();
    Ok(kl)
}

/// Jensen-Shannon divergence: JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
/// where m = 0.5 * (p + q). Always finite, symmetric, bounded [0, ln(2)].
fn js_divergence_impl(p: &[f64], q: &[f64]) -> Result<f64, &'static str> {
    if p.len() != q.len() {
        return Err("distributions must have equal length");
    }
    let sum_p: f64 = p.iter().sum();
    if (sum_p - 1.0).abs() > 1e-6 {
        return Err("p must sum to 1");
    }
    let sum_q: f64 = q.iter().sum();
    if (sum_q - 1.0).abs() > 1e-6 {
        return Err("q must sum to 1");
    }
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
fn cross_entropy_impl(p: &[f64], q: &[f64]) -> Result<f64, &'static str> {
    if p.len() != q.len() {
        return Err("distributions must have equal length");
    }
    let sum_p: f64 = p.iter().sum();
    if (sum_p - 1.0).abs() > 1e-6 {
        return Err("p must sum to 1");
    }
    let sum_q: f64 = q.iter().sum();
    if (sum_q - 1.0).abs() > 1e-6 {
        return Err("q must sum to 1");
    }
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi < 0.0 || qi < 0.0 {
            return Err("probabilities must be non-negative");
        }
        if pi > 0.0 && qi == 0.0 {
            return Err("q must be > 0 wherever p > 0");
        }
    }
    let ce: f64 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| -pi * qi.ln())
        .sum();
    Ok(ce)
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
