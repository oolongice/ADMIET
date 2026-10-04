use crate::{Observations, Result};

/// Reverse-mode interface for trajectory fitting. PDE implementations can
/// apply a spatial stencil transpose in `rhs_pullback` without a dense Jacobian.
pub trait AdjointSystem {
    fn state_dim(&self) -> usize;
    fn parameter_count(&self) -> usize;
    fn rhs(&self, t: f64, state: &[f64], output: &mut [f64]);
    /// Add J_state^T cotangent and J_parameter^T cotangent to the output buffers.
    fn rhs_pullback(
        &self,
        t: f64,
        state: &[f64],
        cotangent: &[f64],
        state_grad: &mut [f64],
        parameter_grad: &mut [f64],
    );
}

/// Fit the vector field to interval secants at observed midpoints. Useful as
/// an optional warm start; noisy observations can make these targets unreliable.
pub fn vector_field_loss<S: AdjointSystem>(
    system: &S,
    data: &Observations,
) -> Result<(f64, Vec<f64>)> {
    data.validate(system.state_dim())?;
    let n = system.state_dim();
    let mut gradient = vec![0.0; system.parameter_count()];
    let mut loss = 0.0;
    let scale = 1.0 / ((data.times.len() - 1) * n) as f64;
    for k in 0..data.times.len() - 1 {
        let dt = data.times[k + 1] - data.times[k];
        let midpoint = (0..n)
            .map(|i| (data.states[k][i] + data.states[k + 1][i]) / 2.0)
            .collect::<Vec<_>>();
        let rhs = checked_rhs(system, (data.times[k] + data.times[k + 1]) / 2.0, &midpoint)?;
        let mut cotangent = vec![0.0; n];
        for i in 0..n {
            let residual = rhs[i] - (data.states[k + 1][i] - data.states[k][i]) / dt;
            loss += scale * residual * residual;
            cotangent[i] = 2.0 * scale * residual;
        }
        system.rhs_pullback(
            (data.times[k] + data.times[k + 1]) / 2.0,
            &midpoint,
            &cotangent,
            &mut vec![0.0; n],
            &mut gradient,
        );
    }
    if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite vector-field loss or gradient".into());
    }
    Ok((loss, gradient))
}

struct Step {
    t: f64,
    h: f64,
    stages: [Vec<f64>; 4],
}

struct ReverseScratch {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    z: Vec<f64>,
}
impl ReverseScratch {
    fn new(n: usize) -> Self {
        Self {
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            z: vec![0.0; n],
        }
    }
}

fn checked_rhs<S: AdjointSystem>(system: &S, t: f64, state: &[f64]) -> Result<Vec<f64>> {
    let mut result = vec![0.0; state.len()];
    system.rhs(t, state, &mut result);
    if result.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite dynamics at t={t}"));
    }
    Ok(result)
}

fn forward_step<S: AdjointSystem>(system: &S, t: f64, h: f64, y: &mut [f64]) -> Result<Step> {
    let z1 = y.to_vec();
    let k1 = checked_rhs(system, t, &z1)?;
    let z2 = y
        .iter()
        .zip(&k1)
        .map(|(v, k)| v + h * k / 2.0)
        .collect::<Vec<_>>();
    let k2 = checked_rhs(system, t + h / 2.0, &z2)?;
    let z3 = y
        .iter()
        .zip(&k2)
        .map(|(v, k)| v + h * k / 2.0)
        .collect::<Vec<_>>();
    let k3 = checked_rhs(system, t + h / 2.0, &z3)?;
    let z4 = y
        .iter()
        .zip(&k3)
        .map(|(v, k)| v + h * k)
        .collect::<Vec<_>>();
    let k4 = checked_rhs(system, t + h, &z4)?;
    for i in 0..y.len() {
        y[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite trajectory at t={}", t + h));
    }
    Ok(Step {
        t,
        h,
        stages: [z1, z2, z3, z4],
    })
}

fn reverse_step<S: AdjointSystem>(
    system: &S,
    step: &Step,
    lambda: &mut [f64],
    gradient: &mut [f64],
    scratch: &mut ReverseScratch,
) {
    let h = step.h;
    for (i, &value) in lambda.iter().enumerate() {
        scratch.k1[i] = h * value / 6.0;
        scratch.k2[i] = h * value / 3.0;
        scratch.k3[i] = scratch.k2[i];
        scratch.k4[i] = scratch.k1[i];
    }
    scratch.z.fill(0.0);
    system.rhs_pullback(
        step.t + h,
        &step.stages[3],
        &scratch.k4,
        &mut scratch.z,
        gradient,
    );
    for (i, value) in lambda.iter_mut().enumerate() {
        *value += scratch.z[i];
        scratch.k3[i] += h * scratch.z[i];
    }
    scratch.z.fill(0.0);
    system.rhs_pullback(
        step.t + h / 2.0,
        &step.stages[2],
        &scratch.k3,
        &mut scratch.z,
        gradient,
    );
    for (i, value) in lambda.iter_mut().enumerate() {
        *value += scratch.z[i];
        scratch.k2[i] += h * scratch.z[i] / 2.0;
    }
    scratch.z.fill(0.0);
    system.rhs_pullback(
        step.t + h / 2.0,
        &step.stages[1],
        &scratch.k2,
        &mut scratch.z,
        gradient,
    );
    for (i, value) in lambda.iter_mut().enumerate() {
        *value += scratch.z[i];
        scratch.k1[i] += h * scratch.z[i] / 2.0;
    }
    scratch.z.fill(0.0);
    system.rhs_pullback(
        step.t,
        &step.stages[0],
        &scratch.k1,
        &mut scratch.z,
        gradient,
    );
    for (i, value) in lambda.iter_mut().enumerate() {
        *value += scratch.z[i];
    }
}

/// Exact discrete adjoint of the RK4 shooting objective.
pub fn windowed_trajectory_loss_adjoint<S: AdjointSystem>(
    system: &S,
    data: &Observations,
    max_step: f64,
    window: usize,
) -> Result<(f64, Vec<f64>)> {
    data.validate(system.state_dim())?;
    if window == 0 || !max_step.is_finite() || max_step <= 0.0 {
        return Err("invalid shooting window or step".into());
    }
    let n = system.state_dim();
    let mut gradient = vec![0.0; system.parameter_count()];
    let mut reverse_scratch = ReverseScratch::new(n);
    let mut loss = 0.0;
    let scale = 1.0 / ((data.times.len() - 1) * n) as f64;
    for start in (0..data.times.len() - 1).step_by(window) {
        let end = (start + window).min(data.times.len() - 1);
        let mut y = data.states[start].clone();
        let mut steps = Vec::new();
        let mut checkpoints = vec![0usize];
        let mut predicted = vec![y.clone()];
        for obs in start..end {
            let count = ((data.times[obs + 1] - data.times[obs]) / max_step).ceil() as usize;
            if count == 0 || count > 1_000_000 {
                return Err("integration interval needs too many steps".into());
            }
            let h = (data.times[obs + 1] - data.times[obs]) / count as f64;
            for k in 0..count {
                steps.push(forward_step(
                    system,
                    data.times[obs] + k as f64 * h,
                    h,
                    &mut y,
                )?);
            }
            checkpoints.push(steps.len());
            predicted.push(y.clone());
        }
        let mut lambda = vec![0.0; n];
        for obs in (1..predicted.len()).rev() {
            for i in 0..n {
                let residual = predicted[obs][i] - data.states[start + obs][i];
                loss += scale * residual * residual;
                lambda[i] += 2.0 * scale * residual;
            }
            for k in (checkpoints[obs - 1]..checkpoints[obs]).rev() {
                reverse_step(
                    system,
                    &steps[k],
                    &mut lambda,
                    &mut gradient,
                    &mut reverse_scratch,
                );
            }
        }
    }
    if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite loss or gradient".into());
    }
    Ok((loss, gradient))
}
