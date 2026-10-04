use crate::Result;

/// A finite dimensional evolution equation. For a PDE, flatten a spatial grid into
/// `state` and implement its local stencil in `rhs_and_sensitivities`.
/// `sensitivity` and `sensitivity_dot` are row-major: state component first,
/// parameter second. The method computes S' = J_state S + J_parameter, and may
/// apply J_state without constructing a dense matrix.
pub trait EvolutionSystem {
    fn state_dim(&self) -> usize;
    fn parameter_count(&self) -> usize;
    fn rhs_and_sensitivities(
        &self,
        t: f64,
        state: &[f64],
        sensitivity: &[f64],
        rhs: &mut [f64],
        sensitivity_dot: &mut [f64],
    );
}

#[derive(Clone, Debug)]
pub struct Observations {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
}

pub type IntegratedTrajectory = (Vec<Vec<f64>>, Vec<Vec<f64>>);
impl Observations {
    pub fn validate(&self, dimension: usize) -> Result<()> {
        if self.times.len() < 2
            || self.times.len() != self.states.len()
            || self.times.iter().any(|v| !v.is_finite())
            || self.times.windows(2).any(|w| w[1] <= w[0])
            || self
                .states
                .iter()
                .any(|s| s.len() != dimension || s.iter().any(|v| !v.is_finite()))
        {
            return Err(
                "observations need matching finite states and strictly increasing times".into(),
            );
        }
        Ok(())
    }
}

fn derivative<S: EvolutionSystem>(system: &S, t: f64, augmented: &[f64]) -> Result<Vec<f64>> {
    let n = system.state_dim();
    let p = system.parameter_count();
    let mut rhs = vec![0.0; n];
    let mut ds = vec![0.0; n * p];
    system.rhs_and_sensitivities(t, &augmented[..n], &augmented[n..], &mut rhs, &mut ds);
    let mut result = vec![0.0; augmented.len()];
    result[..n].copy_from_slice(&rhs);
    result[n..].copy_from_slice(&ds);
    if result.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite dynamics at t={t}"));
    }
    Ok(result)
}

fn rk4_step<S: EvolutionSystem>(system: &S, t: f64, h: f64, y: &mut [f64]) -> Result<()> {
    let k1 = derivative(system, t, y)?;
    let mut stage = y
        .iter()
        .zip(&k1)
        .map(|(v, k)| v + h * k / 2.0)
        .collect::<Vec<_>>();
    let k2 = derivative(system, t + h / 2.0, &stage)?;
    for i in 0..y.len() {
        stage[i] = y[i] + h * k2[i] / 2.0;
    }
    let k3 = derivative(system, t + h / 2.0, &stage)?;
    for i in 0..y.len() {
        stage[i] = y[i] + h * k3[i];
    }
    let k4 = derivative(system, t + h, &stage)?;
    for i in 0..y.len() {
        y[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite trajectory at t={}", t + h));
    }
    Ok(())
}

/// Integrate through every observation time with a maximum internal RK4 step.
/// Returns states and sensitivities d(state)/d(parameter), each row-major.
pub fn integrate<S: EvolutionSystem>(
    system: &S,
    times: &[f64],
    initial: &[f64],
    max_step: f64,
) -> Result<IntegratedTrajectory> {
    let n = system.state_dim();
    let p = system.parameter_count();
    if times.is_empty()
        || initial.len() != n
        || !max_step.is_finite()
        || max_step <= 0.0
        || times.iter().any(|v| !v.is_finite())
        || times.windows(2).any(|w| w[1] <= w[0])
        || initial.iter().any(|v| !v.is_finite())
    {
        return Err("invalid integration inputs".into());
    }
    let mut y = vec![0.0; n + n * p];
    y[..n].copy_from_slice(initial);
    let mut states = vec![initial.to_vec()];
    let mut sensitivities = vec![vec![0.0; n * p]];
    for window in times.windows(2) {
        let steps = ((window[1] - window[0]) / max_step).ceil() as usize;
        if steps == 0 || steps > 1_000_000 {
            return Err("integration interval needs too many steps".into());
        }
        let h = (window[1] - window[0]) / steps as f64;
        for k in 0..steps {
            rk4_step(system, window[0] + k as f64 * h, h, &mut y)?;
        }
        states.push(y[..n].to_vec());
        sensitivities.push(y[n..].to_vec());
    }
    Ok((states, sensitivities))
}

/// Mean squared error over observed times after the initial condition.
pub fn trajectory_loss<S: EvolutionSystem>(
    system: &S,
    data: &Observations,
    max_step: f64,
) -> Result<(f64, Vec<f64>)> {
    windowed_trajectory_loss(system, data, max_step, data.times.len().saturating_sub(1))
}

/// Multiple shooting: each window restarts from its observed state.
/// A full-length window gives the original trajectory objective.
pub fn windowed_trajectory_loss<S: EvolutionSystem>(
    system: &S,
    data: &Observations,
    max_step: f64,
    window: usize,
) -> Result<(f64, Vec<f64>)> {
    data.validate(system.state_dim())?;
    if window == 0 {
        return Err("shooting window must be positive".into());
    }
    let p = system.parameter_count();
    let mut loss = 0.0;
    let mut gradient = vec![0.0; p];
    let scale = 1.0 / ((data.times.len() - 1) * system.state_dim()) as f64;
    for start in (0..data.times.len() - 1).step_by(window) {
        let end = (start + window).min(data.times.len() - 1);
        let (states, sensitivities) = integrate(
            system,
            &data.times[start..=end],
            &data.states[start],
            max_step,
        )?;
        for k in 1..states.len() {
            for i in 0..system.state_dim() {
                let residual = states[k][i] - data.states[start + k][i];
                loss += scale * residual * residual;
                for q in 0..p {
                    gradient[q] += 2.0 * scale * residual * sensitivities[k][i * p + q];
                }
            }
        }
    }
    if !loss.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite loss or gradient".into());
    }
    Ok((loss, gradient))
}
