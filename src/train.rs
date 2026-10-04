use crate::{vector_field_loss, windowed_trajectory_loss_adjoint, Forest, Observations, Result};

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub epochs: usize,
    pub pretrain_epochs: usize,
    pub learning_rate: f64,
    pub max_step: f64,
    pub shooting_window: usize,
    pub one_hot_penalty: f64,
    pub print_every: usize,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            epochs: 1000,
            pretrain_epochs: 0,
            learning_rate: 0.01,
            max_step: 0.05,
            shooting_window: 1,
            one_hot_penalty: 0.0,
            print_every: 100,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FitReport {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub epochs_completed: usize,
}

struct Adam {
    first: Vec<f64>,
    second: Vec<f64>,
    beta1_power: f64,
    beta2_power: f64,
}
impl Adam {
    fn new(p: usize) -> Self {
        Self {
            first: vec![0.0; p],
            second: vec![0.0; p],
            beta1_power: 1.0,
            beta2_power: 1.0,
        }
    }
    fn update(
        &mut self,
        model: &mut Forest,
        mut gradient: Vec<f64>,
        options: &FitOptions,
    ) -> Result<()> {
        for simplex in &model.simplexes {
            let range = simplex.start..simplex.start + simplex.len;
            for (g, &w) in gradient[range.clone()]
                .iter_mut()
                .zip(&model.parameters[range])
            {
                *g -= 2.0 * options.one_hot_penalty * w;
            }
        }
        self.beta1_power *= 0.9;
        self.beta2_power *= 0.999;
        let mut next = model.parameters().to_vec();
        for q in 0..next.len() {
            self.first[q] = 0.9 * self.first[q] + 0.1 * gradient[q];
            self.second[q] = 0.999 * self.second[q] + 0.001 * gradient[q] * gradient[q];
            let m = self.first[q] / (1.0 - self.beta1_power);
            let v = self.second[q] / (1.0 - self.beta2_power);
            next[q] -= options.learning_rate * m / (v.sqrt() + 1e-8);
        }
        model.set_parameters(&next)
    }
}

fn mean_objective<F>(model: &Forest, data: &[Observations], evaluate: F) -> Result<(f64, Vec<f64>)>
where
    F: Fn(&Forest, &Observations) -> Result<(f64, Vec<f64>)>,
{
    let mut loss = 0.0;
    let mut gradient = vec![0.0; model.parameters().len()];
    for trajectory in data {
        let (value, local_gradient) = evaluate(model, trajectory)?;
        loss += value / data.len() as f64;
        for (g, local) in gradient.iter_mut().zip(local_gradient) {
            *g += local / data.len() as f64;
        }
    }
    Ok((loss, gradient))
}

pub fn fit(model: &mut Forest, data: &Observations, options: &FitOptions) -> Result<FitReport> {
    fit_many(model, std::slice::from_ref(data), options)
}

/// Fit several independent trajectories with different initial conditions.
/// Each trajectory has equal weight in the objective.
pub fn fit_many(
    model: &mut Forest,
    data: &[Observations],
    options: &FitOptions,
) -> Result<FitReport> {
    if options.epochs == 0
        || data.is_empty()
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || !options.one_hot_penalty.is_finite()
        || options.one_hot_penalty < 0.0
        || options.shooting_window == 0
    {
        return Err("invalid training options".into());
    }
    let p = model.parameters().len();
    let objective = |model: &Forest| {
        mean_objective(model, data, |m, d| {
            windowed_trajectory_loss_adjoint(m, d, options.max_step, options.shooting_window)
        })
    };
    let pretrain_objective = |model: &Forest| mean_objective(model, data, vector_field_loss);
    let initial_loss = objective(model)?.0;
    let mut adam = Adam::new(p);
    for epoch in 1..=options.pretrain_epochs {
        let (_, gradient) = pretrain_objective(model)?;
        adam.update(model, gradient, options)?;
        if options.print_every > 0 && epoch % options.print_every == 0 {
            eprintln!(
                "pretrain epoch {epoch}: vector-field loss={:.8}",
                pretrain_objective(model)?.0
            );
        }
    }
    adam = Adam::new(p);
    for epoch in 1..=options.epochs {
        let (_, gradient) = objective(model)?;
        adam.update(model, gradient, options)?;
        if options.print_every > 0 && epoch % options.print_every == 0 {
            eprintln!("epoch {epoch}: loss={:.8}", objective(model)?.0);
        }
    }
    let final_loss = objective(model)?.0;
    Ok(FitReport {
        initial_loss,
        final_loss,
        epochs_completed: options.epochs + options.pretrain_epochs,
    })
}
