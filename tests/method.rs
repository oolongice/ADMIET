use admiet::{
    fit, fit_many, integrate, trajectory_loss, windowed_trajectory_loss,
    windowed_trajectory_loss_adjoint, BinaryOperator, EvolutionSystem, FitOptions, Forest,
    Observations,
};
use std::sync::Arc;

#[derive(Debug)]
struct SumOfSquares;
impl BinaryOperator for SumOfSquares {
    fn name(&self) -> &'static str {
        "sum_of_squares"
    }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        (a * a + b * b, 2.0 * a, 2.0 * b)
    }
}

#[test]
fn custom_operator_and_sparse_dependencies_work() {
    let model = Forest::with_operators_and_dependencies(
        3,
        1,
        vec![Arc::new(SumOfSquares)],
        vec![vec![0], vec![1, 2], vec![2]],
        (-1.0, 1.0),
        2,
    )
    .unwrap();
    assert_eq!(model.parameters().len(), 35);
    assert!(model.equations()[0].contains("sum_of_squares"));
    let mut rhs = vec![0.0; 3];
    let mut jx = vec![0.0; 9];
    let mut jp = vec![0.0; 3 * model.parameters().len()];
    model.rhs_and_jacobians(&[0.2, 0.3, 0.4], &mut rhs, &mut jx, &mut jp);
    assert_eq!(jx[1], 0.0);
    assert_eq!(jx[2], 0.0);
    assert_eq!(jx[3], 0.0);
    assert_eq!(jx[6], 0.0);
    assert_eq!(jx[7], 0.0);
    let mut perturbed = vec![0.0; 3];
    model.rhs_and_jacobians(
        &[0.2 + 1e-6, 0.3, 0.4],
        &mut perturbed,
        &mut [0.0; 9],
        &mut vec![0.0; 3 * model.parameters().len()],
    );
    assert!(((perturbed[0] - rhs[0]) / 1e-6 - jx[0]).abs() < 1e-5);
    assert!(
        Forest::with_dependencies(3, 1, vec![vec![0], vec![1, 1], vec![2]], (-1.0, 1.0), 1)
            .is_err()
    );
}

struct CoupledGrid {
    rate: f64,
}
impl EvolutionSystem for CoupledGrid {
    fn state_dim(&self) -> usize {
        2
    }
    fn parameter_count(&self) -> usize {
        1
    }
    fn rhs_and_sensitivities(
        &self,
        _t: f64,
        x: &[f64],
        s: &[f64],
        rhs: &mut [f64],
        ds: &mut [f64],
    ) {
        // Two-cell diffusion: a minimal method-of-lines PDE example.
        rhs[0] = self.rate * (x[1] - x[0]);
        rhs[1] = self.rate * (x[0] - x[1]);
        ds[0] = x[1] - x[0] + self.rate * (s[1] - s[0]);
        ds[1] = x[0] - x[1] + self.rate * (s[0] - s[1]);
    }
}

#[test]
fn rk4_sensitivity_supports_coupled_spatial_states() {
    let rate = 0.4;
    let times = [0.0, 0.2, 0.7, 1.0];
    let (states, sens) = integrate(&CoupledGrid { rate }, &times, &[1.0, 0.0], 0.02).unwrap();
    for (k, &t) in times.iter().enumerate() {
        let decay = (-2.0 * rate * t).exp();
        assert!((states[k][0] - (1.0 + decay) / 2.0).abs() < 1e-7);
        assert!((states[k][1] - (1.0 - decay) / 2.0).abs() < 1e-7);
        assert!((sens[k][0] + t * decay).abs() < 1e-7);
        assert!((sens[k][1] - t * decay).abs() < 1e-7);
    }
}

#[test]
fn fit_reduces_synthetic_trajectory_loss() {
    let truth = Forest::new(1, 1, (-0.5, 0.5), 11).unwrap();
    let times = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let states = integrate(&truth, &times, &[0.2], 0.02).unwrap().0;
    let data = Observations { times, states };
    let mut candidate = Forest::new(1, 1, (-0.5, 0.5), 19).unwrap();
    let before = trajectory_loss(&candidate, &data, 0.02).unwrap().0;
    let report = fit(
        &mut candidate,
        &data,
        &FitOptions {
            epochs: 80,
            pretrain_epochs: 0,
            learning_rate: 0.01,
            max_step: 0.02,
            shooting_window: 4,
            one_hot_penalty: 0.0,
            print_every: 0,
        },
    )
    .unwrap();
    assert_eq!(report.initial_loss, before);
    assert!(
        report.final_loss < before * 0.1,
        "{} vs {}",
        report.final_loss,
        before
    );
}

#[test]
fn discrete_adjoint_agrees_with_forward_sensitivities() {
    let model = Forest::new(3, 2, (-0.5, 0.5), 3).unwrap();
    let times = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let states = times
        .iter()
        .map(|&t| vec![0.3 + t, -0.2 + t * t, 0.1 - t])
        .collect();
    let data = Observations { times, states };
    for window in [1, 2, 4] {
        let (loss_f, grad_f) = windowed_trajectory_loss(&model, &data, 0.04, window).unwrap();
        let (loss_a, grad_a) =
            windowed_trajectory_loss_adjoint(&model, &data, 0.04, window).unwrap();
        assert!((loss_f - loss_a).abs() < 1e-12);
        for (a, b) in grad_f.iter().zip(grad_a.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }
}

#[test]
fn sparse_six_state_model_fits_multiple_initial_conditions() {
    let dependencies = (0..6).map(|i| vec![i, (i + 1) % 6]).collect::<Vec<_>>();
    let truth = Forest::with_dependencies(6, 1, dependencies.clone(), (-0.5, 0.5), 8).unwrap();
    let times = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let data = (0..4)
        .map(|run| {
            let initial = (0..6)
                .map(|i| 0.2 + 0.15 * ((i + run * 2) % 6) as f64)
                .collect::<Vec<_>>();
            let states = integrate(&truth, &times, &initial, 0.05).unwrap().0;
            Observations {
                times: times.clone(),
                states,
            }
        })
        .collect::<Vec<_>>();
    let mut candidate = Forest::with_dependencies(6, 1, dependencies, (-0.5, 0.5), 21).unwrap();
    let report = fit_many(
        &mut candidate,
        &data,
        &FitOptions {
            epochs: 40,
            pretrain_epochs: 40,
            learning_rate: 0.01,
            max_step: 0.05,
            shooting_window: 1,
            one_hot_penalty: 0.0,
            print_every: 0,
        },
    )
    .unwrap();
    assert_eq!(report.epochs_completed, 80);
    assert!(
        report.final_loss < report.initial_loss * 0.2,
        "{} vs {}",
        report.final_loss,
        report.initial_loss
    );
}
