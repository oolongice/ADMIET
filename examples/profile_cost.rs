use admiet::{windowed_trajectory_loss_adjoint, AdjointSystem, Forest, Observations};
use std::hint::black_box;
use std::time::Instant;

fn median_ms(mut sample: impl FnMut(), repetitions: usize) -> f64 {
    let mut measurements = Vec::new();
    for _ in 0..9 {
        let start = Instant::now();
        for _ in 0..repetitions {
            sample();
        }
        measurements.push(start.elapsed().as_secs_f64() * 1000.0 / repetitions as f64);
    }
    measurements.sort_by(|a, b| a.total_cmp(b));
    measurements[4]
}

fn main() {
    for n in [8, 16] {
        let model = Forest::new(n, 1, (-1.0, 1.0), 1).unwrap();
        let state = vec![0.1; n];
        let cotangent = vec![0.01; n];
        let mut rhs = vec![0.0; n];
        let mut dx = vec![0.0; n];
        let mut dp = vec![0.0; model.parameters().len()];
        let data = Observations {
            times: (0..11).map(|k| k as f64 * 0.1).collect(),
            states: vec![state.clone(); 11],
        };
        // The objective uses 20 RK4 steps: 80 RHS calls and 80 pullbacks.
        let objective = median_ms(
            || {
                black_box(windowed_trajectory_loss_adjoint(&model, &data, 0.05, 1).unwrap());
            },
            100,
        );
        let forward = median_ms(
            || {
                for _ in 0..80 {
                    model.rhs(0.0, &state, &mut rhs);
                    black_box(&rhs);
                }
            },
            100,
        );
        let reverse = median_ms(
            || {
                dp.fill(0.0);
                for _ in 0..80 {
                    dx.fill(0.0);
                    model.rhs_pullback(0.0, &state, &cotangent, &mut dx, &mut dp);
                    black_box(&dx);
                }
                black_box(&dp);
            },
            100,
        );
        let mut projected = model.clone();
        let projection = median_ms(
            || {
                projected.project();
                black_box(&projected);
            },
            100,
        );
        println!("dimension={n} objective_ms={objective:.3} rhs_80_ms={forward:.3} pullback_80_ms={reverse:.3} projection_ms={projection:.3}");
    }
}
