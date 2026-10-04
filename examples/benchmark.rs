use admiet::{windowed_trajectory_loss_adjoint, Forest, Observations};
use std::hint::black_box;
use std::time::Instant;

fn measure(model: &Forest, data: &Observations) -> f64 {
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        black_box(windowed_trajectory_loss_adjoint(model, data, 0.05, 1).unwrap());
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[5]
}

fn main() {
    println!("dimension,full_parameters,full_ms,sparse_parameters,sparse_ms");
    for dimension in [2, 4, 8, 16] {
        let model = Forest::new(dimension, 1, (-1.0, 1.0), 1).unwrap();
        let data = Observations {
            times: (0..11).map(|k| k as f64 * 0.1).collect(),
            states: vec![vec![0.1; dimension]; 11],
        };
        let full_ms = measure(&model, &data);
        let dependencies = (0..dimension)
            .map(|i| {
                let mut row = vec![(i + dimension - 1) % dimension, i, (i + 1) % dimension];
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        let sparse = Forest::with_dependencies(dimension, 1, dependencies, (-1.0, 1.0), 1).unwrap();
        let sparse_ms = measure(&sparse, &data);
        println!(
            "{dimension},{},{full_ms:.3},{},{sparse_ms:.3}",
            model.parameters().len(),
            sparse.parameters().len()
        );
    }
}
