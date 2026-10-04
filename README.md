# ADMIET in Rust

This crate implements the dynamical-system part of the C++ ADMIET project. Each component of `dx/dt = f(x)` is a weighted binary expression tree. A leaf is

`w0*c0 + w1*c1*(v1*x1 + ... + vn*xn)`.

Each branch mixes the registered binary operators with nonnegative weights summing to one. Fitting integrates the model with fourth-order Runge–Kutta, differentiates the discrete solver in reverse, and updates parameters with Adam. Choice weights are projected onto probability simplexes after each update; constants are clipped to their bounds. An optional `1 - sum(w_i^2)` penalty encourages discrete operator and variable choices.

## Build and run

```bash
cd Sym_rust_repo
cargo test
cargo run --release -- ../ADMIET/scripts/real_data/data_t.txt ../ADMIET/scripts/real_data/data_x.txt \
  --branches 1 --pretrain 30 --epochs 100 --dt 0.1 --window 1 --bounds -1 1
```

Input times have one value per line; state rows may be whitespace or comma separated. The CLI prints the initial and final loss and one expression per state component. `--dt` is the maximum internal integration step, not the spacing of observations. `--window 1` fits one-step flows and is the default; increase it toward the full observation count to emphasize long trajectories once a stable model is found. Decrease `--dt` for accuracy or increase it for speed after checking convergence.

## Adding operators

The command line has no operator argument. Its registry is [`default_operators`](src/operators.rs). To change the CLI search space, implement `BinaryOperator`, add it to that function, and rebuild. Library users can supply a registry directly:

```rust
use admiet::{default_operators, BinaryOperator, Forest};
use std::sync::Arc;

#[derive(Debug)]
struct SumOfSquares;
impl BinaryOperator for SumOfSquares {
    fn name(&self) -> &'static str { "sum_of_squares" }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        (a*a + b*b, 2.0*a, 2.0*b)
    }
}

let mut operators = default_operators();
operators.push(Arc::new(SumOfSquares));
let model = Forest::with_operators(2, 1, operators, (-1.0, 1.0), 1).unwrap();
```

The returned tuple is `(value, derivative_by_left, derivative_by_right)`. The registry can include protected division or other differentiable binary functions. A custom operator can override `format_expression` to control printed equations.

## Larger systems

The main statistical difficulty is that each leaf searches over every state variable by default. A one-branch 16-state forest has 688 parameters; one trajectory may not explore enough independent states to identify them. Use known dependency structure when available:

For a four-state system, write `dependencies.txt` with one row per equation and one-based state indices:

```text
1 2
2 3
3 4
1 4
```

Pass `--dependencies dependencies.txt`. In library code, use `Forest::with_dependencies` or `Forest::with_operators_and_dependencies`. The constraints are explicit assumptions; excluding a real dependency prevents the model from recovering it. Add independent trajectories in the CLI with repeated `--trajectory TIME_FILE STATE_FILE` pairs, or use `fit_many` in library code. Multiple initial conditions supply more independent state variation. `--pretrain N` optionally fits short-interval derivative estimates for N epochs before trajectory fitting; use it cautiously with noisy or sparsely sampled data.

The reverse RK4 objective reduces the cost of gradients when parameter count grows. The forward-sensitivity API remains available for models implementing `EvolutionSystem`; a model can instead implement `AdjointSystem` for reverse training without forming a dense Jacobian.

## Benchmark

Run `cargo run --release --example benchmark`. It measures the median of 11 objective calls on 10 observation intervals, with one binary branch per equation and RK4 maximum step 0.05. On the development machine:

| Dimension | Parameters, full | Previous forward sensitivities | Reverse RK4, full | Parameters, 3 dependencies | Reverse RK4, sparse |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 30 | 0.026 ms | 0.017 ms | 30 | 0.017 ms |
| 4 | 76 | 0.125 ms | 0.030 ms | 68 | 0.029 ms |
| 8 | 216 | 1.001 ms | 0.064 ms | 136 | 0.053 ms |
| 16 | 688 | 19.110 ms | 0.160 ms | 272 | 0.097 ms |

The 16-state reverse objective was 0.214 ms before reusing tree scratch buffers and cached leaf/operator values; it is 0.160 ms after. `cargo run --release --example profile_cost` separately times forward RHS calls, pullbacks, and weight projection. These are objective evaluation times, not time to recover the correct equations. They depend on the machine, operator registry, tree size, and integration steps.

## PDE extension

`EvolutionSystem` operates on a flattened state. A method-of-lines PDE discretization can map grid cells to state entries and implement `rhs_and_sensitivities`. Its second output is `S' = J_state S + J_parameter`, where `S` has shape `(number of cells × number of parameters)` in row-major order. A local stencil can apply `J_state` directly, without creating a dense grid Jacobian. `AdjointSystem` offers a transpose-Jacobian action for reverse training. The [two-cell diffusion test](tests/method.rs) is a small example. A future PDE discovery model can add spatial derivative features while reusing the solver and objectives.

## Differences and scope

- The core standard weighted-tree forest and trajectory fitting are ported. The C++ tree-file JSON format, function-fitting executables, interpolation path, and pretraining file format are not yet ported.
- RK4 replaces the C++ Euler/RK45 selection. It hits observation times exactly and rejects nonfinite dynamics. It is an explicit solver; stiff PDE discretizations will need an implicit or adaptive solver.
- The C++ source has a separate operator setting file. The Rust CLI operator registry is in source code. Protected division is included as an optional built-in; the experimental Hill operator is not yet included.
- Training still has a nonconvex objective. Try multiple seeds and inspect stability, loss, and learned equations before interpreting a formula.
