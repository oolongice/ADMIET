use crate::{default_operators, AdjointSystem, EvolutionSystem, OperatorRef, Result};

#[derive(Clone, Debug)]
enum Node {
    // The original terminal is w0*c0 + w1*c1*(sum_j v_j*x_j).
    Leaf {
        choice: usize,
        c0: usize,
        c1: usize,
        variable: usize,
    },
    Branch {
        choice: usize,
        left: usize,
        right: usize,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct Simplex {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

#[derive(Clone, Debug)]
pub struct Forest {
    state_dim: usize,
    inputs: Vec<Vec<usize>>,
    nodes: Vec<Vec<Node>>, // postorder; child indices precede parents
    operators: Vec<OperatorRef>,
    pub(crate) parameters: Vec<f64>,
    pub(crate) simplexes: Vec<Simplex>,
    constants: Vec<usize>,
    constant_bounds: (f64, f64),
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(2685821657736338717) >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

impl Forest {
    /// `branches` is the number of binary operator nodes in each output tree.
    pub fn new(state_dim: usize, branches: usize, bounds: (f64, f64), seed: u64) -> Result<Self> {
        Self::with_operators(state_dim, branches, default_operators(), bounds, seed)
    }

    /// Use a custom operator registry from the library API.
    pub fn with_operators(
        state_dim: usize,
        branches: usize,
        operators: Vec<OperatorRef>,
        bounds: (f64, f64),
        seed: u64,
    ) -> Result<Self> {
        let inputs = (0..state_dim).map(|_| (0..state_dim).collect()).collect();
        Self::with_operators_and_dependencies(state_dim, branches, operators, inputs, bounds, seed)
    }

    /// Restrict each output equation to its listed state indices. This reduces
    /// both parameter count and search ambiguity in sparse high-dimensional systems.
    pub fn with_dependencies(
        state_dim: usize,
        branches: usize,
        inputs: Vec<Vec<usize>>,
        bounds: (f64, f64),
        seed: u64,
    ) -> Result<Self> {
        Self::with_operators_and_dependencies(
            state_dim,
            branches,
            default_operators(),
            inputs,
            bounds,
            seed,
        )
    }

    pub fn with_operators_and_dependencies(
        state_dim: usize,
        branches: usize,
        operators: Vec<OperatorRef>,
        inputs: Vec<Vec<usize>>,
        bounds: (f64, f64),
        seed: u64,
    ) -> Result<Self> {
        if state_dim == 0
            || operators.is_empty()
            || branches == 0
            || inputs.len() != state_dim
            || inputs.iter().any(|row| {
                row.is_empty() || row.iter().any(|&j| j >= state_dim) || {
                    let mut sorted = row.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    sorted.len() != row.len()
                }
            })
            || !bounds.0.is_finite()
            || !bounds.1.is_finite()
            || bounds.0 >= bounds.1
        {
            return Err(
                "invalid state dimension, branches, operators, dependencies, or constant bounds"
                    .into(),
            );
        }
        let mut forest = Self {
            state_dim,
            inputs,
            nodes: Vec::new(),
            operators,
            parameters: Vec::new(),
            simplexes: Vec::new(),
            constants: Vec::new(),
            constant_bounds: bounds,
        };
        let mut rng = Rng(seed.max(1));
        for equation in 0..state_dim {
            let mut nodes = Vec::new();
            forest.build_subtree(
                0,
                branches,
                forest.inputs[equation].len(),
                &mut nodes,
                &mut rng,
            );
            forest.nodes.push(nodes);
        }
        Ok(forest)
    }

    fn simplex(&mut self, len: usize) -> usize {
        let start = self.parameters.len();
        self.parameters
            .extend(std::iter::repeat_n(1.0 / len as f64, len));
        self.simplexes.push(Simplex { start, len });
        start
    }
    fn constant(&mut self, rng: &mut Rng) -> usize {
        let index = self.parameters.len();
        // Keep initial nonlinear dynamics near zero so integration is viable.
        let scale = (self.constant_bounds.1 - self.constant_bounds.0).min(0.2);
        self.parameters.push(
            ((rng.next() - 0.5) * scale).clamp(self.constant_bounds.0, self.constant_bounds.1),
        );
        self.constants.push(index);
        index
    }
    fn build_subtree(
        &mut self,
        i: usize,
        branches: usize,
        input_count: usize,
        nodes: &mut Vec<Node>,
        rng: &mut Rng,
    ) -> usize {
        if i >= branches {
            let choice = self.simplex(2);
            let c0 = self.constant(rng);
            let c1 = self.constant(rng);
            let variable = self.simplex(input_count);
            let index = nodes.len();
            nodes.push(Node::Leaf {
                choice,
                c0,
                c1,
                variable,
            });
            index
        } else {
            let left = self.build_subtree(2 * i + 1, branches, input_count, nodes, rng);
            let right = self.build_subtree(2 * i + 2, branches, input_count, nodes, rng);
            let choice = self.simplex(self.operators.len());
            let index = nodes.len();
            nodes.push(Node::Branch {
                choice,
                left,
                right,
            });
            index
        }
    }
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    pub fn set_parameters(&mut self, values: &[f64]) -> Result<()> {
        if values.len() != self.parameters.len() || values.iter().any(|v| !v.is_finite()) {
            return Err("invalid parameter vector".into());
        }
        self.parameters.copy_from_slice(values);
        self.project();
        Ok(())
    }
    /// Euclidean projection onto each probability simplex; constants are clipped.
    pub fn project(&mut self) {
        for simplex in &self.simplexes {
            let values = &mut self.parameters[simplex.start..simplex.start + simplex.len];
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let mut sum = 0.0;
            let mut rho = 0;
            for (j, &v) in sorted.iter().enumerate() {
                sum += v;
                if v > (sum - 1.0) / (j + 1) as f64 {
                    rho = j + 1;
                }
            }
            let theta = (sorted[..rho].iter().sum::<f64>() - 1.0) / rho as f64;
            for v in values {
                *v = (*v - theta).max(0.0);
            }
        }
        for &i in &self.constants {
            self.parameters[i] =
                self.parameters[i].clamp(self.constant_bounds.0, self.constant_bounds.1);
        }
    }

    fn eval_tree(
        &self,
        tree: &[Node],
        x: &[f64],
        inputs: &[usize],
        mut dx: Option<&mut [f64]>,
        mut dp: Option<&mut [f64]>,
        seed: f64,
    ) -> f64 {
        let differentiating = seed != 0.0 && (dx.is_some() || dp.is_some());
        let mut values = vec![0.0; tree.len()];
        let mut slopes = if differentiating {
            vec![[0.0; 2]; tree.len()]
        } else {
            Vec::new()
        };
        for (i, node) in tree.iter().enumerate() {
            match *node {
                Node::Leaf {
                    choice,
                    c0,
                    c1,
                    variable,
                } => {
                    let z: f64 = inputs
                        .iter()
                        .enumerate()
                        .map(|(j, &index)| self.parameters[variable + j] * x[index])
                        .sum();
                    values[i] = self.parameters[choice] * self.parameters[c0]
                        + self.parameters[choice + 1] * self.parameters[c1] * z;
                }
                Node::Branch {
                    choice,
                    left,
                    right,
                } => {
                    for (j, op) in self.operators.iter().enumerate() {
                        let (v, da, db) = op.evaluate(values[left], values[right]);
                        values[i] += self.parameters[choice + j] * v;
                        if differentiating {
                            slopes[i][0] += self.parameters[choice + j] * da;
                            slopes[i][1] += self.parameters[choice + j] * db;
                        }
                    }
                }
            }
        }
        if differentiating {
            let mut adjoint = vec![0.0; tree.len()];
            adjoint[tree.len() - 1] = seed;
            for (i, node) in tree.iter().enumerate().rev() {
                let a = adjoint[i];
                match *node {
                    Node::Leaf {
                        choice,
                        c0,
                        c1,
                        variable,
                    } => {
                        let z: f64 = inputs
                            .iter()
                            .enumerate()
                            .map(|(j, &index)| self.parameters[variable + j] * x[index])
                            .sum();
                        if let Some(g) = dp.as_deref_mut() {
                            g[choice] += a * self.parameters[c0];
                            g[choice + 1] += a * self.parameters[c1] * z;
                            g[c0] += a * self.parameters[choice];
                            g[c1] += a * self.parameters[choice + 1] * z;
                            for (j, &index) in inputs.iter().enumerate() {
                                g[variable + j] += a
                                    * self.parameters[choice + 1]
                                    * self.parameters[c1]
                                    * x[index];
                            }
                        }
                        if let Some(g) = dx.as_deref_mut() {
                            for (j, &index) in inputs.iter().enumerate() {
                                g[index] += a
                                    * self.parameters[choice + 1]
                                    * self.parameters[c1]
                                    * self.parameters[variable + j];
                            }
                        }
                    }
                    Node::Branch {
                        choice,
                        left,
                        right,
                    } => {
                        adjoint[left] += a * slopes[i][0];
                        adjoint[right] += a * slopes[i][1];
                        if let Some(g) = dp.as_deref_mut() {
                            for (j, op) in self.operators.iter().enumerate() {
                                g[choice + j] += a * op.evaluate(values[left], values[right]).0;
                            }
                        }
                    }
                }
            }
        }
        values[tree.len() - 1]
    }

    pub fn equations(&self) -> Vec<String> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(equation, tree)| {
                let mut expressions = Vec::<String>::new();
                for node in tree {
                    let s = match *node {
                        Node::Leaf {
                            choice,
                            c0,
                            c1,
                            variable,
                        } => {
                            let vars = self.inputs[equation]
                                .iter()
                                .enumerate()
                                .map(|(j, &index)| {
                                    format!("{:.3}x{}", self.parameters[variable + j], index + 1)
                                })
                                .collect::<Vec<_>>()
                                .join("+");
                            format!(
                                "({:.3}*{:.3}+{:.3}*{:.3}*({vars}))",
                                self.parameters[choice],
                                self.parameters[c0],
                                self.parameters[choice + 1],
                                self.parameters[c1]
                            )
                        }
                        Node::Branch {
                            choice,
                            left,
                            right,
                        } => self
                            .operators
                            .iter()
                            .enumerate()
                            .map(|(j, op)| {
                                format!(
                                    "{:.3}*{}",
                                    self.parameters[choice + j],
                                    op.format_expression(&expressions[left], &expressions[right])
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("+"),
                    };
                    expressions.push(s);
                }
                expressions.pop().unwrap()
            })
            .collect()
    }
}

impl Forest {
    /// Row-major Jacobians of the vector field with respect to state and parameters.
    pub fn rhs_and_jacobians(
        &self,
        state: &[f64],
        rhs: &mut [f64],
        state_jac: &mut [f64],
        param_jac: &mut [f64],
    ) {
        assert_eq!(state.len(), self.state_dim);
        assert_eq!(rhs.len(), self.state_dim);
        assert_eq!(state_jac.len(), self.state_dim * self.state_dim);
        assert_eq!(param_jac.len(), self.state_dim * self.parameters.len());
        state_jac.fill(0.0);
        param_jac.fill(0.0);
        for k in 0..self.state_dim {
            rhs[k] = self.eval_tree(
                &self.nodes[k],
                state,
                &self.inputs[k],
                Some(&mut state_jac[k * self.state_dim..(k + 1) * self.state_dim]),
                Some(&mut param_jac[k * self.parameters.len()..(k + 1) * self.parameters.len()]),
                1.0,
            );
        }
    }
}

impl EvolutionSystem for Forest {
    fn state_dim(&self) -> usize {
        self.state_dim
    }
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    fn rhs_and_sensitivities(
        &self,
        _t: f64,
        state: &[f64],
        sensitivity: &[f64],
        rhs: &mut [f64],
        sensitivity_dot: &mut [f64],
    ) {
        let n = self.state_dim;
        let p = self.parameters.len();
        assert_eq!(sensitivity.len(), n * p);
        assert_eq!(sensitivity_dot.len(), n * p);
        let mut state_jac = vec![0.0; n * n];
        self.rhs_and_jacobians(state, rhs, &mut state_jac, sensitivity_dot);
        for i in 0..n {
            for q in 0..p {
                for j in 0..n {
                    sensitivity_dot[i * p + q] += state_jac[i * n + j] * sensitivity[j * p + q];
                }
            }
        }
    }
}

impl AdjointSystem for Forest {
    fn state_dim(&self) -> usize {
        self.state_dim
    }
    fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    fn rhs(&self, _t: f64, state: &[f64], output: &mut [f64]) {
        assert_eq!(state.len(), self.state_dim);
        assert_eq!(output.len(), self.state_dim);
        for (k, tree) in self.nodes.iter().enumerate() {
            output[k] = self.eval_tree(tree, state, &self.inputs[k], None, None, 0.0);
        }
    }
    fn rhs_pullback(
        &self,
        _t: f64,
        state: &[f64],
        cotangent: &[f64],
        state_grad: &mut [f64],
        parameter_grad: &mut [f64],
    ) {
        assert_eq!(state.len(), self.state_dim);
        assert_eq!(cotangent.len(), self.state_dim);
        assert_eq!(state_grad.len(), self.state_dim);
        assert_eq!(parameter_grad.len(), self.parameters.len());
        for (k, tree) in self.nodes.iter().enumerate() {
            if cotangent[k] != 0.0 {
                self.eval_tree(
                    tree,
                    state,
                    &self.inputs[k],
                    Some(state_grad),
                    Some(parameter_grad),
                    cotangent[k],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{integrate, trajectory_loss, vector_field_loss, Observations};
    #[test]
    fn jacobians_match_finite_differences() {
        let mut model = Forest::new(2, 2, (-2.0, 2.0), 7).unwrap();
        let x = [0.7, 1.3];
        let n = 2;
        let p = model.parameters.len();
        let mut rhs = vec![0.0; n];
        let mut jx = vec![0.0; n * n];
        let mut jp = vec![0.0; n * p];
        model.rhs_and_jacobians(&x, &mut rhs, &mut jx, &mut jp);
        let h = 1e-6;
        for j in 0..n {
            let mut xp = x;
            xp[j] += h;
            let mut fp = vec![0.0; n];
            model.rhs_and_jacobians(&xp, &mut fp, &mut vec![0.0; n * n], &mut vec![0.0; n * p]);
            for i in 0..n {
                assert!(((fp[i] - rhs[i]) / h - jx[i * n + j]).abs() < 1e-4);
            }
        }
        for q in 0..p {
            let base = model.parameters[q];
            model.parameters[q] = base + h;
            let mut fp = vec![0.0; n];
            model.rhs_and_jacobians(&x, &mut fp, &mut vec![0.0; n * n], &mut vec![0.0; n * p]);
            model.parameters[q] = base;
            for i in 0..n {
                assert!(
                    ((fp[i] - rhs[i]) / h - jp[i * p + q]).abs() < 1e-4,
                    "parameter {q}, equation {i}"
                );
            }
        }
    }

    #[test]
    fn trajectory_gradient_matches_finite_differences() {
        let truth = Forest::new(1, 1, (-1.0, 1.0), 4).unwrap();
        let times = vec![0.0, 0.1, 0.2, 0.3];
        let states = integrate(&truth, &times, &[0.4], 0.02).unwrap().0;
        let data = Observations { times, states };
        let mut model = Forest::new(1, 1, (-1.0, 1.0), 5).unwrap();
        let (loss, gradient) = trajectory_loss(&model, &data, 0.02).unwrap();
        let h = 1e-6;
        for (q, expected) in gradient.iter().enumerate() {
            model.parameters[q] += h;
            let shifted = trajectory_loss(&model, &data, 0.02).unwrap().0;
            model.parameters[q] -= h;
            assert!(
                ((shifted - loss) / h - expected).abs() < 1e-5,
                "parameter {q}"
            );
        }
    }

    #[test]
    fn vector_field_gradient_matches_finite_differences() {
        let mut model = Forest::new(2, 1, (-0.5, 0.5), 4).unwrap();
        let data = Observations {
            times: vec![0.0, 0.1, 0.2],
            states: vec![vec![0.2, 0.3], vec![0.22, 0.31], vec![0.25, 0.29]],
        };
        let (loss, gradient) = vector_field_loss(&model, &data).unwrap();
        let h = 1e-6;
        for (q, expected) in gradient.iter().enumerate() {
            model.parameters[q] += h;
            let shifted = vector_field_loss(&model, &data).unwrap().0;
            model.parameters[q] -= h;
            assert!(
                ((shifted - loss) / h - expected).abs() < 1e-5,
                "parameter {q}"
            );
        }
    }
}
