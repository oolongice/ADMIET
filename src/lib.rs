//! ADMIET: weighted expression trees fitted to observed dynamics.
//! A spatially discretized PDE can implement [`EvolutionSystem`] with a flattened grid state.

mod adjoint;
mod model;
mod operators;
mod solver;
mod train;

pub use adjoint::{vector_field_loss, windowed_trajectory_loss_adjoint, AdjointSystem};
pub use model::Forest;
pub use operators::{
    default_operators, Add, BinaryOperator, Multiply, OperatorRef, ProtectedDivide, Subtract,
};
pub use solver::{
    integrate, trajectory_loss, windowed_trajectory_loss, EvolutionSystem, Observations,
};
pub use train::{fit, fit_many, FitOptions, FitReport};

pub type Result<T> = std::result::Result<T, String>;
