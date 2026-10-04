use std::fmt::Debug;
use std::sync::Arc;

/// Extend the search grammar by implementing this trait and registering an
/// operator in `default_operators`, or pass a registry to `Forest::with_operators`.
/// Derivatives are with respect to the left and right inputs.
pub trait BinaryOperator: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn evaluate(&self, left: f64, right: f64) -> (f64, f64, f64);
    fn format_expression(&self, left: &str, right: &str) -> String {
        format!("{}({left},{right})", self.name())
    }
}

pub type OperatorRef = Arc<dyn BinaryOperator>;

#[derive(Debug)]
pub struct Add;
impl BinaryOperator for Add {
    fn name(&self) -> &'static str {
        "add"
    }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        (a + b, 1.0, 1.0)
    }
    fn format_expression(&self, a: &str, b: &str) -> String {
        format!("({a}+{b})")
    }
}
#[derive(Debug)]
pub struct Subtract;
impl BinaryOperator for Subtract {
    fn name(&self) -> &'static str {
        "subtract"
    }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        (a - b, 1.0, -1.0)
    }
    fn format_expression(&self, a: &str, b: &str) -> String {
        format!("({a}-{b})")
    }
}
#[derive(Debug)]
pub struct Multiply;
impl BinaryOperator for Multiply {
    fn name(&self) -> &'static str {
        "multiply"
    }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        (a * b, b, a)
    }
    fn format_expression(&self, a: &str, b: &str) -> String {
        format!("({a}*{b})")
    }
}
#[derive(Debug)]
pub struct ProtectedDivide;
impl BinaryOperator for ProtectedDivide {
    fn name(&self) -> &'static str {
        "protected_divide"
    }
    fn evaluate(&self, a: f64, b: f64) -> (f64, f64, f64) {
        if b.abs() < 1e-8 {
            (1.0, 0.0, 0.0)
        } else {
            (a / b, 1.0 / b, -a / (b * b))
        }
    }
    fn format_expression(&self, a: &str, b: &str) -> String {
        format!("protected_divide({a},{b})")
    }
}

/// Operators used by the CLI and `Forest::new`. Edit this registry to change
/// the application's search space; no command-line option is required.
pub fn default_operators() -> Vec<OperatorRef> {
    vec![Arc::new(Add), Arc::new(Subtract), Arc::new(Multiply)]
}
