use ndarray::{Array, Array2, ArrayView2};
use rand::RngExt;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array2<f32>,
}

impl Layer {
    pub fn new(inputs: usize, nodes: usize) -> Self {
        let mut rng = rand::rng();

        // TODO: maybe tweak weight generation
        // Initialize random weights
        Self {
            weights: Array::from_shape_simple_fn((nodes, inputs), || rng.random_range(-1.0..1.0)),
            biases: Array::from_shape_simple_fn((nodes, 1), || rng.random_range(-1.0..1.0)),
        }
    }

    /// Calculates the output of a layer given an input
    pub fn feed_layer<T: Fn(&f32) -> f32>(
        &self,
        input: &ArrayView2<f32>,
        activation: T,
    ) -> Array2<f32> {
        let weighted_sum = self.weights.dot(input) + &self.biases;
        weighted_sum.map(activation)
    }

    pub fn to_vec(&self) -> Vec<Vec<f32>> {
        self.weights
            .clone()
            .into_raw_vec_and_offset()
            .0
            .chunks(self.weights.dim().1)
            .map(|row| row.to_vec())
            .collect()
    }
}
