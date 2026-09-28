use ndarray::{Array, Dim};
use rand::RngExt;

pub mod perceptron;

#[derive(Debug)]
pub struct NeuralNetwork {
    // TODO: generalize layers
    /// Weights between input layer and hidden layer
    weights_ih: Array<f32, Dim<[usize; 2]>>,

    /// Weights between hidden layer and output layer
    weights_ho: Array<f32, Dim<[usize; 2]>>,
}

impl NeuralNetwork {
    pub fn new(input: usize, hidden_layer: usize, output: usize) -> Self {
        let mut rng = rand::rng();
        NeuralNetwork {
            // Initialize random weights
            weights_ih: Array::from_shape_simple_fn((hidden_layer, input), || rng.random::<f32>()),
            weights_ho: Array::from_shape_simple_fn((output, hidden_layer), || rng.random::<f32>()),
        }
    }

    pub fn feed_forward(&self, input: &[f32]) -> Vec<f32> {
        let input_array = match Array::from_shape_vec((input.len(), 1), input.to_vec()) {
            Ok(array) => array,
            Err(e) => panic!("Failed to convert slice to array: {e}"),
        };

        // Feed input to the hidden layer (weighted sum)
        let hidden_out = self.weights_ih.dot(&input_array);

        // Feed hidden layer's output to the output layer (weighted sum)
        let (output_out, _) = self
            .weights_ho
            .dot(&hidden_out)
            .map(NeuralNetwork::activate) // Apply activation function
            .into_raw_vec_and_offset();

        output_out
    }

    // TODO: make the activation function as a parameter
    /// Sigmoid activation function
    fn activate(x: &f32) -> f32 {
        1. / (1. + f32::exp(-x))
    }

    pub fn train(&mut self, _input: &[f32], _expected: &[f32]) {
        todo!()
    }
}
