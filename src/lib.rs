use ndarray::{Array, Dim};
use rand::RngExt;

pub mod perceptron;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    // TODO: generalize layers
    /// Weights between input layer and hidden layer
    weights_ih: Array<f32, Dim<[usize; 2]>>,

    /// Weights between hidden layer and output layer
    weights_ho: Array<f32, Dim<[usize; 2]>>,

    options: NetworkOptions,
}

#[derive(Debug, Clone)]
pub struct NetworkOptions {
    learning_rate: f32,
}

impl Default for NetworkOptions {
    fn default() -> Self {
        Self { learning_rate: 0.3 }
    }
}

impl NeuralNetwork {
    // TODO: add biases
    pub fn new(input: usize, hidden_layer: usize, output: usize, options: NetworkOptions) -> Self {
        let mut rng = rand::rng();
        NeuralNetwork {
            // TODO: maybe tweak weight generation
            // Initialize random weights
            weights_ih: Array::from_shape_simple_fn((hidden_layer, input), || {
                rng.random_range(-1.0..1.0)
            }),
            weights_ho: Array::from_shape_simple_fn((output, hidden_layer), || {
                rng.random_range(-1.0..1.0)
            }),

            options,
        }
    }

    pub fn feed_forward(&self, input: &[f32]) -> Vec<f32> {
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);

        // Feed input to the hidden layer (weighted sum)
        let hidden_in = self.weights_ih.dot(&input_array);
        let hidden_out = hidden_in.map(NeuralNetwork::activation);

        // Feed hidden layer's output to the output layer (weighted sum)
        let output_in = self.weights_ho.dot(&hidden_out);
        let (output_out, _) = output_in
            .map(NeuralNetwork::activation)
            .into_raw_vec_and_offset();

        output_out
    }

    // TODO: make the activation function and derivative as options
    fn activation(x: &f32) -> f32 {
        1. / (1. + f32::exp(-x))
    }

    fn activation_derivative(activated_input: &f32) -> f32 {
        activated_input * (1. - activated_input)
    }

    pub fn train(&mut self, input: &[f32], expected: &[f32]) {
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);
        let expected_array = Array::from_shape_vec((expected.len(), 1), expected.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);

        // Feed input to the hidden layer (weighted sum)
        let hidden_in = self.weights_ih.dot(&input_array);
        let hidden_out = hidden_in.map(NeuralNetwork::activation);

        // Feed hidden layer's output to the output layer (weighted sum)
        let output_in = self.weights_ho.dot(&hidden_out);
        let output_out = output_in.map(NeuralNetwork::activation); // Apply activation function

        // Calculate error and backpropagate
        let output_err = expected_array - &output_out;
        let hidden_err = &self.weights_ho.t().dot(&output_err);

        // Gradient descent
        let output_gradient = output_err * output_out.map(NeuralNetwork::activation_derivative);
        let hidden_gradient = hidden_err * hidden_out.map(NeuralNetwork::activation_derivative);

        self.weights_ho += &(self.options.learning_rate * output_gradient.dot(&hidden_out.t()));
        self.weights_ih += &(self.options.learning_rate * hidden_gradient.dot(&input_array.t()));
    }
}
