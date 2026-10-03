use ndarray::{Array, Array2};
use rand::RngExt;

pub mod perceptron;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    // TODO: generalize layers
    /// Weights between input layer and hidden layer
    weights_ih: Array2<f32>,

    /// Hidden layer biases
    bias_h: Array2<f32>,

    /// Weights between hidden layer and output layer
    weights_ho: Array2<f32>,

    /// Output layer biases
    bias_o: Array2<f32>,

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

            // Initialize random biases
            bias_h: Array::from_shape_simple_fn((hidden_layer, 1), || rng.random_range(-1.0..1.0)),
            bias_o: Array::from_shape_simple_fn((output, 1), || rng.random_range(-1.0..1.0)),

            options,
        }
    }

    /// Calculates the output of a layer given an input
    fn feed_layer(
        input: &Array2<f32>,
        weight: &Array2<f32>,
        bias: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let bias = match bias {
            Some(b) => b,
            None => &Array::zeros((1, 1)),
        };

        let weighted_sum = weight.dot(input) + bias;
        weighted_sum.map(NeuralNetwork::activation)
    }

    pub fn feed_forward(&self, input: &[f32]) -> Vec<f32> {
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);

        // Feed input to the hidden layer
        let hidden_out =
            NeuralNetwork::feed_layer(&input_array, &self.weights_ih, Some(&self.bias_h));

        // Feed hidden layer's output to the output layer
        let (output_out, _) =
            NeuralNetwork::feed_layer(&hidden_out, &self.weights_ho, Some(&self.bias_o))
                .into_raw_vec_and_offset();

        output_out
    }

    // TODO: make the activation function and derivative as options
    fn activation(x: &f32) -> f32 {
        1. / (1. + f32::exp(-x))
    }

    // TODO: since the output of the network is already activated, to calculate the gradient this
    // function uses the already activated value, maybe change to the real derivative and pass the
    // non activated signal instead of the activated output directly
    fn activation_derivative(activated_input: &f32) -> f32 {
        activated_input * (1. - activated_input)
    }

    pub fn train(&mut self, input: &[f32], expected: &[f32]) {
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);
        let expected_array = Array::from_shape_vec((expected.len(), 1), expected.to_vec()).unwrap(/* An array can always be converted to a one dimensional matrix */);

        let hidden_out =
            NeuralNetwork::feed_layer(&input_array, &self.weights_ih, Some(&self.bias_h));
        let output_out =
            NeuralNetwork::feed_layer(&hidden_out, &self.weights_ho, Some(&self.bias_o));

        // Calculate error and backpropagate
        let output_err = expected_array - &output_out;
        let hidden_err = &self.weights_ho.t().dot(&output_err);

        // Gradient descent
        let output_gradient = output_err * output_out.map(NeuralNetwork::activation_derivative);
        let hidden_gradient = hidden_err * hidden_out.map(NeuralNetwork::activation_derivative);

        self.weights_ho += &(self.options.learning_rate * output_gradient.dot(&hidden_out.t()));
        self.weights_ih += &(self.options.learning_rate * hidden_gradient.dot(&input_array.t()));

        self.bias_o += &(self.options.learning_rate * output_gradient);
        self.bias_h += &(self.options.learning_rate * hidden_gradient);
    }
}
