use std::{collections::HashMap, fs, path::Path};

use ndarray::{Array, Array2};
use rand::RngExt;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

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

    options: NeuralNetworkOptions,
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkOptions {
    learning_rate: f32,
}

impl Default for NeuralNetworkOptions {
    fn default() -> Self {
        Self { learning_rate: 0.3 }
    }
}

impl NeuralNetwork {
    pub fn new(
        input: usize,
        hidden_layer: usize,
        output: usize,
        options: NeuralNetworkOptions,
    ) -> Self {
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

    pub fn shape(&self) -> Vec<usize> {
        vec![
            self.weights_ih.dim().1, // Number of input neurons
            self.weights_ih.dim().0, // Number of hidden neurons
            self.weights_ho.dim().0, // Number of output neurons
        ]
    }

    /// Returns the weight matrix between two layers
    pub fn weights(&self, layer: usize) -> Result<Vec<Vec<f32>>, String> {
        match layer {
            0 => Ok(self
                .weights_ih
                .clone()
                .into_raw_vec_and_offset()
                .0
                .chunks(self.weights_ih.dim().1)
                .map(|row| row.to_vec())
                .collect()),
            1 => Ok(self
                .weights_ho
                .clone()
                .into_raw_vec_and_offset()
                .0
                .chunks(self.weights_ho.dim().1)
                .map(|row| row.to_vec())
                .collect()),
            _ => Err(String::from("Invalid layer indices")),
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
        // δe/δo
        // How much the error is sensitive to changes in output
        let output_err = expected_array - &output_out;
        let hidden_err = &self.weights_ho.t().dot(&output_err);

        // Gradient descent
        // δe/δo * δo/δi
        // Multiply the derivative of the error with the derivative of the output
        let output_gradient = output_err * output_out.map(NeuralNetwork::activation_derivative);
        let hidden_gradient = hidden_err * hidden_out.map(NeuralNetwork::activation_derivative);

        // Sensitivity of the error to changes in weights
        // δe/δw = δi/δw * δo/δi * δe/δo
        self.weights_ho += &(self.options.learning_rate * output_gradient.dot(&hidden_out.t()));
        self.weights_ih += &(self.options.learning_rate * hidden_gradient.dot(&input_array.t()));

        // The derivative of the bias over the weights is 1 so just multiply by the gradient
        self.bias_o += &(self.options.learning_rate * output_gradient);
        self.bias_h += &(self.options.learning_rate * hidden_gradient);
    }

    /// Saves the neural network to a file in safetensors format
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let tensors = vec![
            (
                "weights_ih",
                TensorView::new(
                    Dtype::F32,
                    self.weights_ih.shape().to_vec(),
                    bytemuck::cast_slice(
                        self.weights_ih
                            .as_slice()
                            .ok_or("Failed to convert weights_ih to slice")?,
                    ),
                )?,
            ),
            (
                "bias_h",
                TensorView::new(
                    Dtype::F32,
                    self.bias_h.shape().to_vec(),
                    bytemuck::cast_slice(
                        self.bias_h
                            .as_slice()
                            .ok_or("Failed to convert bias_h to slice")?,
                    ),
                )?,
            ),
            (
                "weights_ho",
                TensorView::new(
                    Dtype::F32,
                    self.weights_ho.shape().to_vec(),
                    bytemuck::cast_slice(
                        self.weights_ho
                            .as_slice()
                            .ok_or("Failed to convert weights_ho to slice")?,
                    ),
                )?,
            ),
            (
                "bias_o",
                TensorView::new(
                    Dtype::F32,
                    self.bias_o.shape().to_vec(),
                    bytemuck::cast_slice(
                        self.bias_o
                            .as_slice()
                            .ok_or("Failed to convert bias_o to slice")?,
                    ),
                )?,
            ),
        ];

        let mut metadata = HashMap::new();
        metadata.insert(
            "learning_rate".to_string(),
            self.options.learning_rate.to_string(),
        );

        safetensors::serialize_to_file(tensors, Some(metadata), Path::new(path))?;

        Ok(())
    }

    /// Loads a neural network from a file in safetensors format
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = fs::read(path)?;

        let tensors = SafeTensors::deserialize(&buffer)?;

        let weights_ih = Self::tensor_view_to_array2(&tensors.tensor("weights_ih")?)?;
        let bias_h = Self::tensor_view_to_array2(&tensors.tensor("bias_h")?)?;
        let weights_ho = Self::tensor_view_to_array2(&tensors.tensor("weights_ho")?)?;
        let bias_o = Self::tensor_view_to_array2(&tensors.tensor("bias_o")?)?;

        if weights_ih.dim().0 != bias_h.dim().0 {
            return Err("Mismatched dimensions between weights_ih and bias_h".into());
        }
        if weights_ho.dim().0 != bias_o.dim().0 {
            return Err("Mismatched dimensions between weights_ho and bias_o".into());
        }
        if weights_ih.dim().0 != weights_ho.dim().1 {
            return Err("Hidden layer size mismatch between input-hidden and hidden-output".into());
        }

        Ok(NeuralNetwork {
            weights_ih,
            bias_h,
            weights_ho,
            bias_o,
            options: NeuralNetworkOptions::default(),
        })
    }

    /// Converts a SafeTensors TensorView into an owned 2D ndarray
    fn tensor_view_to_array2(view: &TensorView) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        if view.dtype() != Dtype::F32 {
            return Err(format!("Expected F32 dtype, found {:?}", view.dtype()).into());
        }

        let shape = view.shape();
        if shape.len() != 2 {
            return Err(format!(
                "Expected 2D matrix, but tensor has {} dimensions",
                shape.len()
            )
            .into());
        }

        // Zero-copy cast &[u8] into &[f32]
        let data_f32: &[f32] = bytemuck::cast_slice(view.data());

        // Reconstruct ndarray (copies the slice into an owned buffer)
        let array = Array2::from_shape_vec((shape[0], shape[1]), data_f32.to_vec())?;
        Ok(array)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_shape() {
        let nn = NeuralNetwork::new(2, 3, 1, NeuralNetworkOptions::default());
        assert_eq!(nn.shape(), vec![2, 3, 1]);
    }

    #[test]
    fn test_weights() {
        let nn = NeuralNetwork::new(2, 3, 1, NeuralNetworkOptions::default());
        let weights_ih = nn.weights(1);
        let weights_ho = nn.weights(2);

        assert_eq!(
            weights_ih,
            Ok(nn
                .weights_ih
                .clone()
                .into_raw_vec_and_offset()
                .0
                .chunks(nn.weights_ih.dim().1)
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f32>>>())
        );

        assert_eq!(
            weights_ho,
            Ok(nn
                .weights_ho
                .clone()
                .into_raw_vec_and_offset()
                .0
                .chunks(nn.weights_ho.dim().1)
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f32>>>())
        );
    }
}
