mod layer;

use crate::layer::Layer;

use ndarray::{Array, Array2};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
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

impl From<NeuralNetworkOptions> for HashMap<String, String> {
    fn from(options: NeuralNetworkOptions) -> Self {
        let mut map = HashMap::new();
        map.insert(
            "learning_rate".to_string(),
            options.learning_rate.to_string(),
        );
        map
    }
}

impl From<&HashMap<String, String>> for NeuralNetworkOptions {
    fn from(map: &HashMap<String, String>) -> Self {
        let learning_rate = map
            .get("learning_rate")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.3);

        Self { learning_rate }
    }
}

impl NeuralNetwork {
    pub fn new(
        input: usize,
        hidden_layers: &[usize],
        output: usize,
        options: NeuralNetworkOptions,
    ) -> Self {
        let sizes: Vec<usize> = [vec![input], hidden_layers.to_vec(), vec![output]]
            .iter()
            .flatten()
            .cloned()
            .collect();

        let mut layers = vec![];
        for size in sizes.windows(2) {
            layers.push(Layer::new(size[0], size[1]));
        }

        NeuralNetwork { layers, options }
    }

    /// Returns the number of neurons for each layer.
    /// The input counts as a layer, so the first element is the number of inputs and the last is
    /// the number of outputs.
    pub fn shape(&self) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.layers.len() + 1);
        let inputs = match self.layers.first() {
            Some(layer) => layer.weights.dim().1,
            None => 0,
        };
        sizes.push(inputs);

        for l in self.layers.iter() {
            sizes.push(l.weights.dim().0)
        }

        sizes
    }

    /// Returns the weight matrix between two layers
    /// Returns `None` if the layer index is out of bounds
    pub fn weights(&self, layer: usize) -> Option<Vec<Vec<f32>>> {
        Some(self.layers.get(layer)?.to_vec())
    }

    pub fn feed_forward(&self, input: &[f32]) -> Vec<f32> {
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec())
            .expect("An array can always be converted to a one dimensional matrix");

        // Feed forward every layer
        let mut output = input_array;
        for layer in &self.layers {
            output = layer.feed_layer(&output, Self::activation)
        }

        output.into_raw_vec_and_offset().0
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
        let input_array = Array::from_shape_vec((input.len(), 1), input.to_vec())
            .expect("An array can always be converted to a one dimensional matrix");
        let expected_array = Array::from_shape_vec((expected.len(), 1), expected.to_vec())
            .expect("An array can always be converted to a one dimensional matrix");

        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        outputs.push(input_array);

        // Feed forward and save each layer output
        for layer in &self.layers {
            let prev_out = outputs
                .last()
                .expect("There is always at least a layer with an output");
            outputs.push(layer.feed_layer(prev_out, Self::activation));
        }

        let mut errors = Vec::with_capacity(self.layers.len() + 1);

        // Calculate the network output error to propagate backwards
        errors.push(
            expected_array
                - outputs
                    .last()
                    .expect("There is always at least a layer with an output"),
        );

        // Backpropagation
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &outputs[i];
            let layer_output = &outputs[i + 1];

            let prev_err = errors
                .last()
                .expect("There is always at least a layer with an output");

            // Gradient descent
            // δe/δo * δo/δi
            // Multiply the derivative of the error with the derivative of the output
            let gradient = prev_err * layer_output.map(Self::activation_derivative);

            // Calculate the error for the previous layer
            // δe/δo
            // How much the error is sensitive to changes in output
            let err = layer.weights.t().dot(prev_err);
            errors.push(err);

            // Update weights
            // Sensitivity of the error to changes in weights
            // δe/δw = δi/δw * δo/δi * δe/δo
            layer.weights += &(self.options.learning_rate * gradient.dot(&layer_input.t()));

            // The derivative of the bias over the weights is 1 so just multiply by the gradient
            layer.biases += &(self.options.learning_rate * &gradient);
        }
    }

    /// Saves the neural network to a file in safetensors format
    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut tensors = Vec::with_capacity(self.layers.len() * 2); // Each layer has also a bias
        for (idx, layer) in self.layers.iter().enumerate() {
            tensors.push((
                format!("layer_{idx}_weights"),
                TensorView::new(
                    Dtype::F32,
                    layer.weights.shape().to_vec(),
                    bytemuck::cast_slice(
                        layer
                            .weights
                            .as_slice()
                            .ok_or(format!("Failed to convert 'layer_{idx}_weights' to slice"))?,
                    ),
                )?,
            ));
            tensors.push((
                format!("layer_{idx}_biases"),
                TensorView::new(
                    Dtype::F32,
                    layer.biases.shape().to_vec(),
                    bytemuck::cast_slice(
                        layer
                            .biases
                            .as_slice()
                            .ok_or(format!("Failed to convert 'layer_{idx}_biases' to slice"))?,
                    ),
                )?,
            ));
        }

        let metadata: HashMap<String, String> = self.options.clone().into();
        safetensors::serialize_to_file(tensors, Some(metadata), Path::new(path))?;

        Ok(())
    }

    /// Loads a neural network from a file in safetensors format
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer = fs::read(path)?;

        let tensors = SafeTensors::deserialize(&buffer)?;

        let mut layers = Vec::new();
        let mut idx = 0;
        while let Ok(weights_tensor) = tensors.tensor(&format!("layer_{idx}_weights")) {
            let biases_tensor = tensors.tensor(&format!("layer_{idx}_biases"))?;

            let weights = Self::tensor_view_to_array2(&weights_tensor)?;
            let biases = Self::tensor_view_to_array2(&biases_tensor)?;

            layers.push(Layer { weights, biases });
            idx += 1;
        }

        let metadata = SafeTensors::read_metadata(&buffer)?.1;
        let options = metadata
            .metadata()
            .as_ref()
            .ok_or("No metadata found")?
            .into();

        Ok(NeuralNetwork { layers, options })
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
        let nn = NeuralNetwork::new(2, &[3], 1, NeuralNetworkOptions::default());
        assert_eq!(nn.shape(), vec![2, 3, 1]);
    }

    #[test]
    fn test_weights() {
        let nn = NeuralNetwork::new(2, &[3], 1, NeuralNetworkOptions::default());
        let weights_ih = nn.weights(0);
        let weights_ho = nn.weights(1);

        assert_eq!(
            weights_ih,
            Some(
                nn.layers[0]
                    .weights
                    .clone()
                    .into_raw_vec_and_offset()
                    .0
                    .chunks(nn.layers[0].weights.dim().1)
                    .map(|row| row.to_vec())
                    .collect::<Vec<Vec<f32>>>()
            )
        );

        assert_eq!(
            weights_ho,
            Some(
                nn.layers[1]
                    .weights
                    .clone()
                    .into_raw_vec_and_offset()
                    .0
                    .chunks(nn.layers[1].weights.dim().1)
                    .map(|row| row.to_vec())
                    .collect::<Vec<Vec<f32>>>()
            )
        );
    }
}
