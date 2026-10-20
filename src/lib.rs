mod layer;

use crate::layer::Layer;

use ndarray::{Array2, ArrayView2};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    pub options: NeuralNetworkOptions,
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkOptions {
    pub learning_rate: f32,
    pub labels: Option<Vec<String>>,
}

impl Default for NeuralNetworkOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.3,
            labels: None,
        }
    }
}

// TODO: Use serde
impl From<NeuralNetworkOptions> for HashMap<String, String> {
    fn from(options: NeuralNetworkOptions) -> Self {
        HashMap::from([
            (
                "learning_rate".to_string(),
                options.learning_rate.to_string(),
            ),
            (
                "labels".to_string(),
                options.labels.unwrap_or(vec![]).join(", "),
            ),
        ])
    }
}

impl From<&HashMap<String, String>> for NeuralNetworkOptions {
    fn from(map: &HashMap<String, String>) -> Self {
        let learning_rate: f32 = map
            .get("learning_rate")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.3);

        let labels: Option<Vec<String>> = map.get("labels").and_then(|s| {
            if s.is_empty() {
                return None;
            };
            Some(s.split(", ").map(|e| e.to_owned()).collect())
        });

        Self {
            learning_rate,
            labels,
        }
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

    pub fn feed_forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if input.len() != self.shape()[0] {
            return Err(format!(
                "Input length {} does not match the number of input neurons {}",
                input.len(),
                self.shape()[0]
            )
            .into());
        }

        let input_array = ArrayView2::from_shape((input.len(), 1), input)?;
        let Some((first_layer, rest_layers)) = self.layers.split_first() else {
            return Ok(input.to_vec());
        };

        // Feed forward every layer
        let mut output = first_layer.feed_layer(&input_array, Self::activation);
        for layer in rest_layers {
            output = layer.feed_layer(&output.view(), Self::activation);
        }

        Ok(output.into_raw_vec_and_offset().0)
    }

    /// Converts a label to an expected output vector with a given value in the position of the
    /// label.
    ///
    /// # Returns
    /// Returns `None` if the model has no labels or if the given label is not contained in the
    /// model's labels
    pub fn label_to_output(&self, label: &str, value: f32) -> Option<Vec<f32>> {
        let labels = self.options.labels.clone()?;
        let idx = labels.iter().position(|l| l == label)?;

        let mut out = vec![0.0; labels.len()];
        let out_val = out.get_mut(idx)?;
        *out_val = value;

        Some(out)
    }

    /// Converts the output of the model into a label
    pub fn output_to_label(&self, output: &[f32]) -> Option<String> {
        let max = output.iter().cloned().reduce(f32::max).unwrap_or_default();
        let argmax = output.iter().position(|&o| o == max)?;

        self.options.labels.clone()?.get(argmax).cloned()
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

    /// Trains the neural network on a single input and expected output pair.
    pub fn train(
        &mut self,
        input: &[f32],
        expected: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if input.len() != self.shape()[0] {
            return Err(format!(
                "Input length {} does not match the number of input neurons {}",
                input.len(),
                self.shape()[0]
            )
            .into());
        }
        if expected.len() != *self.shape().last().ok_or("No layers in the network")? {
            return Err(format!(
                "Expected length {} does not match the number of output neurons {}",
                expected.len(),
                self.shape().last().copied().unwrap_or(0)
            )
            .into());
        }

        let input_array = ArrayView2::from_shape((input.len(), 1), input)?;
        let expected_array = ArrayView2::from_shape((expected.len(), 1), expected)?;

        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        outputs.push(input_array.to_owned());

        // Feed forward and save each layer output
        for layer in &self.layers {
            let prev_out = outputs
                .last()
                .ok_or("There is always at least a layer with an output")?;
            outputs.push(layer.feed_layer(&prev_out.view(), Self::activation));
        }

        let mut errors = Vec::with_capacity(self.layers.len() + 1);

        // Calculate the network output error to propagate backwards
        errors.push(
            &expected_array
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
            // TODO: this works only for the sigmoid activation function, when using other
            // functions the derivative should be calculated by taking as input the non activated
            // output of the layer
            let gradient = prev_err * layer_output.map(Self::activation_derivative);

            // Calculate the error for the previous layer
            // δe/δo
            // How much the error is sensitive to changes in output
            let err = layer.weights.t().dot(&gradient);
            errors.push(err);

            // Update weights
            // Sensitivity of the error to changes in weights
            // δe/δw = δi/δw * δo/δi * δe/δo
            layer.weights += &(self.options.learning_rate * gradient.dot(&layer_input.t()));

            // The derivative of the bias over the weights is 1 so just multiply by the gradient
            layer.biases += &(self.options.learning_rate * &gradient);
        }

        Ok(())
    }

    /// Validates the model accuracy on a validation set given as an array of (data, expected)
    ///
    /// Provides an estimate of the overall model quality, but it disregards class balance and the cost of different errors.
    pub fn accuracy(&self, validation_set: &[(&[f32], &str)]) -> Option<f32> {
        let mut correct = 0;
        for (data, expected) in validation_set {
            let expected = self.label_to_output(expected, 1.)?;

            let guess = self.feed_forward(data).ok()?;
            let guessed_class = argmax(&guess);
            let expected_class = argmax(&expected);

            if guessed_class == expected_class {
                correct += 1;
            }
        }

        Some(correct as f32 / validation_set.len() as f32)
    }

    /// Validates the model precision on a validation set given as an array of (data, expected)
    ///
    /// Precision measures the model's ability to identify instances of a particular class correctly
    /// by dividing the instances correctly classified as belonging to a specific class by all the
    /// instances the model predicted to belong to that class
    pub fn precision(&self, validation_set: &[(&[f32], &str)], label: &str) -> Option<f32> {
        let selected_class = argmax(&self.label_to_output(label, 1.)?);

        let mut true_positive = 0;
        let mut false_positive = 0;
        for (data, expected) in validation_set {
            let expected = self.label_to_output(expected, 1.)?;

            let guess = self.feed_forward(data).ok()?;
            let guessed_class = argmax(&guess);
            let expected_class = argmax(&expected);

            if guessed_class == selected_class {
                if guessed_class == expected_class {
                    true_positive += 1;
                } else {
                    false_positive += 1;
                }
            }
        }

        let total_predicted_positive = true_positive + false_positive;
        if total_predicted_positive == 0 {
            return Some(0.0);
        }

        Some(true_positive as f32 / total_predicted_positive as f32)
    }

    /// Validates the model recall on a validation set given as an array of (data, expected)
    ///
    /// Recall measures the model's ability to identify all instances of a particular class by
    /// dividing the instances in a class that the model correctly classified by all instances in that class
    ///
    /// # Returns
    /// Returns `None` if the label cannot be converted to an output vector
    pub fn recall(&self, validation_set: &[(&[f32], &str)], label: &str) -> Option<f32> {
        let selected_class = argmax(&self.label_to_output(label, 1.)?);

        let mut true_positive = 0;
        let mut false_negative = 0;
        for (data, expected) in validation_set {
            let expected = self.label_to_output(expected, 1.)?;

            let guess = self.feed_forward(data).ok()?;
            let guessed_class = argmax(&guess);
            let expected_class = argmax(&expected);

            if expected_class == selected_class {
                if guessed_class == selected_class {
                    true_positive += 1;
                } else {
                    false_negative += 1;
                }
            }
        }

        let total_actual_positive = true_positive + false_negative;
        if total_actual_positive == 0 {
            return Some(0.0);
        }

        Some(true_positive as f32 / total_actual_positive as f32)
    }

    /// Performs the macro-average of the precision on all labels
    ///
    /// # Returns
    /// Returns `None` if the model has no labels stored or if the stored label cannot be converted
    /// to an output vector
    pub fn precision_avg(&self, validation_set: &[(&[f32], &str)]) -> Option<f32> {
        let labels = self.options.labels.clone()?;
        let mut precision_sum = 0.;
        for label in &labels {
            precision_sum += self.precision(validation_set, label)?;
        }

        Some(precision_sum / labels.len() as f32)
    }

    /// Performs the macro-average of the recall on all labels
    ///
    /// # Returns
    /// Returns `None` if the model has no labels stored or if the stored label cannot be converted
    /// to an output vector
    pub fn recall_avg(&self, validation_set: &[(&[f32], &str)]) -> Option<f32> {
        let labels = self.options.labels.clone()?;
        let mut recall_sum = 0.;
        for label in &labels {
            recall_sum += self.recall(validation_set, label)?;
        }

        Some(recall_sum / labels.len() as f32)
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

        let mut layers: Vec<Layer> = Vec::new();
        let mut idx = 0;
        while let Ok(weights_tensor) = tensors.tensor(&format!("layer_{idx}_weights")) {
            let biases_tensor = tensors.tensor(&format!("layer_{idx}_biases"))?;

            let weights = Self::tensor_view_to_array2(&weights_tensor)?;
            let biases = Self::tensor_view_to_array2(&biases_tensor)?;

            if weights.dim().0 != biases.dim().0 || biases.dim().1 != 1 {
                return Err(format!(
                    "Mismatched dimensions between weights and biases in layer {idx}"
                )
                .into());
            }

            if let Some(prev_layer) = layers.last()
                && prev_layer.weights.dim().0 != weights.dim().1
            {
                return Err(format!(
                    "Layer size mismatch between layer {} and layer {idx}",
                    idx - 1
                )
                .into());
            }

            layers.push(Layer { weights, biases });
            idx += 1;
        }

        if layers.is_empty() {
            return Err("No layers found in file".into());
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

/// Returns the index of the maximum value
fn argmax(input: &[f32]) -> Option<usize> {
    let max = input.iter().cloned().reduce(f32::max)?;
    input.iter().position(|&i| i == max)
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

    #[test]
    fn test_save_and_load() {
        let path = "test_model.safetensors";
        let nn = NeuralNetwork::new(2, &[3, 4], 1, NeuralNetworkOptions::default());
        nn.save_to_file(path).expect("Failed to save");

        let loaded = NeuralNetwork::from_file(path).expect("Failed to load");
        assert_eq!(nn.shape(), loaded.shape());
        assert_eq!(nn.weights(0), loaded.weights(0));
        assert_eq!(nn.weights(1), loaded.weights(1));
        assert_eq!(nn.weights(2), loaded.weights(2));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_from_file_validation_layer_mismatch() {
        let path = "test_invalid_layers.safetensors";
        let layer0_w = Array2::<f32>::zeros((3, 2));
        let layer0_b = Array2::<f32>::zeros((3, 1));
        let layer1_w = Array2::<f32>::zeros((1, 4)); // Expected (1, 3) to match layer0 output 3
        let layer1_b = Array2::<f32>::zeros((1, 1));

        let tensors = vec![
            (
                "layer_0_weights".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer0_w.shape().to_vec(),
                    bytemuck::cast_slice(layer0_w.as_slice().unwrap()),
                )
                .unwrap(),
            ),
            (
                "layer_0_biases".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer0_b.shape().to_vec(),
                    bytemuck::cast_slice(layer0_b.as_slice().unwrap()),
                )
                .unwrap(),
            ),
            (
                "layer_1_weights".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer1_w.shape().to_vec(),
                    bytemuck::cast_slice(layer1_w.as_slice().unwrap()),
                )
                .unwrap(),
            ),
            (
                "layer_1_biases".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer1_b.shape().to_vec(),
                    bytemuck::cast_slice(layer1_b.as_slice().unwrap()),
                )
                .unwrap(),
            ),
        ];

        let mut metadata = HashMap::new();
        metadata.insert("learning_rate".to_string(), "0.3".to_string());
        safetensors::serialize_to_file(tensors, Some(metadata), Path::new(path)).unwrap();

        let result = NeuralNetwork::from_file(path);
        let _ = fs::remove_file(path);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Layer size mismatch")
        );
    }

    #[test]
    fn test_from_file_validation_bias_mismatch() {
        let path = "test_invalid_bias.safetensors";
        let layer0_w = Array2::<f32>::zeros((3, 2));
        let layer0_b = Array2::<f32>::zeros((4, 1)); // Expected (3, 1) to match nodes 3

        let tensors = vec![
            (
                "layer_0_weights".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer0_w.shape().to_vec(),
                    bytemuck::cast_slice(layer0_w.as_slice().unwrap()),
                )
                .unwrap(),
            ),
            (
                "layer_0_biases".to_string(),
                TensorView::new(
                    Dtype::F32,
                    layer0_b.shape().to_vec(),
                    bytemuck::cast_slice(layer0_b.as_slice().unwrap()),
                )
                .unwrap(),
            ),
        ];

        let mut metadata = HashMap::new();
        metadata.insert("learning_rate".to_string(), "0.3".to_string());
        safetensors::serialize_to_file(tensors, Some(metadata), Path::new(path)).unwrap();

        let result = NeuralNetwork::from_file(path);
        let _ = fs::remove_file(path);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Mismatched dimensions between weights and biases")
        );
    }

    #[test]
    fn test_from_file_validation_empty() {
        let path = "test_empty.safetensors";
        let tensors: Vec<(String, TensorView)> = vec![];
        let mut metadata = HashMap::new();
        metadata.insert("learning_rate".to_string(), "0.3".to_string());
        safetensors::serialize_to_file(tensors, Some(metadata), Path::new(path)).unwrap();

        let result = NeuralNetwork::from_file(path);
        let _ = fs::remove_file(path);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No layers found in file")
        );
    }

    #[test]
    fn test_labels_to_output() {
        let options = NeuralNetworkOptions {
            learning_rate: 0.3,
            labels: Some(vec!["cat".to_string(), "dog".to_string()]),
        };
        let nn = NeuralNetwork::new(2, &[3], 2, options);

        let output = nn.label_to_output("cat", 1.0);
        assert_eq!(output, Some(vec![1.0, 0.0]));

        let output = nn.label_to_output("dog", 1.0);
        assert_eq!(output, Some(vec![0.0, 1.0]));

        let output = nn.label_to_output("bird", 1.0);
        assert_eq!(output, None);
    }
}
