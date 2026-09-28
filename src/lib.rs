use rand::RngExt;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: Bias,
    pub learning_rate: f32,
}

#[derive(Debug)]
pub struct Bias {
    pub value: f32,
    pub weight: f32,
}

impl Neuron {
    /// Creates a neuron with random weights for every input
    pub fn new(input_num: usize, learning_rate: f32) -> Self {
        let mut rng = rand::rng();

        Self {
            weights: (0..input_num).map(|_| rng.random_range(-1. ..1.)).collect(),
            bias: Bias {
                value: 1.,
                weight: rng.random_range(-1. ..1.),
            },
            learning_rate,
        }
    }

    /// Generates the output of the neuron by summing every input scaled by its weight
    pub fn feed_forward(&self, input: &[f32]) -> Result<f32, String> {
        if input.len() != self.weights.len() {
            return Err(format!(
                "Wrong input size, expected {} got {}",
                self.weights.len(),
                input.len()
            ));
        }

        let output = input
            .iter()
            .enumerate()
            .map(|(idx, input)| input * self.weights[idx])
            .sum::<f32>()
            + self.bias.value * self.bias.weight;

        Ok(Neuron::activate(output))
    }

    fn activate(output: f32) -> f32 {
        output.signum()
    }

    /// Trains the neuron by calculating the error as:
    /// ```
    /// error = desired output - guess output
    /// ```
    ///
    /// The error is used to move the weights towards a better value:
    /// ```
    /// new_weight = weight + dw = weights + (error * input)
    /// ```
    ///
    /// A learning rate reduces or increases the rate of change of the weights
    /// ```
    /// new_weight = weight + (error * input) * learning_rate
    /// ```
    pub fn train(&mut self, input: &[f32], expected: f32) -> Result<(), String> {
        let guessed = self.feed_forward(input)?;
        let error = expected - guessed;

        for (i, weight) in self.weights.iter_mut().enumerate() {
            *weight += error * input[i] * self.learning_rate
        }
        self.bias.weight += error * self.bias.value * self.learning_rate;

        Ok(())
    }
}
