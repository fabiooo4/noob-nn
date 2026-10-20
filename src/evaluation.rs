use ndarray::Array2;

pub struct ConfusionMatrix {
    pub matrix: Array2<usize>,
    pub labels: Vec<String>,
}

impl ConfusionMatrix {
    /// Validates the model accuracy on a validation set given as an array of (data, expected)
    ///
    /// Provides an estimate of the overall model quality, but it disregards class balance and the cost of different errors.
    pub fn accuracy(&self) -> f32 {
        let total = self.matrix.sum();
        if total == 0 {
            return 0.0;
        }

        let correct = self.matrix.diag().sum();
        correct as f32 / total as f32
    }

    pub fn precision_by_index(&self, class_idx: usize) -> f32 {
        let true_positive = self.matrix[[class_idx, class_idx]];
        let total_predicted = self.matrix.column(class_idx).sum();

        if total_predicted == 0 {
            return 0.0;
        }

        true_positive as f32 / total_predicted as f32
    }

    pub fn recall_by_index(&self, class_idx: usize) -> f32 {
        let true_positive = self.matrix[[class_idx, class_idx]];
        let total_actual = self.matrix.row(class_idx).sum();

        if total_actual == 0 {
            return 0.0;
        }

        true_positive as f32 / total_actual as f32
    }

    /// Validates the model precision
    ///
    /// Precision measures the model's ability to identify instances of a particular class correctly
    /// by dividing the instances correctly classified as belonging to a specific class by all the
    /// instances the model predicted to belong to that class
    ///
    /// # Returns
    /// Returns `None` if the position of the label is not found in the labels vector
    pub fn precision(&self, label: &str) -> Option<f32> {
        let idx = self.labels.iter().position(|l| l == label)?;
        Some(self.precision_by_index(idx))
    }

    /// Validates the model recall on a validation set
    ///
    /// Recall measures the model's ability to identify all instances of a particular class by
    /// dividing the instances in a class that the model correctly classified by all instances in that class
    ///
    /// # Returns
    /// Returns `None` if the position of the label is not found in the labels vector
    pub fn recall(&self, label: &str) -> Option<f32> {
        let idx = self.labels.iter().position(|l| l == label)?;
        Some(self.recall_by_index(idx))
    }

    /// Performs the macro-average of the precision on all labels
    pub fn precision_avg(&self) -> f32 {
        if self.labels.is_empty() {
            return 0.0;
        }
        let sum: f32 = (0..self.labels.len())
            .map(|i| self.precision_by_index(i))
            .sum();
        sum / self.labels.len() as f32
    }

    /// Performs the macro-average of the recall on all labels
    pub fn recall_avg(&self) -> f32 {
        if self.labels.is_empty() {
            return 0.0;
        }
        let sum: f32 = (0..self.labels.len())
            .map(|i| self.recall_by_index(i))
            .sum();
        sum / self.labels.len() as f32
    }
}
