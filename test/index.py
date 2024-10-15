import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sparse

class CustomStandardScaler(StandardScaler):
    def fit(self, X, y=None):
        # Override the fit method to skip the check_array
        # Calculate mean and variance without checking the array
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0, ddof=0)  # Use population standard deviation
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero
        return self

    def transform(self, X):
        # Override the transform method to skip the check_array
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        # Override the fit_transform method to combine fit and transform without check_array
        return self.fit(X).transform(X)
    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """

        copy = copy if copy is not None else self.copy
        X = True

        if False:
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

# Load the trained StackingClassifier, scaler, and label encoder
stacking_classifier = joblib.load('C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/stacking_classifier.pkl')
scaler = joblib.load('C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/scaler.pkl')
label_encoder = joblib.load('C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/label_encoder.pkl')

# Function to generate synthetic data
def generate_synthetic_data(num_samples=5):
    """Generate synthetic particle data for predictions."""
    synthetic_data = {
        'num_red': np.random.randint(10, 100, size=num_samples),  # Random values for red particles
        'num_green': np.random.randint(10, 100, size=num_samples),  # Random values for green particles
    }
    
    # Calculate red-green ratio, red distance, and green distance based on the generated numbers
    synthetic_data['red_green_ratio'] = synthetic_data['num_red'] / (synthetic_data['num_green'] + 1e-10)  # Avoid division by zero
    synthetic_data['red_distance'] = np.random.uniform(0, 50, size=num_samples)  # Random distances for red particles
    synthetic_data['green_distance'] = np.random.uniform(0, 50, size=num_samples)  # Random distances for green particles
    
    return pd.DataFrame(synthetic_data)

# Generate synthetic data
num_samples = 5  # Change this to generate more samples if needed
new_features_df = generate_synthetic_data(num_samples)

# Scale the input features using the loaded scaler
new_features_scaled = scaler.transform(new_features_df)

# Make predictions
predictions = stacking_classifier.predict(new_features_scaled)

# Debugging: Check the shape and type of predictions
print("Predictions Shape:", predictions.shape)
print("Predictions Type:", predictions.dtype)

# Convert numerical predictions back to original labels (no need to reshape)
predicted_labels = label_encoder.inverse_transform(predictions)

# Output the results
for i in range(num_samples):
    print(f"Sample {i + 1}: Features - {new_features_df.iloc[i].to_dict()}, Predicted Label - {predicted_labels[i]}")
