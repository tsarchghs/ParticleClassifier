# Import necessary libraries
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, ttest_ind
import joblib
import seaborn as sns
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

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
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy
        X = True

        if sparse.issparse(X):
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

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
SAVE_FOLDER = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/data'
NUM_PARTICLES = 100
BIN_SIZE = (10, 10)

# ===========================
# Feature Extraction Function
# ===========================
def extract_features(image_path):
    """Extract features from a given image."""
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image {image_path}.")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    # Count red and green pixels
    num_red = np.sum(np.all(image == [255, 0, 0], axis=-1))
    num_green = np.sum(np.all(image == [0, 255, 0], axis=-1))
    red_green_ratio = num_red / (num_green + 1e-10)  # Avoid division by zero
    
    # Calculate average distances
    red_positions = np.argwhere(np.all(image == [255, 0, 0], axis=-1))
    green_positions = np.argwhere(np.all(image == [0, 255, 0], axis=-1))
    
    red_distance = calculate_average_distance(red_positions)
    green_distance = calculate_average_distance(green_positions)
    
    # Compute distributions
    vertical_distribution_red = np.sum(image[:, :, 0] == 255, axis=1)
    vertical_distribution_green = np.sum(image[:, :, 1] == 255, axis=1)
    
    horizontal_distribution_red = np.sum(image[:, :, 0] == 255, axis=0)
    horizontal_distribution_green = np.sum(image[:, :, 1] == 255, axis=0)

    return {
        'image_path': image_path,
        'num_red': num_red,
        'num_green': num_green,
        'red_green_ratio': red_green_ratio,
        'red_distance': red_distance,
        'green_distance': green_distance,
        'vertical_distribution_red': vertical_distribution_red.tolist(),
        'vertical_distribution_green': vertical_distribution_green.tolist(),
        'horizontal_distribution_red': horizontal_distribution_red.tolist(),
        'horizontal_distribution_green': horizontal_distribution_green.tolist()
    }

def calculate_average_distance(positions):
    """Calculate the average distance between particles."""
    if len(positions) <= 1:
        return 0
    distances = [np.linalg.norm(positions[i] - positions[j]) 
                 for i in range(len(positions)) 
                 for j in range(i + 1, len(positions))]
    return np.mean(distances)

# ===========================
# Data Generation Function
# ===========================
def generate_bin_distribution(num_particles=NUM_PARTICLES, red_ratio=0.5, bin_size=BIN_SIZE):
    """Generate a random bin distribution of red and green particles."""
    total_slots = bin_size[0] * bin_size[1]
    num_red = int(total_slots * red_ratio)
    num_green = total_slots - num_red

    red_particles = np.array([255, 0, 0], dtype=np.uint8)
    green_particles = np.array([0, 255, 0], dtype=np.uint8)

    particles = np.concatenate([np.tile(red_particles, (num_red, 1)), 
                                np.tile(green_particles, (num_green, 1))])
    
    np.random.shuffle(particles)
    particles = particles.reshape((bin_size[0], bin_size[1], 3))
    
    return particles

# ===========================
# Generate and Save Data
# ===========================
def generate_and_save_data(num_samples=100, bin_size=BIN_SIZE):
    """Generate and save synthetic particle data as images."""
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    for i in range(num_samples):
        red_ratio = np.random.uniform(0.1, 0.9)
        distribution = generate_bin_distribution(num_particles=NUM_PARTICLES, red_ratio=red_ratio, bin_size=bin_size)
        file_name = os.path.join(SAVE_FOLDER, f'bin_distribution_{i+1}.png')
        plt.imsave(file_name, distribution)

    return os.listdir(SAVE_FOLDER)

# ===========================
# Load and Process Data
# ===========================
def load_and_process_data(file_path):
    """Load and process extracted features from a CSV file."""
    data = pd.read_csv(file_path)
    for col in ['vertical_distribution_red', 'vertical_distribution_green', 
                 'horizontal_distribution_red', 'horizontal_distribution_green']:
        data[col] = data[col].apply(ast.literal_eval)
    return data

# ===========================
# Hypothesis Testing
# ===========================
def hypothesis_testing(data):
    """Perform hypothesis testing on particle distribution data."""
    contingency_table = pd.DataFrame({'Red': data['num_red'], 'Green': data['num_green']})
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    logging.info(f"Chi-Squared Statistic: {chi2}")
    logging.info(f"P-value: {p}")
    logging.info(f"Degrees of Freedom: {dof}")
    logging.info("Expected Frequencies:")
    logging.info(expected)

    if p < 0.05:
        logging.info("Reject the null hypothesis: significant relationship between red and green particle distributions.")
    else:
        logging.info("Fail to reject the null hypothesis: no significant relationship.")

    red_distances = data['red_distance']
    green_distances = data['green_distance']
    t_stat, p_value = ttest_ind(red_distances, green_distances)

    logging.info(f"T-Statistic: {t_stat}, P-Value: {p_value}")

    if p_value < 0.05:
        logging.info("Reject the null hypothesis: average distances of red and green particles are significantly different.")
    else:
        logging.info("Fail to reject the null hypothesis: average distances are not significantly different.")

    correlation_matrix = data[['num_red', 'num_green', 'red_green_ratio', 'red_distance', 'green_distance']].corr()
    logging.info("Correlation Matrix:")
    logging.info(correlation_matrix)

# ===========================
# Model Training
# ===========================
def train_and_evaluate_models(features_df, labels_list):
    """Train and evaluate various models on the extracted features."""
    # Select only the features the model expects
    expected_features = ['num_red', 'num_green', 'red_green_ratio', 'red_distance', 'green_distance']
    X = features_df[expected_features]

    # Fit and save the StandardScaler on the training data
    scaler = CustomStandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale the training data
    joblib.dump(scaler, 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/scaler.pkl')  # Save the scaler

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels_list, test_size=0.2, random_state=42)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    label_encoder.fit(labels_list)  # Fit on the complete label set to ensure all classes are recognized
    joblib.dump(scaler, 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/label_encoder.pkl')  # Save the scaler

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define base models
    base_models = [
        ('rf', RandomForestClassifier()),
        ('ada', AdaBoostClassifier(algorithm='SAMME'))
    ]

    # Use StackingClassifier for the meta-model
    stacking_classifier = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression()  # Ensure this is a classifier
    )

    # Train the stacking classifier
    stacking_classifier.fit(X_train, y_train_encoded)

    # Predictions
    stacking_train_predictions = stacking_classifier.predict(X_train)
    stacking_test_predictions = stacking_classifier.predict(X_test)

    # Evaluate model performance
    train_accuracy = accuracy_score(y_train_encoded, stacking_train_predictions)
    test_accuracy = accuracy_score(y_test_encoded, stacking_test_predictions)

    results = {
        'Stacking Classifier': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test_encoded, stacking_test_predictions)
        }
    }

    logging.info(f"Stacking Classifier - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    # Save the trained model
    joblib.dump(stacking_classifier, 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/stacking_classifier.pkl')  # Save the stacking classifier

    return results

# ===========================
# Main Execution Flow
# ===========================
if __name__ == "__main__":
    # Step 1: Generate synthetic data and save it
    generated_files = generate_and_save_data(num_samples=NUM_PARTICLES, bin_size=BIN_SIZE)

    # Step 2: Extract features from saved images and create a DataFrame
    feature_list = []
    for file_name in generated_files:
        feature = extract_features(os.path.join(SAVE_FOLDER, file_name))
        if feature:
            feature_list.append(feature)

    features_df = pd.DataFrame(feature_list)

    # Step 3: Generate labels for classification
    labels_list = np.where(features_df['num_red'] > features_df['num_green'], 'Red Dominant', 'Green Dominant')

    # Step 4: Hypothesis Testing
    hypothesis_testing(features_df)

    # Step 5: Train and evaluate models
    results = train_and_evaluate_models(features_df, labels_list)

    # Log the final results
    logging.info("Final Results:")
    for model, metrics in results.items():
        logging.info(f"{model}: {metrics}")

    # Optionally, save results to a CSV for further analysis
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv('C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/results.csv')
