import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import os
from feature_extraction import extract_features

# Function to load and process images
def load_and_process_image(image_path):
    features = extract_features(image_path)
    
    # Define labels based on particle counts
    if features['num_red'] > features['num_green']:
        label = 'Red Dominant'
    elif features['num_green'] > features['num_red']:
        label = 'Green Dominant'
    else:
        label = 'Equal'
    
    return features, label

# Directory containing the generated images
image_folder = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/data'

# Accumulate features and labels for all images
features_list = []
labels_list = []

for i in range(100):  # Assuming you have 100 images
    image_path = os.path.join(image_folder, f'bin_distribution_{i + 1}.png')
    features, label = load_and_process_image(image_path)
    
    # Flatten features to a dictionary and ensure only numeric values
    flat_features = {key: value for key, value in features.items() if isinstance(value, (int, float))}
    features_list.append(flat_features)
    labels_list.append(label)

# Convert features list to a DataFrame
features_df = pd.DataFrame(features_list)

# Extract features and labels for the model
X = features_df  # Use the cleaned DataFrame
y = labels_list  # Use the accumulated labels

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, algorithm='SAMME')  # Use SAMME instead of SAMME.R
}

# Store results
results = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Perform k-fold cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    # Store results
    results[model_name] = {
        'Test Accuracy': accuracy,
        'Cross-Validation Mean': np.mean(cross_val_scores),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }

# Display results
for model_name, metrics in results.items():
    print(f"{model_name} Test Accuracy: {metrics['Test Accuracy']:.2f}")
    print(f"{model_name} Cross-Validation Mean Accuracy: {metrics['Cross-Validation Mean']:.2f}\n")
    print(f"{model_name} Classification Report:\n", metrics['Classification Report'])

# Ensemble method using Voting Classifier
voting_model = VotingClassifier(estimators=[
    ('rf', models['Random Forest']),
    ('ab', models['AdaBoost'])
], voting='hard')

# Fit and evaluate ensemble model
voting_model.fit(X_train, y_train)
ensemble_accuracy = voting_model.score(X_test, y_test)
print(f"Ensemble Model Test Accuracy: {ensemble_accuracy:.2f}")

# AIC/BIC Calculation Function
def calculate_aic(n, residual_sum_of_squares, k):
    """Calculate AIC for model comparison."""
    return n * np.log(residual_sum_of_squares / n) + 2 * k

def calculate_bic(n, residual_sum_of_squares, k):
    """Calculate BIC for model comparison."""
    return n * np.log(residual_sum_of_squares / n) + np.log(n) * k

# Example usage of AIC/BIC calculation
# Placeholder values for demonstration purposes
n = len(X_train)  # Number of training samples
k_rf = 10  # Number of parameters in Random Forest (example)
k_ab = 10  # Number of parameters in AdaBoost (example)

# Residual sum of squares (you will need to calculate these based on your model)
residual_sum_of_squares_rf = np.sum((y_test != y_pred) ** 2)  # Example placeholder
residual_sum_of_squares_ab = np.sum((y_test != y_pred) ** 2)  # Example placeholder

# Calculate AIC/BIC for each model
aic_rf = calculate_aic(n, residual_sum_of_squares_rf, k_rf)
bic_rf = calculate_bic(n, residual_sum_of_squares_rf, k_rf)

aic_ab = calculate_aic(n, residual_sum_of_squares_ab, k_ab)
bic_ab = calculate_bic(n, residual_sum_of_squares_ab, k_ab)

print(f"Random Forest AIC: {aic_rf:.2f}, BIC: {bic_rf:.2f}")
print(f"AdaBoost AIC: {aic_ab:.2f}, BIC: {bic_ab:.2f}")
