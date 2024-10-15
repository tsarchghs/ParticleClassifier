import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import ast
from scipy.stats import chi2_contingency, ttest_ind

def load_and_process_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Convert string representations of lists to actual lists
    data['vertical_distribution_red'] = data['vertical_distribution_red'].apply(ast.literal_eval)
    data['vertical_distribution_green'] = data['vertical_distribution_green'].apply(ast.literal_eval)
    data['horizontal_distribution_red'] = data['horizontal_distribution_red'].apply(ast.literal_eval)
    data['horizontal_distribution_green'] = data['horizontal_distribution_green'].apply(ast.literal_eval)

    return data

def hypothesis_testing(data):
    # Create a contingency table
    contingency_table = pd.DataFrame({
        'Red': data['num_red'],
        'Green': data['num_green']
    })

    # Perform the Chi-Squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Squared Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)

    # Interpret results
    if p < 0.05:
        print("Reject the null hypothesis: there is a significant relationship between red and green particle distributions.")
    else:
        print("Fail to reject the null hypothesis: there is no significant relationship between red and green particle distributions.")

    # Perform the t-test for distances
    red_distances = data['red_distance']
    green_distances = data['green_distance']

    t_stat, p_value = ttest_ind(red_distances, green_distances)

    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")

    # Interpret results
    if p_value < 0.05:
        print("Reject the null hypothesis: the average distances of red and green particles are significantly different.")
    else:
        print("Fail to reject the null hypothesis: the average distances of red and green particles are not significantly different.")

    # Calculate correlation matrix
    correlation_matrix = data[['num_red', 'num_green', 'red_green_ratio', 'red_distance', 'green_distance']].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

# Assuming your CSV file path is defined here
file_path = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/particle_distribution_features.csv'

# Load and process the data
data = load_and_process_data(file_path)

# Perform hypothesis testing
hypothesis_testing(data)
