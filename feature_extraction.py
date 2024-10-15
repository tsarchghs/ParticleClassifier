import numpy as np
import cv2
import os
import pandas as pd

def extract_features(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    height, width, _ = image.shape
    
    # Count red and green particles
    num_red = np.sum(np.all(image == [255, 0, 0], axis=-1))
    num_green = np.sum(np.all(image == [0, 255, 0], axis=-1))
    
    # Calculate red/green ratio
    red_green_ratio = num_red / (num_green + 1e-10)  # Avoid division by zero
    
    # Get positions of red and green particles
    red_positions = np.argwhere(np.all(image == [255, 0, 0], axis=-1))
    green_positions = np.argwhere(np.all(image == [0, 255, 0], axis=-1))
    
    # Calculate average distance between red particles
    if len(red_positions) > 1:
        red_distance = np.mean([np.linalg.norm(red_positions[i] - red_positions[j]) 
                                 for i in range(len(red_positions)) 
                                 for j in range(i + 1, len(red_positions))])
    else:
        red_distance = 0

    # Calculate average distance between green particles
    if len(green_positions) > 1:
        green_distance = np.mean([np.linalg.norm(green_positions[i] - green_positions[j]) 
                                   for i in range(len(green_positions)) 
                                   for j in range(i + 1, len(green_positions))])
    else:
        green_distance = 0

    # Vertical distribution: Count red and green particles per row
    vertical_distribution_red = np.sum(image[:, :, 0] == 255, axis=1)  # Red count per row
    vertical_distribution_green = np.sum(image[:, :, 1] == 255, axis=1)  # Green count per row

    # Horizontal distribution: Count red and green particles per column
    horizontal_distribution_red = np.sum(image[:, :, 0] == 255, axis=0)  # Red count per column
    horizontal_distribution_green = np.sum(image[:, :, 1] == 255, axis=0)  # Green count per column
    
    # Store features in a dictionary
    features = {
        'image_path': image_path,
        'num_red': num_red,  # Count of red particles
        'num_green': num_green,  # Count of green particles
        'red_green_ratio': red_green_ratio,
        'red_distance': red_distance,
        'green_distance': green_distance,
        'vertical_distribution_red': vertical_distribution_red.tolist(),
        'vertical_distribution_green': vertical_distribution_green.tolist(),
        'horizontal_distribution_red': horizontal_distribution_red.tolist(),
        'horizontal_distribution_green': horizontal_distribution_green.tolist()
    }
    
    return features

# Directory containing the generated images
image_folder = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/data'

# Extract features from all images and store them in a DataFrame
features_list = []
for image_file in os.listdir(image_folder):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_folder, image_file)
        features = extract_features(image_path)
        features_list.append(features)

# Convert the features list to a DataFrame
features_df = pd.DataFrame(features_list)

# Save the features DataFrame to a CSV file
features_df.to_csv('particle_distribution_features.csv', index=False)

print("Feature extraction completed and saved to particle_distribution_features.csv.")
