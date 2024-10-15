import unittest
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.datasets import make_classification
from particle_analysis import extract_features, generate_bin_distribution, hypothesis_testing, train_and_evaluate_models

SAVE_FOLDER = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/test_data'
NUM_PARTICLES = 100
BIN_SIZE = (10, 10)

class TestParticleAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.makedirs(SAVE_FOLDER, exist_ok=True)

    def test_feature_extraction(self):
        """Test feature extraction from a generated image."""
        # Generate a test image with known quantities
        red_ratio = 0.6
        test_image = generate_bin_distribution(num_particles=NUM_PARTICLES, red_ratio=red_ratio, bin_size=BIN_SIZE)
        test_image_path = os.path.join(SAVE_FOLDER, 'test_image.png')
        cv2.imwrite(test_image_path, test_image)

        features = extract_features(test_image_path)
        self.assertIsNotNone(features)
        self.assertEqual(features['num_red'] + features['num_green'], NUM_PARTICLES)
        self.assertAlmostEqual(features['red_green_ratio'], red_ratio / (1 - red_ratio), places=1)

    def test_generate_bin_distribution(self):
        """Test the generation of bin distribution."""
        particles = generate_bin_distribution(num_particles=NUM_PARTICLES, red_ratio=0.5, bin_size=BIN_SIZE)
        unique, counts = np.unique(particles.reshape(-1, 3), axis=0, return_counts=True)
        
        num_red = counts[np.where((unique == [255, 0, 0]).all(axis=1))[0][0]]
        num_green = counts[np.where((unique == [0, 255, 0]).all(axis=1))[0][0]]

        self.assertEqual(num_red + num_green, NUM_PARTICLES)
        self.assertTrue(0 <= num_red <= NUM_PARTICLES)
        self.assertTrue(0 <= num_green <= NUM_PARTICLES)

    def test_hypothesis_testing(self):
        """Test hypothesis testing for significant relationships."""
        # Create a simple DataFrame for testing
        data = pd.DataFrame({
            'num_red': np.random.randint(0, 100, size=50),
            'num_green': np.random.randint(0, 100, size=50),
            'red_distance': np.random.rand(50) * 10,
            'green_distance': np.random.rand(50) * 10
        })
        
        # Since we only want to check if it runs without error
        try:
            hypothesis_testing(data)
        except Exception as e:
            self.fail(f"Hypothesis testing failed: {str(e)}")

    def test_train_and_evaluate_models(self):
        """Test the model training and evaluation process."""
        # Create synthetic features and labels
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        features_df = pd.DataFrame(X, columns=['num_red', 'num_green', 'red_green_ratio', 'red_distance', 'green_distance'])
        labels_list = np.where(y == 1, 'Red Dominant', 'Green Dominant')

        # Run the training and evaluation
        results = train_and_evaluate_models(features_df, labels_list)
        
        self.assertIn('Stacking Classifier', results)
        self.assertIn('train_accuracy', results['Stacking Classifier'])
        self.assertIn('test_accuracy', results['Stacking Classifier'])
        self.assertGreater(results['Stacking Classifier']['train_accuracy'], 0)
        self.assertGreater(results['Stacking Classifier']['test_accuracy'], 0)

    @classmethod
    def tearDownClass(cls):
        """Clean up test artifacts."""
        for filename in os.listdir(SAVE_FOLDER):
            file_path = os.path.join(SAVE_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(SAVE_FOLDER)

if __name__ == "__main__":
    unittest.main()
