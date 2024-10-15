import numpy as np
import os
import matplotlib.pyplot as plt

def generate_bin_distribution(num_particles=100, red_ratio=0.5, bin_size=(10, 10)):
    # Ensure the number of particles matches the bin size (no empty space)
    total_slots = bin_size[0] * bin_size[1]
    
    # Calculate number of red and green particles (ensure they fill the entire bin)
    num_red = int(total_slots * red_ratio)
    num_green = total_slots - num_red

    # Create red and green particles as uint8 arrays (ensure correct type)
    red_particles = np.array([255, 0, 0], dtype=np.uint8)  # Red in RGB
    green_particles = np.array([0, 255, 0], dtype=np.uint8)  # Green in RGB

    # Create an array of red and green particles
    particles = np.concatenate([np.tile(red_particles, (num_red, 1)), 
                                np.tile(green_particles, (num_green, 1))])

    # Shuffle the particles randomly to distribute them within the bin
    np.random.shuffle(particles)

    # Reshape the particles to match the bin size
    particles = particles.reshape((bin_size[0], bin_size[1], 3))
    
    return particles

# Generate and save multiple particle distributions with different parameters
def generate_and_save_data(num_samples=100, bin_size=(10, 10)):
    # Define the folder where the files will be saved
    save_folder = 'C:/Users/Gjergj/Desktop/Projects/MachineLearning/particles/data'
    os.makedirs(save_folder, exist_ok=True)  # Create directory if it doesn't exist
    
    for i in range(num_samples):
        # Randomly select the number of particles for each sample (max is bin size area)
        num_particles = 100  # Random number between 50 and bin capacity

        # Randomly select red/green ratio for each sample
        red_ratio = np.random.uniform(0.1, 0.9)

        # Generate particle distribution
        distribution = generate_bin_distribution(num_particles=num_particles, red_ratio=red_ratio, bin_size=bin_size)

        # Save each distribution as an image file
        file_name = os.path.join(save_folder, f'bin_distribution_{i+1}.png')
        plt.imsave(file_name, distribution)

    # List the generated files to verify
    return os.listdir(save_folder)

# Generate and save 100 different distributions with a specified bin size
generated_files = generate_and_save_data(num_samples=100, bin_size=(10, 10))

# Example: Display one of the generated bin distributions
sample_distribution = generate_bin_distribution(num_particles=100, red_ratio=0.6, bin_size=(10, 10))
plt.imshow(sample_distribution)
plt.axis('off')
plt.show()