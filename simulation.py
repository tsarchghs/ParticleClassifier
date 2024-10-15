import numpy as np
import matplotlib.pyplot as plt

def generate_bin_distribution(num_particles=100, red_ratio=0.5, bin_size=(10, 10)):
    # Ensure the number of particles matches the bin size (no empty space)
    total_slots = bin_size[0] * bin_size[1]
    
    # Calculate number of red and green particles (ensure they fill the entire bin)
    num_red = int(total_slots * red_ratio)
    num_green = total_slots - num_red

    # Create red and green particles
    red_particles = np.array([255, 0, 0])  # Red in RGB
    green_particles = np.array([0, 255, 0])  # Green in RGB

    # Create an array of red and green particles
    particles = np.concatenate([np.tile(red_particles, (num_red, 1)), 
                                np.tile(green_particles, (num_green, 1))])

    # Shuffle the particles randomly to distribute them within the bin
    np.random.shuffle(particles)

    # Reshape the particles to match the bin size
    particles = particles.reshape((bin_size[0], bin_size[1], 3))
    
    return particles

def display_bin(bin_distribution):
    plt.imshow(bin_distribution)
    plt.axis('off')
    plt.show()

# Example: Generate and display bin distribution
bin_distribution = generate_bin_distribution(num_particles=100, red_ratio=0.6, bin_size=(10, 10))
display_bin(bin_distribution)
