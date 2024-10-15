import numpy as np
import matplotlib.pyplot as plt

def generate_bin_distribution_from_counts(vertical_distribution_red, horizontal_distribution_red,
                                           vertical_distribution_green, horizontal_distribution_green, 
                                           bin_size=(10, 10)):
    # Create an empty bin initialized with green particles (or background color)
    bin_distribution = np.zeros((bin_size[0], bin_size[1], 3), dtype=np.uint8)  # RGB
    red_particle = np.array([255, 0, 0], dtype=np.uint8)  # Red in RGB
    green_particle = np.array([0, 255, 0], dtype=np.uint8)  # Green in RGB

    # Fill the bin with green particles initially
    bin_distribution[:, :] = green_particle

    # Initialize counters for total placed particles
    total_red_placed = 0
    total_green_placed = 0

    # Place red particles based on vertical and horizontal distributions
    for row in range(len(vertical_distribution_red)):
        count = vertical_distribution_red[row]
        placed_count = 0
        
        for col in range(bin_size[1]):
            if placed_count < count and horizontal_distribution_red[col] > 0:
                # Check if the position is green
                if bin_distribution[row, col, 0] == 0 and bin_distribution[row, col, 1] == 255:
                    bin_distribution[row, col] = red_particle
                    horizontal_distribution_red[col] -= 1  # Decrease the count for that column
                    placed_count += 1  # Increment the placed count
        
        total_red_placed += placed_count  # Update the total placed count for red

    # Place green particles based on vertical and horizontal distributions
    for row in range(len(vertical_distribution_green)):
        count = vertical_distribution_green[row]
        placed_count = 0
        
        for col in range(bin_size[1]):
            if placed_count < count and horizontal_distribution_green[col] > 0:
                # Check if the position is still green
                if bin_distribution[row, col, 1] == 0 and bin_distribution[row, col, 0] == 0:
                    bin_distribution[row, col] = green_particle
                    horizontal_distribution_green[col] -= 1  # Decrease the count for that column
                    placed_count += 1  # Increment the placed count
        
        total_green_placed += placed_count  # Update the total placed count for green

    final_red_count = np.sum(np.all(bin_distribution == red_particle, axis=-1))
    final_green_count = np.sum(np.all(bin_distribution == green_particle, axis=-1))

    # Log final counts
    print(f"Final red particle count: {final_red_count}")
    print(f"Final green particle count: {final_green_count}")

    return bin_distribution

def display_bin(bin_distribution):
    plt.imshow(bin_distribution)
    plt.axis('off')
    plt.show()

# Example usage with your data
vertical_distribution_red = [5, 4, 6, 7, 6, 7, 6, 4, 6, 4]  # Counts of red particles in each row
horizontal_distribution_red = [5, 6, 4, 3, 4, 3, 4, 6, 4, 6]  # Counts of red particles in each column

vertical_distribution_green = [5, 8, 6, 4, 4, 6, 4, 4, 8, 6]  # Counts of green particles in each row
horizontal_distribution_green = [5, 2, 4, 6, 6, 4, 6, 6, 2, 4]  # Counts of green particles in each column

# Generate and display the bin distribution based on both distributions
bin_distribution = generate_bin_distribution_from_counts(
    vertical_distribution_red, horizontal_distribution_red,
    vertical_distribution_green, horizontal_distribution_green
)

display_bin(bin_distribution)
