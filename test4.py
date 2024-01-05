# Import libraries
import numpy as np
import cv2

# Define a function to perform mean shift segmentation
def mean_shift_segmentation(image, grid_size, window_radius, color_distance, max_iter):
    # Convert the image to CIELAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Get the image shape
    height, width, channels = lab_image.shape
    # Initialize the centers of the classes on a uniform grid
    centers = []
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            center = [i, j, 0]
            centers.append(center)
    # Initialize the mean shift vectors for each center
    mean_shifts = np.zeros((len(centers), channels))
    # Initialize the iteration counter
    iter_count = 0
    # Repeat until convergence or maximum iterations
    while True:
        # Create a copy of the current centers
        old_centers = np.copy(centers)
        # Loop through each center
        for i, center in enumerate(centers):
            # Initialize the sum of weighted pixels and the sum of weights
            weighted_sum = np.zeros(channels)
            weight_sum = 0
            # Loop through the window around the center
            for m in range(-window_radius, window_radius + 1):
                for n in range(-window_radius, window_radius + 1):
                    # Get the coordinates of the pixel in the window
                    x = center[0] + m
                    y = center[1] + n
                    # Check if the pixel is inside the image
                    if 0 <= x < height and 0 <= y < width:
                        # Get the pixel value
                        pixel = lab_image[int(center[0] + m), int(center[1] + n)]
                        # Calculate the color distance between the pixel and the center
                        color_dist = np.linalg.norm(pixel - center)
                        # Check if the color distance is within the threshold
                        if color_dist < color_distance:
                            # Update the weighted sum and the weight sum
                            weighted_sum += pixel * color_dist
                            weight_sum += color_dist
            # Calculate the new center as the mean of the weighted pixels
            if weight_sum > 0:
                new_center = weighted_sum / weight_sum
            else:
                new_center = center
            # Update the center and the mean shift vector
            centers[i] = new_center
            mean_shifts[i] = [a - b for a, b in zip(new_center, center)]
        # Increment the iteration counter
        iter_count += 1
        # Check for convergence or maximum iterations
        if np.linalg.norm(mean_shifts) < 1e-3 or iter_count >= max_iter:
            break

    # Merge the centers that coincide
    unique_centers = np.unique(centers, axis=0)
    # Assign each pixel to the nearest center
    labels = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixel = lab_image[i, j]
            # Find the nearest center for the pixel
            nearest_center = np.argmin(np.linalg.norm(unique_centers - pixel, axis=1))
            # Assign the label of the nearest center to the pixel
            labels[i, j] = nearest_center

    # Return the labels and the number of segments
    return labels, len(unique_centers)

# Load the image
image = cv2.imread('images/butterfly.jpg')

# Define the parameters for mean shift segmentation
grid_size = 30  # Size of the grid for initial centers
window_radius = 20  # Radius of the window for mean shift
color_distance = 40  # Maximum color distance for pixel inclusion
max_iter = 50  # Maximum number of iterations

# Apply mean shift segmentation
labels, n_segments = mean_shift_segmentation(image, grid_size, window_radius, color_distance, max_iter)

# Print the number of segments
print(f'Number of segments: {n_segments}')

# Plot the original and segmented images
cv2.imshow('Original image', image)
cv2.imshow('Segmented image', labels * 255 // n_segments)  # Rescale the labels to range 0-255
cv2.waitKey(0)
cv2.destroyAllWindows()