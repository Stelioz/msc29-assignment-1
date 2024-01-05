import cv2
import numpy as np

def initialize_mean_shift(image, grid_size):
    height, width, _ = image.shape
    y, x = np.meshgrid(np.linspace(0, height-1, grid_size),
                       np.linspace(0, width-1, grid_size))
    initial_centers = np.column_stack((x.flatten(), y.flatten())).astype(np.int32)

    mean_shift_vectors = []
    for center in initial_centers:
        mean_shift_vector = calculate_mean_shift_vector(image, center)
        mean_shift_vectors.append(mean_shift_vector)

    return initial_centers, mean_shift_vectors

def calculate_mean_shift_vector(image, center):
    x_center, y_center = center
    spatial_radius = 20
    color_radius = 40

    # Define the spatial and color windows
    spatial_window = np.array([x_center, y_center])
    color_window = image[y_center, x_center]

    # Iterate until convergence or maximum iterations (T)
    max_iterations = 50
    epsilon = 1
    for _ in range(max_iterations):
        # Calculate the spatial and color distances
        spatial_distance = np.linalg.norm(spatial_window - center)
        color_distance = np.linalg.norm(color_window - image[y_center, x_center])

        # Check for convergence
        if spatial_distance < epsilon and color_distance < epsilon:
            break

        # Update the spatial and color windows
        spatial_window = np.mean(np.argwhere(np.linalg.norm(np.indices(image.shape[:2]) - spatial_window.reshape(-1, 1, 1), axis=0) < spatial_radius), axis=0)
        color_window = np.mean(image[
            max(0, int(y_center - color_radius)):min(image.shape[0], int(y_center + color_radius) + 1),
            max(0, int(x_center - color_radius)):min(image.shape[1], int(x_center + color_radius) + 1)
        ], axis=(0, 1))

    # Calculate the mean-shift vector
    mean_shift_vector = spatial_window - center
    return mean_shift_vector

def merge_duplicate_centers(centers):
    unique_centers = []
    for center in centers:
        if all(np.any(np.abs(center - unique_center) > 1) for unique_center in unique_centers):
            unique_centers.append(center)
    return unique_centers

def mean_shift_segmentation(image_path, grid_size):
    original_image = cv2.imread(image_path)

    # Check if the image is loaded correctly
    if original_image is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return 0

    lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)

    initial_centers, mean_shift_vectors = initialize_mean_shift(lab_image, grid_size)

    # Apply mean-shift segmentation for each initial center
    segmented_image = np.zeros_like(original_image, dtype=np.uint8)
    centers = []
    for center, mean_shift_vector in zip(initial_centers, mean_shift_vectors):
        x_center, y_center = center
        # Apply mean-shift vector to each pixel in the neighborhood
        mask = np.linalg.norm(lab_image - lab_image[y_center, x_center], axis=2) < 40
        segmented_image += np.where(mask[:, :, np.newaxis], 255, 0).astype(np.uint8)

        # Update the center using mean-shift vector
        new_center = center + mean_shift_vector.astype(np.int32)
        centers.append(new_center)

    # Merge duplicate centers
    centers = merge_duplicate_centers(centers)

    # Display the result and print the number of detected objects
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(centers)

# Example usage
image_path = 'images/butterfly.jpg'
grid_size = 30
detected_objects = mean_shift_segmentation(image_path, grid_size)
print(f"Total number of detected objects: {detected_objects}")