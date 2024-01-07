import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import glob
import os
from itertools import product

# Function to extract normalized brightness histogram
def normalized_brightness_histogram(image):
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    norm_hist = hist / hist.sum()
    return norm_hist

# Function to extract normalized histogram of Local Binary Pattern (LBP)
def normalized_lbp_histogram(image):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

    # Compute histogram
    hist, _ = np.histogram(lbp.flatten(), 256, [0, 256])
    norm_hist = hist / hist.sum()
    return norm_hist

# Function to calculate L1 dissimilarity
def calculate_l1_distance(f1, f2):
    return np.sum(np.abs(f1 - f2))

# Function to calculate L2 dissimilarity
def calculate_l2_distance(f1, f2):
    return np.sqrt(np.sum((f1 - f2) ** 2))

# Path to the Flower Image Dataset
dataset_path = 'flowers/*.jpg'

# Initialize lists to store features and labels
brightness_histograms = []
lbp_histograms = []
labels = []

# Loop through each image in the dataset
image_paths = glob.glob(dataset_path)

# Check if there are images available
if not image_paths:
    print("No images found in the specified path.")
else:
    for image_path in image_paths:
        # Read the image
        image = cv2.imread(image_path)

        # Check if the image is successfully loaded
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract features
        brightness_hist = normalized_brightness_histogram(gray_image)
        lbp_hist = normalized_lbp_histogram(gray_image)

        # Extract label from the folder name using os.path
        label = os.path.basename(os.path.dirname(image_path))
        labels.append(label)

        # Append features
        brightness_histograms.append(brightness_hist)
        lbp_histograms.append(lbp_hist)

# Convert lists to NumPy arrays
brightness_histograms = np.array(brightness_histograms)
lbp_histograms = np.array(lbp_histograms)

# Check if there are features available for plotting
if brightness_histograms.shape[0] > 0 and lbp_histograms.shape[0] > 0:
    # Randomly select 5 images from different categories
    unique_labels = np.unique(labels)
    selected_images = []

    for label in unique_labels:
        images_with_label = np.where(labels == label)[0]
        if len(images_with_label) > 0:
            selected_images.append(np.random.choice(images_with_label))

    # Perform retrieval for each query image and combination of features/metrics
    for query_index in selected_images:
        query_brightness = brightness_histograms[query_index]
        query_lbp = lbp_histograms[query_index]

        print(f"\nQuery Image: {image_paths[query_index]}")

        feature_combinations = ['A1', 'A2']
        metric_combinations = ['B1', 'B2']

        for feature, metric in product(feature_combinations, metric_combinations):
            distances = []

            for i, (brightness, lbp) in enumerate(zip(brightness_histograms, lbp_histograms)):
                if i != query_index:  # Exclude the query image itself
                    if feature == 'A1':
                        feature_vector = brightness
                    elif feature == 'A2':
                        feature_vector = lbp

                    if metric == 'B1':
                        distance = calculate_l1_distance(query_brightness, feature_vector)
                    elif metric == 'B2':
                        distance = calculate_l2_distance(query_brightness, feature_vector)
 
                    distances.append((distance, i))

            # Sort distances and print top-10 retrieval results
            distances.sort()
            top_10_results = distances[:10]
            print(f"\n{feature} - {metric} Dissimilarity")
            for rank, (distance, result_index) in enumerate(top_10_results):
                print(f"Rank {rank+1}: {image_paths[result_index]} (Distance: {distance})")
    
    # Example: Calculate dissimilarity metrics for the first two feature vectors
    f1_brightness = brightness_histograms[0]
    f2_brightness = brightness_histograms[1]
    l1_brightness_distance = calculate_l1_distance(f1_brightness, f2_brightness)
    l2_brightness_distance = calculate_l2_distance(f1_brightness, f2_brightness)

    f1_lbp = lbp_histograms[0]
    f2_lbp = lbp_histograms[1]
    l1_lbp_distance = calculate_l1_distance(f1_lbp, f2_lbp)
    l2_lbp_distance = calculate_l2_distance(f1_lbp, f2_lbp)
    
    # Print or use the extracted features as needed
    print("Brightness Histograms shape:", brightness_histograms.shape)
    print("LBP Histograms shape:", lbp_histograms.shape)

    # Print or use the calculated distances as needed
    print("L1 Brightness Distance:", l1_brightness_distance)
    print("L2 Brightness Distance:", l2_brightness_distance) 
    print("L1 LBP Distance:", l1_lbp_distance)
    print("L2 LBP Distance:", l2_lbp_distance)

    # Plot histograms for a sample image (change the index as needed)
    sample_index = 0
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(brightness_histograms[sample_index])
    plt.title("Normalized Brightness Histogram")
    plt.subplot(1, 2, 2)
    plt.plot(lbp_histograms[sample_index])
    plt.title("Normalized LBP Histogram")
    plt.show()
else:
    print("No features available for plotting.")

