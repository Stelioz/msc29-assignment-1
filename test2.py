import cv2
import numpy as np
from matplotlib import pyplot as plt

# A. Feature Extraction
def extract_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # A1. Normalized brightness histogram
    hist_brightness, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256], density=True)

    # A2. Local Binary Pattern (LBP) histogram
    lbp = cv2.LBP_create()
    lbp_values = lbp.compute(gray)[0]
    hist_lbp, _ = np.histogram(lbp_values.flatten(), bins=256, range=[0, 256], density=True)

    return np.concatenate([hist_brightness, hist_lbp])

# B. Dissimilarity Metrics
def l1_distance(f1, f2):
    return np.sum(np.abs(f1 - f2))

def l2_distance(f1, f2):
    return np.sqrt(np.sum((f1 - f2) ** 2))

# C. Top-10 Similar Images
def top_10_similar_images(query_features, feature_matrix, metric_function):
    distances = []
    for features in feature_matrix:
        distances.append(metric_function(query_features, features))
    
    sorted_indices = np.argsort(distances)

    return sorted_indices[1:11]  # Exclude the query image itself

# Load Flower Image Dataset
dataset_path = 'path/to/flower_dataset'  # Replace with the actual path
categories = ['Bougainvillea', 'Tulips', 'Orchids', 'Peonies', 'Hydrangeas',
              'Lilies', 'Gardenias', 'Garden Roses', 'Daisies', 'Hibiscus']

# Randomly select 5 images from different categories
random_query_images = []

for _ in range(5):
    category = np.random.choice(categories)
    image_files = os.listdir(f'{dataset_path}/{category}')
    image_file = np.random.choice(image_files)
    image_path = f'{dataset_path}/{category}/{image_file}'
    query_image = cv2.imread(image_path)
    random_query_images.append((query_image, category))

# Extract features for each image in the dataset
all_features = []
labels = []

for category in categories:
    image_files = os.listdir(f'{dataset_path}/{category}')

    for image_file in image_files:
        image_path = f'{dataset_path}/{category}/{image_file}'
        image = cv2.imread(image_path)
        features = extract_features(image)
        all_features.append(features)
        labels.append(category)

all_features_matrix = np.array(all_features)

# C. Top-10 Similar Images for each query image
for query_image, query_category in random_query_images:
    print(f"\nQuery Image Category: {query_category}")

    for feature_type in ['A1', 'A2']:
        for metric_type in ['B1', 'B2']:
            query_features = extract_features(query_image)

            if feature_type == 'A1':
                query_features = query_features[:256]  # Use only brightness histogram

            if feature_type == 'A2':
                query_features = query_features[256:]  # Use only LBP histogram

            metric_function = l1_distance if metric_type == 'B1' else l2_distance

            top_10_indices = top_10_similar_images(query_features, all_features_matrix, metric_function)

            print(f"\nResults for Feature {feature_type} and Metric {metric_type}:")
            for i, index in enumerate(top_10_indices, 1):
                result_category = labels[index]
                print(f"{i}. Category: {result_category}")

# You can compare and evaluate the results here.
