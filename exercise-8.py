import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from skimage.feature import local_binary_pattern


# Συνάρτηση για τον υπολογισμό του ιστόγραμματος φωτεινότητας
def normalized_brightness_histogram(image):
    # Δημιουργία του ιστόγραμματος
    histogram, _ = np.histogram(image.flatten(), 256, [0, 256])
    norm_histogram = histogram / histogram.sum()
    
    return norm_histogram


# Συνάρτηση για τον υπολογισμό του ιστόγραμματος τιμών υφής
def normalized_lbp_histogram(image):
    # Υπολογισμός του LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

    # Δημιουργία του ιστόγραμματος
    histogram, _ = np.histogram(lbp.flatten(), 256, [0, 256])
    norm_histogram = histogram / histogram.sum()
    
    return norm_histogram


# Συνάρτηση για τον υπολογισμό του L1
def l1_distance(f1, f2):
    # Υπολογισμός του L1
    l1 = np.sum(np.abs(f1 - f2))

    return l1


# Συνάρτηση για τον υπολογισμό του L2
def l2_distance(f1, f2):
    # Υπολογισμός του L2
    l2 = np.sqrt(np.sum((f1 - f2) ** 2))
    
    return l2


# Ορισμός του path του dataset
dataset_path = 'flowers/*.jpg'
image_paths = glob.glob(dataset_path)

# Αρχικοποίηση των λιστών για αποθήκευση των δεδομένων
brightness_histograms = []
lbp_histograms = []
labels = []

# Βρόχος για φόρτωση κάθε εικόνας του dataset
for i, image_path in enumerate(image_paths):
    # Φόρτωση εικόνας και μετατροπή σε grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Εξαγωγή χαρακτηριστικών με κλήση των συναρτήσεων
    brightness_histogram = normalized_brightness_histogram(image)
    lbp_histogram = normalized_lbp_histogram(image)

    # Εμφάνιση των ιστογραμμάτων κάθε εικόνας
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(brightness_histogram)
    plt.title("Κανονικοποιημένο Ιστόγραμμα Φωτεινότητας")
    plt.subplot(1, 2, 2)
    plt.plot(lbp_histogram)
    plt.title("Κανονικοποιημένο Ιστόγραμμα LBP")
    
    # Το όνομα του plot θα σχετίζεται με το όνομα της εξεταζόμενης εικόνας
    image_filename = os.path.basename(image_path)

    # Αποθήκευση των ιστογραμμάτων κάθε εικόνας στον φάκελο plots/
    plot_filename = os.path.join("plots/", f"{os.path.splitext(image_filename)[0]}_plot.png")
    plt.savefig(plot_filename)

    plt.show()

    print(f"Τα ιστογράμματα της εικόνας {image_filename} αποθηκεύτηκαν.")



    # Extract label from the folder name using os.path
    label = os.path.basename(os.path.dirname(image_path))
    labels.append(label)

    # Αποθήκευση των χαρακτηριστικών στις λίστες
    brightness_histograms.append(brightness_histogram)
lbp_histograms.append(lbp_histogram)

# # Μετατροπή των λιστών σε πίνακες του NumPy
# brightness_histograms = np.array(brightness_histograms)
# lbp_histograms = np.array(lbp_histograms)

# # Check if there are features available for plotting
# if brightness_histograms.shape[0] > 0 and lbp_histograms.shape[0] > 0:
#     # Randomly select 5 images from different categories
#     unique_labels = np.unique(labels)
#     selected_images = []

#     for label in unique_labels:
#         images_with_label = np.where(labels == label)[0]
#         if len(images_with_label) > 0:
#             selected_images.append(np.random.choice(images_with_label))

# # Perform retrieval for each query image and combination of features/metrics
# for query_index in selected_images:
#     query_brightness = brightness_histograms[query_index]
#     query_lbp = lbp_histograms[query_index]

#     print(f"\nQuery Image: {image_paths[query_index]}")

#     feature_combinations = ['A1', 'A2']
#     metric_combinations = ['B1', 'B2']

#     for feature, metric in product(feature_combinations, metric_combinations):
#         distances = []

#         for i, (brightness, lbp) in enumerate(zip(brightness_histograms, lbp_histograms)):
#             if i != query_index:  # Exclude the query image itself
#                 if feature == 'A1':
#                     feature_vector = brightness
#                 elif feature == 'A2':
#                     feature_vector = lbp

#                 if metric == 'B1':
#                      distance = l1_distance(query_brightness, feature_vector)
#                 elif metric == 'B2':
#                     distance = distance(query_brightness, feature_vector)
 
#                 distances.append((distance, i))

#         # Sort distances and print top-10 retrieval results
#         distances.sort()
#         top_10_results = distances[:10]
#         print(f"\n{feature} - {metric} Dissimilarity")
#         for rank, (distance, result_index) in enumerate(top_10_results):
#             print(f"Βαθμός {rank+1}: {image_paths[result_index]} (Απόσταση: {distance})")
    
# # Example: Calculate dissimilarity metrics for the first two feature vectors
# f1_brightness = brightness_histograms[0]
# f2_brightness = brightness_histograms[1]
# l1_brightness_distance = l1_distance(f1_brightness, f2_brightness)
# l2_brightness_distance = l2_distance(f1_brightness, f2_brightness)

# f1_lbp = lbp_histograms[0]
# f2_lbp = lbp_histograms[1]
# l1_lbp_distance = l1_distance(f1_lbp, f2_lbp)
# l2_lbp_distance = l2_distance(f1_lbp, f2_lbp)
    
# # Print or use the extracted features as needed
# print("Brightness Histograms shape:", brightness_histograms.shape)
# print("LBP Histograms shape:", lbp_histograms.shape)

# # Print or use the calculated distances as needed
# print("L1 Brightness Distance:", l1_brightness_distance)
# print("L2 Brightness Distance:", l2_brightness_distance) 
# print("L1 LBP Distance:", l1_lbp_distance)
# print("L2 LBP Distance:", l2_lbp_distance)