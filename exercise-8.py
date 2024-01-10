import os
import cv2
import csv
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
dataset_path = 'testflowers/*.jpg'
image_paths = glob.glob(dataset_path)

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

# Ορισμός του path των αποστάσεων L1 και L2
distances = 'results/distances.csv' 

# Υπολογισμός και αποθήκευση των αποτελεσμάτων σε ένα αρχείο CSV
with open(distances, "w", newline="", encoding='utf-8') as csv_file:
    # Δημιουργία του CSV αρχείου και των επικεφαλίδων
    dis_results = csv.writer(csv_file)
    dis_results.writerow(["Εικόνα 1", "Εικόνα 2", "L1", "L2"])

    # Συγκρίνουμε ανά ζεύγη τις εικόνες
    for i, dataset_path_1 in enumerate(image_paths):
        for j, dataset_path_2 in enumerate(image_paths):
            # Αποτροπή σύγκρισης μίας εικόνας με τον ευατό της
            if i != j:
                # Φόρτωση εικόνων και μετατροπή σε grayscale
                image_1 = cv2.imread(dataset_path_1, cv2.IMREAD_GRAYSCALE)
                image_2 = cv2.imread(dataset_path_2, cv2.IMREAD_GRAYSCALE)

                # Υπολογισμός των ιστογραμμάτων και των L1, L2
                histogram_1 = normalized_brightness_histogram(image_1)
                histogram_2 = normalized_brightness_histogram(image_2)
                l1 = l1_distance(histogram_1, histogram_2)
                l2 = l2_distance(histogram_1, histogram_2)

                # Εγγραφή των αποτελεσμάτων στο αρχείο CSV και εκτύπωσή τους
                dis_results.writerow([f"{dataset_path_1}", f"{dataset_path_2}", f"{l1}", f"{l2}"])
                print(f"Οι αποστάσεις μεταξύ των εικόνων {dataset_path_1} και {dataset_path_2} είναι L1: {l1} και L2: {l2}.")

print(f"Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο: ../{distances}")

# Αρχικοποίηση των λιστών για αποθήκευση των δεδομένων
brightness_histograms = []
lbp_histograms = []
categories = ['bougainvillea', 'tulips', 'orchids', 'peonies', 'hydrangeas',
              'lilies', 'gardenias', 'gardenroses', 'daisies', 'hibiscus']

# Calculate histograms for all images in the dataset
for image_path in image_paths:
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Compute brightness and LBP histograms
    brightness_histogram = normalized_brightness_histogram(image)
    lbp_histogram = normalized_lbp_histogram(image)
    
    # Append histograms to the respective lists
    brightness_histograms.append(brightness_histogram)
    lbp_histograms.append(lbp_histogram)

# Randomly select 5 images from different categories
unique_labels = np.unique(categories)
selected_images = []

for category in unique_labels:
    labeled_images = np.where(np.array(categories) == category)[0]
    if len(labeled_images) > 0:
        selected_images.append(np.random.choice(labeled_images))

# Perform retrieval for each query image and combination of features/metrics
for query_image in selected_images:
    query_brightness = brightness_histograms[query_image]
    query_lbp = lbp_histograms[query_image]

    print(f"\nQuery Image: {image_paths[query_image]}")

    feature_combinations = ['A1', 'A2']
    metric_combinations = ['B1', 'B2']

    for feature, metric in product(feature_combinations, metric_combinations):
        distances = []

        for i, (brightness, lbp) in enumerate(zip(brightness_histograms, lbp_histograms)):
            if i != query_image:  # Exclude the query image itself
                if feature == 'A1':
                    feature_vector = brightness
                elif feature == 'A2':
                    feature_vector = lbp

                if metric == 'B1':
                     distance = l1_distance(query_brightness, feature_vector)
                elif metric == 'B2':
                    distance = l2_distance(query_brightness, feature_vector)
 
                distances.append((distance, i))

        # Sort distances and print top-10 retrieval results
        distances.sort()
        top_10_results = distances[:10]
        print(f"\n{feature} - {metric} Dissimilarity")
        for rank, (distance, result_index) in enumerate(top_10_results):
            print(f"Βαθμός {rank+1}: {image_paths[result_index]} (Απόσταση: {distance})")

# # Ορισμός του path των συγκρίσεων
# comparisons = 'results/comparisons.csv' 

# # Υπολογισμός και αποθήκευση των αποτελεσμάτων σε ένα αρχείο CSV
# with open(comparisons, "w", newline="", encoding='utf-8') as csv_file:
#     # Δημιουργία του CSV αρχείου και των επικεφαλίδων
#     comp_results = csv.writer(csv_file)
#     comp_results.writerow(["Εικόνα Αναζήτησης", "Εικόνα Σύγκρισης", "L1", "L2"])

#     # Συγκρίνουμε ανά ζεύγη τις εικόνες
#     for i, dataset_path_1 in enumerate(image_paths):
#         for j, dataset_path_2 in enumerate(image_paths):
#             # Αποτροπή σύγκρισης μίας εικόνας με τον ευατό της
#             if i != j:
#                 # Φόρτωση εικόνων και μετατροπή σε grayscale
#                 image_1 = cv2.imread(dataset_path_1, cv2.IMREAD_GRAYSCALE)
#                 image_2 = cv2.imread(dataset_path_2, cv2.IMREAD_GRAYSCALE)

#                 # Υπολογισμός των ιστογραμμάτων και των L1, L2
#                 histogram_1 = normalized_brightness_histogram(image_1)
#                 histogram_2 = normalized_brightness_histogram(image_2)
#                 l1 = l1_distance(histogram_1, histogram_2)
#                 l2 = l2_distance(histogram_1, histogram_2)

#                 # Εγγραφή των αποτελεσμάτων στο αρχείο CSV και εκτύπωσή τους
#                 comp_results.writerow([f"{dataset_path_1}", f"{dataset_path_2}", f"{l1}", f"{l2}"])
#                 print(f"Οι αποστάσεις μεταξύ των εικόνων {dataset_path_1} και {dataset_path_2} είναι L1: {l1} και L2: {l2}.")

# print(f"Τα αποτελέσματα αποθηκεύτηκαν στο αρχείο: {distances_path}")
    


