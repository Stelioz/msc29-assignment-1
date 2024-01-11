import os
import cv2
import glob
import time
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
dataset_path = 'dataset/*.jpg'
image_paths = glob.glob(dataset_path)

# Ορισμός μετρικών για παρακολούθηση των υπολογισμών
num_images = len(image_paths)
num_calc = len(image_paths)**2

#  Αρχικοποίηση των λιστών για αποθήκευση των δεδομένων
brightness_histograms = []
lbp_histograms = []
i = 250
# Βρόχος για φόρτωση κάθε εικόνας του dataset
for i, image_path in enumerate(image_paths):
    # Φόρτωση εικόνας και μετατροπή σε grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Εξαγωγή χαρακτηριστικών με κλήση των συναρτήσεων
    brightness_histogram = normalized_brightness_histogram(image)
    lbp_histogram = normalized_lbp_histogram(image)

    # Υπολογισμός των τιμών των ιστογραμμάτων
    brightness_histogram = normalized_brightness_histogram(image)
    lbp_histogram = normalized_lbp_histogram(image)
    
    # Αποθήκευση των τιμών των ιστογραμμάτων στις λίστες τους
    brightness_histograms.append(brightness_histogram)
    lbp_histograms.append(lbp_histogram)

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
    print(f"Τα ιστογράμματα της εικόνας {image_path} αποθηκεύτηκε. Απομένουν {num_images - i - 1} υπολογισμοί.")

    # plt.show()
    plt.close()

print(f"Όλα τα ιστογράμματα αποθηκεύτηκαν!\n")
time.sleep(2)


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

            # Εκτύπωση των αποτελεσμάτων
            print(f"Οι αποστάσεις των εικόνων {dataset_path_1} και {dataset_path_2} είναι L1: {l1} και L2: {l2}.")
            print(f"Απομένουν {num_calc - (i * len(image_paths) + j + 1)} υπολογισμοί.")
  

# Τυχαία επιλογή 5 εικόνων από διαφορετικές κατηγορίες
categories = ['bougainvillea', 'tulips', 'orchids', 'peonies', 'hydrangeas', 'lilies', 'gardenias', 'garden_roses', 'daisies', 'hibiscus']
unique_labels = np.unique(categories)
labels = np.array(categories)
selected_images = []
for label in unique_labels:
    labeled_images = np.where(labels == label)[0]
    if len(labeled_images) > 0:
        selected_images.append(np.random.choice(labeled_images))

# Ανάκτηση των χαρακτηριστικών και των μετρικών για κάθε query_image
for query_image in selected_images:
    query_brightness = brightness_histograms[query_image]
    query_lbp = lbp_histograms[query_image]

    print(f"\nΕικόνα Αναζήτησης: {image_paths[query_image]}")

    # Χαρακτηριστικά και μετρικές
    features = ['A1', 'A2']
    metrics = ['B1', 'B2']
    all_distances = []

    # Υπολογισμός των συνδιασμών A1 - B1, A1 - B2, A2 - B1 και A2 - B2
    for feature, metric in product(features, metrics):
        distances = []

        # Προσπάλαση κάθε εικόνας του dataset
        for i, (brightness, lbp) in enumerate(zip(brightness_histograms, lbp_histograms)):
            
            # Αποκλεισμός της εικόνας αναζήτησης
            if i != query_image:
                
                # Επιλογή των χαρακτηριστικών για υπολογισμό των αποστάσεων
                if feature == 'A1':
                    feature_vector = brightness
                elif feature == 'A2':
                    feature_vector = lbp

                 # Επιλογή των μετρικών για υπολογισμό των αποστάσεων
                if metric == 'B1':
                     distance = l1_distance(query_brightness, feature_vector)
                elif metric == 'B2':
                    distance = l2_distance(query_brightness, feature_vector)

                # Αποθήκευση των αποστάσεων
                distances.append((distance, i))

        # Ταξινόμηση των αποστάσων και επιλογή των top-10
        distances.sort()
        top_10_results = distances[:10]

        # Υπολογισμός και αποθήκευση των αποστάσεων και των μέσω τιμών
        distances_values = [distance for distance, _ in top_10_results]
        mean_distance = np.mean(distances_values)
        all_distances[f"{feature} - {metric}"] = {'distances': distances_values, 'mean': mean_distance}


        # Εκτύπωση των αποτελεσματων για κάθε συνδιασμό χαρακτηριστικών - μετρικών
        print(f"Ανομοιότητα: {feature} - {metric}")
        for rank, (distance, result_index) in enumerate(top_10_results):
            print(f"{rank + 1}: {image_paths[result_index]} με Απόσταση: {distance}")

        # Εκτύπωση της μέσης τιμής για κάθε συνδιασμό χαρακτηριστικών - μετρικών
        print(f"Μέση Απόσταση {feature} - {metric}: {mean_distance}\n")