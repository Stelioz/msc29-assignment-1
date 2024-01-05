import cv2
import numpy as np
import matplotlib.pyplot as plt



# Συνάρτηση τμηματοποίησης Mean Shift
def mean_shift_segmentation(image, grid_size):
    # Μετατροπή των χρωμάτων στον χώρο CIELAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    
    
   
    

    return 




def initialize_mean_shift(image, grid_size):
    height, width, channels = image.shape
    
    # Επιλογή των αρχικών κέντρων m(x,y)
    centers = []
    for i in range(0, height, grid_size, window_radius, color_distance):
        for j in range(0, width, grid_size):
            # Η τιμή του κάθε pixel γίνεται append στον πίνακα των κέντρων
            center = image[i, j].copy()
            centers.append(center)

    # Αρχικοποίηση και υπολογισμός των διανυσμάτων mean-shift
    mean_shift_vectors = []
    for center in centers:
        x, y = center
        color_vector = image[y, x].astype(np.float32)

        # Calculate mean-shift vector in the 3D color space
        mean_shift_vector = cv2.pyrMeanShiftFiltering(
            image, spatial_radius = 30, color_radius = 20, termcrit=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 1))

        mean_shift_vectors.append(mean_shift_vector)

    return centers, mean_shift_vectors, color_vector












# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images/butterfly.jpg')

# Καθορισμός του μεγέθους του πλέγματος
grid_size = 30
window_radius = 20
color_distance = 40