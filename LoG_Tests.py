import cv2
import numpy as np
from matplotlib import pyplot as plt

# Συνάρτηση εύρεσης ακμών με χρήση δεύτερων παραγώγων (LoG)
def log_method(image, variance, threshold):
    # Προσθήκη Γκαουσιανού φίλτρου στην εικόνα
    blurred = cv2.GaussianBlur(image, (0, 0), variance**2)

    # Υπολογισμός του τελεστή Laplace
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Αρχικοποίηση με 0 του πίνακα των ακμών
    edges = np.zeros_like(laplacian, dtype=np.uint8)
    
    # Εμφάνιση των σημείων των ακμών όπου laplacian > threshold
    edges[laplacian > threshold] = 255

    return edges


def main(image, variances, thresholds):
    # Δημιουργία του καμβά του σχήματος
    plt.figure(figsize=(4, 2))
    index = 1
    
    for i, variance in enumerate(variances):
        for j, threshold in enumerate(thresholds):
            # Εμφάνιση των επεξεργασμένων εικόνων
            log = log_method(image, variance, threshold)
            plt.subplot(1, 1, index)
            plt.imshow(log, cmap='gray')
            plt.title(f'Εικόνα με V = {variance} & Τ = {threshold}')
            
            # Αύξηση του δείκτη υπογραφικού πίνακα
            index += 1
    
    plt.show()

# Φόρτωση εικόνας από τον φάκελο image\
image_butterfly = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
image_cameraman = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
image_lenna = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Πειραματικές τιμές
variances = [1.3]
thresholds = [5]

# Κλήση συναρτήσεων main()
detection_butterfly = main(image_butterfly, variances, thresholds) 
# detection_cameraman = main(image_cameraman, variances, thresholds)
# detection_lenna = main(image_lenna, variances, thresholds)