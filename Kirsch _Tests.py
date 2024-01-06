import cv2
import numpy as np
from matplotlib import pyplot as plt

# Συνάρτηση εύρεσης ακμών με χρήση πρώτων παραγώγων Krisch
def kirsch_method(image, experiment):
    # Μετατροπή της εικόνας σε 8bit
    image_8 = cv2.convertScaleAbs(image)
    
    # Μάσκες του Krisch
    masks = [
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])]
    
    # Εφαρμογή των μασκών
    responses = []
    for mask in masks:
        response = cv2.filter2D(image_8, cv2.CV_64F, mask)
        responses.append(response)

    # Προσδιορισμός των συντελεστών με επιλογή της μέγιστης τιμής
    stack = np.stack(responses, axis=-1)
    edges = np.max(stack, axis=-1)
    
    # Υπολογισμός του κατωφλίου με πειραματικό τρόπο
    _, threshold = cv2.threshold(edges, experiment * 100, 255, cv2.THRESH_BINARY)

    return threshold

def main(image, experiments):
    # Δημιουργία του καμβά του σχήματος
    plt.figure(figsize=(16, 8))
    plt.subplot(3, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Αρχική Εικόνα')
    
    for i, experiment in enumerate(experiments):
        # Εμφάνιση των επεξεργασμένων εικόνων
        kirsch = kirsch_method(image, experiment)
        plt.subplot(3, 3, i + 2)
        plt.imshow(kirsch, cmap='gray')
        plt.title(f'Επεξεργασμενη Εικόνα με Τ = {experiment}')
    
    plt.show()

# Φόρτωση εικόνας από τον φάκελο image\
image_butterfly = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
image_cameraman = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
image_lenna = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Πειραματικές τιμές
experiments = [3, 4, 5, 6, 7, 8, 9, 10]

# Κλήση συναρτήσεων main()
detection_butterfly = main(image_butterfly, experiments)
detection_cameraman = main(image_cameraman, experiments)
detection_lenna = main(image_lenna, experiments)