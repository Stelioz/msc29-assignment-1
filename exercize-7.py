import cv2
import numpy as np
from matplotlib import pyplot as plt


# Συνάρτηση εύρεσης ακμών με χρήση πρώτων παραγώγων Sobel
def sobel_method(image):
    # Μετατροπή της εικόνας σε 8bit
    image_8 = cv2.convertScaleAbs(image)

    # Μάσκες του Sobel
    PGr = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    PGc = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Προσδιορισμός των συντελεστών των μασκών ανίχνευσης ακμών 
    Gr = cv2.filter2D(image_8, cv2.CV_64F, PGr)
    Gc = cv2.filter2D(image_8, cv2.CV_64F, PGc)
    edges = np.sqrt(Gr**2 + Gc**2)

    # Μετατροπή των ακμών σε 8bit για χρήση από τη μέθοδο κατωφλίωσης Otsu
    edges_8 = cv2.convertScaleAbs(edges)

    # Υπολογισμός του κατωφλίου με χρήση της μεθόδου κατωφλίωσης Otsu
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


# Συνάρτηση εύρεσης ακμών με χρήση πρώτων παραγώγων Roberts
def roberts_method(image):
    # Μετατροπή της εικόνας σε 8bit
    image_8 = cv2.convertScaleAbs(image)

    # Μάσκες του Roberts
    PGr = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    PGc = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])

    # Προσδιορισμός των συντελεστών των μασκών ανίχνευσης ακμών 
    Gr = cv2.filter2D(image_8, cv2.CV_64F, PGr)
    Gc = cv2.filter2D(image_8, cv2.CV_64F, PGc)
    edges = np.sqrt(Gr**2 + Gc**2)

    # Μετατροπή των ακμών σε 8bit για χρήση από τη μέθοδο κατωφλίωσης Otsu
    edges_8 = cv2.convertScaleAbs(edges)

    # Υπολογισμός του κατωφλίου με χρήση της μεθόδου κατωφλίωσης Otsu
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


# Συνάρτηση εύρεσης ακμών με χρήση πρώτων παραγώγων Prewitt
def prewitt_method(image):
    # Μετατροπή της εικόνας σε 8bit
    image_8 = cv2.convertScaleAbs(image)

    # Μάσκες του Prewitt
    PGr = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    PGc = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Προσδιορισμός των συντελεστών των μασκών ανίχνευσης ακμών 
    Gr = cv2.filter2D(image_8, cv2.CV_64F, PGr)
    Gc = cv2.filter2D(image_8, cv2.CV_64F, PGc)
    edges = np.sqrt(Gr**2 + Gc**2)

    # Μετατροπή των ακμών σε 8bit για χρήση από τη μέθοδο κατωφλίωσης Otsu
    edges_8 = cv2.convertScaleAbs(edges)

    # Υπολογισμός του κατωφλίου με χρήση της μεθόδου κατωφλίωσης Otsu
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


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
    _, threshold = cv2.threshold(edges, experiment, 255, cv2.THRESH_BINARY)

    return threshold


# Συνάρτηση εύρεσης ακμών με χρήση δεύτερων παραγώγων (LoG)
def log_method(image, variance, threshold):
    # Προσθήκη Γκαουσιανού φίλτρου στην εικόνα
    blurred = cv2.GaussianBlur(image, (0, 0), variance)

    # Υπολογισμός του τελεστή Laplace
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Αρχικοποίηση με 0 του πίνακα των ακμών
    edges = np.zeros_like(laplacian, dtype=np.uint8)
    
    # Εμφάνιση των σημείων των ακμών όπου laplacian > threshold
    edges[laplacian > threshold] = 255

    return edges


# Συνάρτηση με χρήση του αλγορίθμου του Canny
def canny_method(image):
    # Χρήση low = 50, high = 150 ως default τιμών
    edges = cv2.Canny(image, 50, 150)

    return edges


def main(image):
    experiment = 30 # Πειραματισμός
    variance = 1.5  # Πειραματισμός
    threshold = 20  # Πειραματισμός

    # Κλήση των συναρτήσεων sobel_method, roberts_method, prewitt_method και kirsch_method
    sobel = sobel_method(image)
    roberts = roberts_method(image)
    prewitt = prewitt_method(image)
    kirsch = kirsch_method(image, experiment)

    # Εμφάνιση εικόνας cameraman.bmp με χρήση πρώτων παραγώγων
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(2, 3, 2), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
    plt.subplot(2, 3, 3), plt.imshow(roberts, cmap='gray'), plt.title('Roberts')
    plt.subplot(2, 3, 4), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
    plt.subplot(2, 3, 5), plt.imshow(kirsch, cmap='gray'), plt.title('Kirsch')
    plt.show()

    # Κλήση των συναρτήσεων log_method και canny_method
    log = log_method(image, variance, threshold)
    canny = canny_method(image)

    # Εμφάνιση εικόνας cameraman.bmp με χρήση δεύτερων παραγώγων (LoG)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(log, cmap='gray'), plt.title('LoG')
    plt.show()

    # Εμφάνιση εικόνας cameraman.bmp με χρήση της μεθόδου Canny
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(canny, cmap='gray'), plt.title('Canny')
    plt.show()

    return


image_butterfly = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
image_cameraman = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
image_lenna = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)

detection_butterfly = main(image_butterfly) 
detection_cameraman = main(image_cameraman)
detection_lenna = main(image_lenna)