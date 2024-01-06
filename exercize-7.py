import cv2
import numpy as np
from matplotlib import pyplot as plt


# Συνάρτηση με χρήση πρώτων παραγώγων Sobel
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
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


# Συνάρτηση με χρήση πρώτων παραγώγων Roberts
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
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


# Συνάρτηση με χρήση πρώτων παραγώγων Prewitt
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
    _, threshold = cv2.threshold(edges_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return threshold


# Συνάρτηση με χρήση πρώτων παραγώγων Krisch
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
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    ]
    
    # Εφαρμογή των μασκών
    responses = []
    for mask in masks:
        response = cv2.filter2D(image_8, cv2.CV_64F, mask)
        responses.append(response)

    # Προσδιορισμός των συντελεστών με επιλογή της μέγιστης τιμής
    stack = np.stack(responses, axis=-1)
    edges = np.max(stack, axis=-1)

    # Convert edges to 8-bit for Otsu's method
    edges_uint8 = cv2.convertScaleAbs(edges)
    _, threshold = cv2.threshold(edges_uint8, experiment, 255, cv2.THRESH_BINARY)

    return threshold


def log_method(image, variance, threshold):
    # Apply LoG (Laplacian of Gaussian)
    blurred = cv2.GaussianBlur(image, (0, 0), variance)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.zeros_like(laplacian, dtype=np.uint8)
    edges[laplacian > threshold] = 255

    return edges


def canny_method(image):
    # Canny
    edges = cv2.Canny(image, 50, 150)

    return edges

# Φόρτωση εικόνας από τον φάκελο image\
image_butterfly = cv2.imread('images/butterfly.jpg', cv2.IMREAD_GRAYSCALE)
print(image_butterfly.shape if image_butterfly is not None else "Image not loaded")
# Experimentally determine optimal values for LoG
experiment_butterfly = 30 # Adjust as needed
variance_butterfly = 1.5  # Adjust as needed
threshold_butterfly = 20  # Adjust as needed

# Κλήση των συναρτήσεων sobel_method, roberts_method, prewitt_method και kirsch_method
sobel_butterfly = sobel_method(image_butterfly)
roberts_butterfly = roberts_method(image_butterfly)
prewitt_butterfly = prewitt_method(image_butterfly)
kirsch_butterfly = kirsch_method(image_butterfly, experiment_butterfly)

# Εμφάνιση εικόνας butterfly.jpg με χρήση πρώτων παραγώγων
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1), plt.imshow(image_butterfly, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(sobel_butterfly, cmap='gray'), plt.title('Sobel')
plt.subplot(2, 3, 3), plt.imshow(roberts_butterfly, cmap='gray'), plt.title('Roberts')
plt.subplot(2, 3, 4), plt.imshow(prewitt_butterfly, cmap='gray'), plt.title('Prewitt')
plt.subplot(2, 3, 5), plt.imshow(kirsch_butterfly, cmap='gray'), plt.title('Kirsch')
plt.show()

# Κλήση των συναρτήσεων log_method και canny_method
log_butterfly = log_method(image_butterfly, variance_butterfly, threshold_butterfly)
canny_butterfly = canny_method(image_butterfly)

# Εμφάνιση εικόνας butterfly.jpg με χρήση δεύτερων παραγώγων (LoG)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_butterfly, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(log_butterfly, cmap='gray'), plt.title('LoG')
plt.show()

# Εμφάνιση εικόνας butterfly.jpg με χρήση της μεθόδου Canny
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_butterfly, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(canny_butterfly, cmap='gray'), plt.title('Canny')
plt.show()


# Φόρτωση εικόνας από τον φάκελο image\
image_cameraman = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)
experiment_cameraman = 30 # Adjust as needed
variance_cameraman = 1.5  # Adjust as needed
threshold_cameraman = 20  # Adjust as needed

# Κλήση των συναρτήσεων sobel_method, roberts_method, prewitt_method και kirsch_method
sobel_cameraman = sobel_method(image_cameraman)
roberts_cameraman = roberts_method(image_cameraman)
prewitt_cameraman = prewitt_method(image_cameraman)
kirsch_cameraman = kirsch_method(image_cameraman,experiment_cameraman)

# Εμφάνιση εικόνας cameraman.bmp με χρήση πρώτων παραγώγων
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1), plt.imshow(image_cameraman, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(sobel_cameraman, cmap='gray'), plt.title('Sobel')
plt.subplot(2, 3, 3), plt.imshow(roberts_cameraman, cmap='gray'), plt.title('Roberts')
plt.subplot(2, 3, 4), plt.imshow(prewitt_cameraman, cmap='gray'), plt.title('Prewitt')
plt.subplot(2, 3, 5), plt.imshow(kirsch_cameraman, cmap='gray'), plt.title('Kirsch')
plt.show()

# Κλήση των συναρτήσεων log_method και canny_method
log_cameraman = log_method(image_cameraman, variance_cameraman, threshold_cameraman)
canny_cameraman = canny_method(image_cameraman)

# Εμφάνιση εικόνας cameraman.bmp με χρήση δεύτερων παραγώγων (LoG)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_cameraman, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(log_cameraman, cmap='gray'), plt.title('LoG')
plt.show()

# Εμφάνιση εικόνας cameraman.bmp με χρήση της μεθόδου Canny
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_cameraman, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(canny_cameraman, cmap='gray'), plt.title('Canny')
plt.show()


# Φόρτωση εικόνας από τον φάκελο image\
image_lenna = cv2.imread('images/lenna.bmp', cv2.IMREAD_GRAYSCALE)
experiment_lenna = 30 # Adjust as needed
variance_lenna = 1.5  # Adjust as needed
threshold_lenna = 20  # Adjust as needed

# Κλήση των συναρτήσεων sobel_method, roberts_method, prewitt_method και kirsch_method
sobel_lenna = sobel_method(image_lenna)
roberts_lenna = roberts_method(image_lenna)
prewitt_lenna = prewitt_method(image_lenna)
kirsch_lenna = kirsch_method(image_lenna, experiment_lenna)

# Εμφάνιση εικόνας lenna.bmp με χρήση πρώτων παραγώγων
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1), plt.imshow(image_lenna, cmap='gray'), plt.title('Original')
plt.subplot(2, 3, 2), plt.imshow(sobel_lenna, cmap='gray'), plt.title('Sobel')
plt.subplot(2, 3, 3), plt.imshow(roberts_lenna, cmap='gray'), plt.title('Roberts')
plt.subplot(2, 3, 4), plt.imshow(prewitt_lenna, cmap='gray'), plt.title('Prewitt')
plt.subplot(2, 3, 5), plt.imshow(kirsch_lenna, cmap='gray'), plt.title('Kirsch')
plt.show()

# Κλήση των συναρτήσεων log_method και canny_method
log_lenna = log_method(image_lenna, variance_lenna, threshold_lenna)
canny_lenna = canny_method(image_lenna)

# Εμφάνιση εικόνας lenna.bmp με χρήση δεύτερων παραγώγων (LoG)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_lenna, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(log_lenna, cmap='gray'), plt.title('LoG')
plt.show()

# Εμφάνιση εικόνας lenna.bmp με χρήση της μεθόδου Canny
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(image_lenna, cmap='gray'), plt.title('Original')
plt.subplot(1, 2, 2), plt.imshow(canny_lenna, cmap='gray'), plt.title('Canny')
plt.show()