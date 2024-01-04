import cv2
import numpy as np
import matplotlib.pyplot as plt


# Συνάρτηση για την προσθήκη θορύβου salt & pepper
def noise_adder(image, density):
    # Μάσκα για τυχαία εμφάνιση του θορύβου εντός της εικόνας
    noise_mask = np.random.rand(*image.shape) < density

    # Μάσκες επιλογής των σημείων όπου έχουμε λευκό και μαύρο θόρυβο
    white_mask = noise_mask & (np.random.rand(*image.shape) < 0.5)
    black_mask = noise_mask & ~white_mask

    # Αντιγραφή της αρχικής εικόνας
    noisy_image = image.copy()

    # Θέσεις όπου τα pixel θα γίνουν λευκά και μαύρα
    noisy_image[white_mask] = 255
    noisy_image[black_mask] = 0

    return noisy_image


# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images\lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Προσθήκη θορύβου salt & pepper
density = 0.05
noisy_image = noise_adder(image, density)

# Εμφάνιση εικόνας πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Θορυβοποιημένη Εικόνα')

plt.show()

# Φίλτρα διαστάσεων 3x3, 5x5 και 7x7
filter_sizes = [3, 5, 7]

# Δημιουργία του καμβά του σχήματος
plt.figure(figsize=(15, 5))

for i, size in enumerate(filter_sizes):
    # Δημιουργία του φίλτρου μέσης τιμής και της φιλτραρισμένης εικόνας
    kernel = np.ones((size, size), np.float32) / (size * size)
    filtered_image = cv2.filter2D(noisy_image, -1, kernel)

    # Εμφάνιση των φιλτραρισμένων εικόνων
    plt.subplot(1, len(filter_sizes), i + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Φιλτραρισμένη Εικόνα {i + 1}.\nΦίτρο διαστάσεων {size}x{size}')

plt.show()