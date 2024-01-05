import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft2, ifft2


# Συνάρτηση για την προθήκη Γκαουσιανού θορύβου
def noise_adder(image, mean, variance):
    # Δημιουργία Gaussian θορύβου με βάση τα δοθέντα mean και variance
    noise = np.random.normal(mean, variance, image.shape)
    
    # Προσθήκη του θορύβου στην αρχική εικόνα
    temp_image = image + noise

    # Κανονικοποίηση των εικονοστοιχείων ώστε να είναι μεταξύ 0 και 255
    noisy_image = np.clip(temp_image, 0, 255).astype(np.uint8)

    return noisy_image, noise


# Συνάρτηση για τη δημιουργία των χαμηλοπερατών φίλτρων Butterworth
def butter_lpf(image, fc, order):
    M, N = image.shape
    # Αρχικοποίηση του πίνακα
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            # Χρήση του μαθηματικού τύπου
            D = np.sqrt((u - M / 2)**2 + (v - N / 2)**2)
            H[u, v] = 1 / (1 + (D / fc)**(2*order))
    
    return H


# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images\lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Υπολογισμός του 2D μετασχηματισμού Fourier (DFT)
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Προσθήκη Γκαουσιανού θορύβου
mean = 0
variance = 10
noisy_image, noise = noise_adder(image, mean, variance)

# Εμφάνιση εικόνας πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Θορυβοποιημένη Εικόνα')

plt.show()

# Χαμηλοπερατά φίλτρα Butterworth 3ης, 5ης και 7ης τάξης
orders = [3, 5, 7]
# Ορισμός της συχνότητας αποκοπής
fc = 30 # Hz

# Δημιουργία του καμβά του σχήματος
plt.figure(figsize=(15, 5))

for i, order in enumerate(orders):
    # Δημιουργία του φίλτρου
    filter = butter_lpf(image, fc, order)

    # Εφαρμογή του φίλτρου στο φάσμα της μετασχηματισμένης εικόνας
    idft_shift = dft_shift * filter

    # Αντίστροφος μετασχηματισμός Fourier και παραβλεψη μιγαδικού μέρους
    idft = np.fft.ifftshift(idft_shift)
    filtered_image = np.real(np.abs(np.fft.ifft2(idft)))

    # Εμφάνιση των φιλτραρισμένων εικόνων
    plt.subplot(1, len(orders), i + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Φιλτραρισμένη εικόνα {i + 1}.\n Φιλτρο Butterworth {order}ης ταξης')

plt.show()