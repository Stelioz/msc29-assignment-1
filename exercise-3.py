import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

# Φόρτωση εικόνας από τον φάκελο image\ και μετατροπή σε float32
image = cv2.imread('images\cameraman.bmp', cv2.IMREAD_GRAYSCALE)
image_32 = image.astype(np.float32)

# Υπολογισμός του 2D DCT και του 2D IDCT
dct2D = dct(dct(image_32.T, norm='ortho').T, norm='ortho')
idct2D = idct(idct(dct2D.T, norm='ortho').T, norm='ortho')

# Υπολογισμός του φάσματος πλάτους και του φάσματος φάσης
magnitude_spectrum = np.abs(dct2D)
phase_spectrum = np.angle(dct2D)

# Εμφάνιση της αρχικής εικόνας, του φάσματος πλάτους και φάσης αυτής
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

plt.subplot(1, 3, 2)
plt.imshow(np.log(magnitude_spectrum), cmap='gray')
plt.title('Φάσμα Πλάτους')

plt.subplot(1, 3, 3)
plt.imshow(phase_spectrum, cmap='gray')
plt.title('Φάσμα Φάσης')

plt.show()

# Ποσοστά των συντελεστών DCT (20%, 40%, 60%, 80%)
percentages = [0.2, 0.4, 0.6, 0.8]

# Δημιουργία του καμβά του σχήματος
plt.figure(figsize=(20, 20))

plt.subplot(1, len(percentages) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

for i, percentage in enumerate(percentages, 1):
    # Δημιουργία μάσκας για την αποκοπή των υψηλών συντελεστών
    mask_rows = int(dct2D.shape[0] * percentage)
    mask_cols = int(dct2D.shape[1] * percentage)
    mask = np.zeros_like(dct2D)
    mask[:mask_rows, :mask_cols] = 1

    # Εφαρμογή της μάσκας
    masked_coefficients = dct2D * mask

    # Χρήση του 2D IDCT
    reconstructed_image = idct(idct(masked_coefficients.T, norm='ortho').T, norm='ortho')

    # Υπολογισμός του μέσου τετραγωνικού σφάλματος (MSE)
    mse = np.mean((image_32 - reconstructed_image) ** 2)
    print(f'{percentage * 100:.0f}% MSE: {mse:.2f}')

    # Εμφάνιση των ανακατασκευασμένων εικόνων
    plt.subplot(1, len(percentages) + 1, i + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Εικόνα με {percentage * 100:.0f}% Συντελεστές\nMSE: {mse:.2f}')

plt.show()
