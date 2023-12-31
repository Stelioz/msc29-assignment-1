import cv2
import numpy as np
import matplotlib.pyplot as plt

# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images\cameraman.bmp', cv2.IMREAD_GRAYSCALE)

# Υπολογισμός του 2D μετασχηματισμού Fourier (DFT)
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Υπολογισμός του φάσματος πλάτους και του φάσματος φάσης
magnitude_spectrum = np.abs(dft_shift)
phase_spectrum = np.angle(dft_shift)

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

# Ποσοστά των συντελεστών DFT (20%, 40%, 60%, 80%)
percentages = [0.2, 0.4, 0.6, 0.8]
mse_values = []

# Δημιουργία του καμβά του σχήματος
plt.figure(figsize=(20, 20))

plt.subplot(1, len(percentages) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for i, percentage in enumerate(percentages, 1):
    # Δημιουργία μάσκας για την αποκοπή των υψηλών συντελεστών
    rows, cols = image.shape
    mask_rows = int(rows * percentage)
    mask_cols = int(cols * percentage)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[rows // 2 - mask_rows // 2: rows // 2 + mask_rows // 2,
         cols // 2 - mask_cols // 2: cols // 2 + mask_cols // 2] = 1

    # Εφαρμογή της μάσκας
    masked_dft_shift = dft_shift * mask

    # Χρήση του αντίστροφου μετασχηματισμού Fourier
    inverse_shifted = np.fft.ifftshift(masked_dft_shift)
    inverse_result = np.fft.ifft2(inverse_shifted)
    reconstructed_image = np.abs(inverse_result)

    # Υπολογισμός του μέσου τετραγωνικού σφάλματος (MSE)
    mse = np.mean((image - reconstructed_image) ** 2)
    mse_values.append(mse)

    # Εμφάνιση των ανακατασκευασμένων εικόνων, του φάσματος πλάτους και φάσης
    plt.subplot(1, len(percentages) + 1, i + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Εικόνα με {percentage * 100:.0f}% Συντελεστές')

plt.show()