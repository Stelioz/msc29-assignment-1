import cv2
import numpy as np
from scipy.fft import idct
import matplotlib.pyplot as plt

# Συνάρτηση για τον υπολογισμός του 2D DFT
def compute_2d_dct(image):
    M, N = image.shape
    dct_result = np.zeros_like(image, dtype=float)

    # Προσπέλαση των στοιχείων συχνότητας (k, l)
    for k in range(M):
        for l in range(N):
            sum_value = 0.0
            
            # Προσπέλαση των χωρικών στοιχείων (m, l)
            for m in range(M):
                for n in range(N):
                    # Υπολογισμός των όρων και του αθροίσματος
                    cos_term_m = np.cos((2 * m + 1) * k * np.pi / (2 * M))
                    cos_term_n = np.cos((2 * n + 1) * l * np.pi / (2 * N))
                    sum_value += image[m, n] * cos_term_m * cos_term_n

            # Υπολογισμός των κανονικοποιημένων παραγόντων
            alpha_k = np.sqrt(2 / M) if k == 0 else np.sqrt(2 / M)
            alpha_l = np.sqrt(2 / N) if l == 0 else np.sqrt(2 / N)

            dct_result[k, l] = alpha_k * alpha_l * sum_value

    return dct_result


# Συνάρτηση για τον υπολογισμός του 2D IDFT
def compute_2d_idct(dct_result):
    M, N = dct_result.shape
    idct_image = np.zeros_like(dct_result, dtype=float)

    # Προσπέλαση των χωρικών στοιχείων (m, l)
    for m in range(M):
        for n in range(N):
            sum_value = 0.0
            
            # Προσπέλαση των στοιχείων συχνότητας (k, l)
            for k in range(M):
                for l in range(N):
                    # Υπολογισμός των όρων, των κανονικοποιημένων παραγόντων και του αθροίσματος
                    cos_term_k = np.cos((2 * m + 1) * k * np.pi / (2 * M))
                    cos_term_l = np.cos((2 * n + 1) * l * np.pi / (2 * N))
                    alpha_k = np.sqrt(2 / M) if k == 0 else np.sqrt(2 / M)
                    alpha_l = np.sqrt(2 / N) if l == 0 else np.sqrt(2 / N)
                    sum_value += alpha_k * alpha_l * dct_result[k, l] * cos_term_k * cos_term_l

            idct_image[m, n] = sum_value

    return idct_image


# Φόρτωση εικόνας από τον φάκελο image\ και μείωση των διαστασεων
image = cv2.imread('images\cameraman.bmp', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (64, 64)) 

# # Υπολογισμός του 2D DCT και του 2D IDCT
dct2D = compute_2d_dct(resized_image)
idct2D = compute_2d_idct(dct2D)

# Υπολογισμός του φάσματος πλάτους και του φάσματος φάσης
magnitude_spectrum = np.abs(dct2D)
phase_spectrum = np.angle(dct2D)

# Εμφάνιση της αρχικής εικόνας, του φάσματος πλάτους και φάσης αυτής
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(resized_image, cmap='gray')
plt.title("Αρχική Εικόνα (64x64)")

plt.subplot(1, 3, 2)
plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
plt.title("Φάσμα Πλάτους")

plt.subplot(1, 3, 3)
plt.imshow(phase_spectrum, cmap='gray')
plt.title("Φάσμα Φάσης")

plt.show()

# Ποσοστά των συντελεστών DCT (20%, 40%, 60%, 80%)
percentages = [0.2, 0.4, 0.6, 0.8]

# Δημιουργία του καμβά του σχήματος
plt.figure(figsize=(20, 20))

plt.subplot(1, len(percentages) + 1, 1)
plt.imshow(resized_image, cmap='gray')
plt.title("Αρχική Εικόνα (64x64)")

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
    mse = np.mean((resized_image - reconstructed_image) ** 2)
    print(f"{percentage * 100:.0f}% MSE: {mse:.2f}")

    # Εμφάνιση των ανακατασκευασμένων εικόνων
    plt.subplot(1, len(percentages) + 1, i + 1)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Εικόνα με {percentage * 100:.0f}% Συντελεστές\nMSE: {mse:.2f}")

plt.show()