import cv2
import numpy as np
import matplotlib.pyplot as plt

# Συνάρτηση υπολογισμού 2D DCT
def dct2D(image):
    M, N = image.shape
    # Αρχικοποίηση του πίνακα
    F_dct = np.zeros((M, N), dtype=np.complex128)
    for k in range(M):
        for l in range(N):
            # Επέκταση της εικόνας
            W_k = np.exp(-1j * 2 * np.pi * k / (2 * M))
            W_l = np.exp(-1j * 2 * np.pi * l / (2 * N))

            # Χρήση του μαθηματικού τύπου
            F_dct[k, l] = W_k**(k / 2) * W_l**(l / 2) * image[k, l]

    return F_dct

# Συνάρτηση υπολογισμού 2D IDCT
def idct2D(image):
    M, N = image.shape
    # Αρχικοποίηση του πίνακα
    F_idct = np.zeros((M, N), dtype=np.complex128)
    for k in range(M):
        for l in range(N):
            # Επέκταση της εικόνας
            W_k = np.exp(1j * 2 * np.pi * k / (2 * M))
            W_l = np.exp(1j * 2 * np.pi * l / (2 * N))

            # Χρήση του μαθηματικού τύπου
            F_idct[k, l] = (W_k**(k / 2) * W_l**(l / 2)) / (M * N) * image[k, l]

    return F_idct

# Load the grayscale image
image = cv2.imread('images/cameraman.bmp', cv2.IMREAD_GRAYSCALE)

# Convert the image to float32
image_float32 = image.astype(np.float32)

# Calculate the 2D DFT of the image using numpy's FFT
F_dft = np.fft.fft2(image_float32)

# Calculate the 2D DCT using the provided code
F_dct = calculate_dct(F_dft)

# Calculate the inverse DCT to reconstruct the image
F_idct = calculate_idct(F_dct)
reconstructed_image = np.real(np.fft.ifft2(F_idct))

# Display the original and reconstructed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')

plt.show()
