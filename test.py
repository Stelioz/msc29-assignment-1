import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_dct(F_dft):
    M, N = F_dft.shape

    # Initialize DCT coefficients array
    F_dct = np.zeros((M, N), dtype=np.complex128)

    # Calculate DCT coefficients using the provided formula
    for k in range(M):
        for l in range(N):
            # Compute scaling factors
            W_k = np.exp(-1j * 2 * np.pi * k / (2 * M))
            W_l = np.exp(-1j * 2 * np.pi * l / (2 * N))

            # Apply the formula to calculate DCT coefficients
            F_dct[k, l] = W_k**(k / 2) * W_l**(l / 2) * F_dft[k, l]

    return F_dct

def calculate_idct(F_dct):
    M, N = F_dct.shape

    # Initialize IDCT coefficients array
    F_idct = np.zeros((M, N), dtype=np.complex128)

    # Calculate IDCT coefficients using the inverse formula
    for k in range(M):
        for l in range(N):
            # Compute scaling factors
            W_k = np.exp(1j * 2 * np.pi * k / (2 * M))
            W_l = np.exp(1j * 2 * np.pi * l / (2 * N))

            # Apply the inverse formula to calculate IDCT coefficients
            F_idct[k, l] = (W_k**(k / 2) * W_l**(l / 2)) / (M * N) * F_dct[k, l]

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
