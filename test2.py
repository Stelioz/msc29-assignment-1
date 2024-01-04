import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal import butter, filtfilt

# A. Load the image and add Gaussian noise
image_path = "images/lenna.bmp"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
noisy_image = image.astype(np.float32) + np.random.normal(0, 10, image.shape)

# Εμφάνιση εικόνας πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Θορυβοποιημένη Εικόνα')

plt.show()

# B. Apply low-pass Butterworth filters in the frequency domain
def butterworth_filter(shape, order, cutoff_frequency):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    u = np.arange(0, rows)
    v = np.arange(0, cols)
    u, v = np.meshgrid(u, v)
    distance = np.sqrt((u - center_row) ** 2 + (v - center_col) ** 2)
    filter = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))
    return filter

orders = [3, 5, 7]
cutoff_frequency = 0.1
filtered_images = []

for order in orders:
    # Design the filter
    b, a = butter(order, cutoff_frequency, btype='low', analog=False, output='ba')
    
    # Apply the filter in the frequency domain
    # fft_noisy_image = fft2(noisy_image)
    # fft_filtered_image = fft_noisy_image * butterworth_filter(image.shape, order, cutoff_frequency)
    
    # Inverse Fourier transform
    # filtered_image = np.real(ifft2(fft_filtered_image))
    
    # Alternatively, you can apply the filter directly in the spatial domain
    filtered_image = filtfilt(b, a, noisy_image)

    filtered_images.append(filtered_image)

# C. Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, len(orders) + 1, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

for i, order in enumerate(orders):
    plt.subplot(1, len(orders) + 1, i + 2)
    plt.title(f"Filtered Image (Order {order})")
    plt.imshow(filtered_images[i], cmap='gray')

plt.show()