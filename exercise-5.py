import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from scipy.signal import butter, lfilter


# Συνάρτηση για την προθήκη Γκαουσιανού θορύβου
def noise_adder(image, mean, variance):
    
    noise = np.random.normal(mean, np.sqrt(variance), image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return noisy_image, noise


# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images\lenna.bmp', cv2.IMREAD_GRAYSCALE)

# Προσθήκη Γκαουσιανού θορύβου
mean = 0
variance = 0.01
noisy_image, noise = noise_adder(image, mean, variance)

# Εμφάνιση εικόνας πριν και μετά την προσθήκη θορύβου
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

plt.subplot(1, 3, 2)
plt.imshow(noise, cmap='gray')
plt.title('Θόρυβος')

plt.subplot(1, 3, 3)
plt.imshow(noisy_image, cmap='gray')
plt.title('Θορυβοποιημένη Εικόνα')

plt.show()

# # Ensure pixel values are within the valid range
# noisy_image = np.clip(noisy_image, 0, 255)

# # Function to create a low-pass Butterworth filter
# def butterworth_lowpass_filter(shape, cutoff, order):
#     rows, cols = shape
#     center = (rows // 2, cols // 2)
#     x = np.arange(cols) - center[1]
#     y = np.arange(rows) - center[0]
#     xx, yy = np.meshgrid(x, y)
#     r_squared = xx**2 + yy**2
#     filter = 1 / (1 + (r_squared / cutoff**2)**order)
#     return filter

# # Apply low-pass Butterworth filters of 3rd, 5th, and 7th order
# filter_orders = [3, 5, 7]
# for i, order in enumerate(filter_orders):
#     # Create Butterworth filter
#     filter_cutoff = 50  # Adjust cutoff frequency as needed
#     butterworth_filter = butterworth_lowpass_filter(image.shape, filter_cutoff, order)

#     # Apply Fourier transform to the noisy image and the filter
#     noisy_image_fft = fftshift(fftn(noisy_image))
#     filter_fft = fftshift(fftn(butterworth_filter))

#     # Apply the filter in the frequency domain
#     filtered_image_fft = noisy_image_fft * filter_fft

#     # Inverse Fourier transform to get the filtered image
#     filtered_image = np.abs(ifftn(ifftshift(filtered_image_fft)))

#     # Display the filtered image
#     plt.subplot(2, 4, i + 5)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f"Filtered Image (Order {order})")
#     plt.axis('off')

# # Show the plots
# plt.tight_layout()
# plt.show()
