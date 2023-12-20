import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('images\cameraman.bmp')

# Function to perform uniform scalar average-stepping quantization
def quantize(image, levels):
    quantized_image = np.zeros_like(image)
    for i in range(levels):
        lower_bound = i * (256 // levels)
        upper_bound = (i + 1) * (256 // levels)
        quantized_image[(image >= lower_bound) & (image < upper_bound)] = (lower_bound + upper_bound) // 2
    return quantized_image

# Function to calculate mean squared quantization error
def calculate_mse(original, quantized):
    return np.mean((original - quantized) ** 2)

# Quantization levels
quantization_levels = [7, 11, 15, 19]

# Plot the transformation function for each quantizer
plt.figure(figsize=(15, 5))
for levels in quantization_levels:
    transformation_function_x = np.arange(0, 256, 1)
    transformation_function_y = np.floor_divide(transformation_function_x * levels, 256) * (256 // levels)
    plt.plot(transformation_function_x, transformation_function_y, label=f'{levels} Levels')

plt.title('Transformation Functions of Quantizers')
plt.xlabel('Input Intensity')
plt.ylabel('Quantized Intensity')
plt.legend()
plt.show()

# Display the original image and quantized images
plt.figure(figsize=(20, 5))

# Plot the original image
plt.subplot(1, len(quantization_levels) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# Plot the quantized images
for i, levels in enumerate(quantization_levels, 1):
    quantized_image = quantize(image, levels)
    mse = calculate_mse(image, quantized_image)

    plt.subplot(1, len(quantization_levels) + 1, i + 1)
    plt.imshow(quantized_image, cmap='gray')
    plt.title(f'{levels} Levels\nMSE: {mse:.2f}')

plt.show()
