import cv2
import numpy as np
import matplotlib.pyplot as plt

# Φόρτωση εικόνας από τον φάκελο image\
image = cv2.imread('images\cameraman.bmp')

# Συνάρτηση για τη δημιουργία του ομοιόμορφου βαθμωτού κβαντιστή μέσου πατήματος
def quantize(image, levels):
    quantized_image = np.zeros_like(image)
    for i in range(levels):
        lower_bound = i * (256 // levels)
        upper_bound = (i + 1) * (256 // levels)
        quantized_image[(image >= lower_bound) & (image < upper_bound)] = (lower_bound + upper_bound) // 2
    return quantized_image

# Συνάρτηση για τον υπολογισμό του μέσου τετραγωνικού σφάλματος κβάντισης
def calculate_mse(original, quantized):
    return np.mean((original - quantized) ** 2)

# Στάθμες κβάντισης
quantization_levels = [7, 11, 15, 19]

# Σχεδιασμός συνάρτησης μετασχηματισμού για κάθε κβαντιστή
plt.figure(figsize=(15, 5))
for levels in quantization_levels:
    transformation_function_x = np.arange(0, 256, 1)
    transformation_function_y = np.floor_divide(transformation_function_x * levels, 256) * (256 // levels)
    plt.plot(transformation_function_x, transformation_function_y, label=f'{levels} Στάθμες')

plt.title('Συναρτήσεις Μετασχηματισμού')
plt.xlabel('Ένταση Εισόδου')
plt.ylabel('Κβαντισμένη Ένταση')
plt.legend()
plt.show()

# Εμφάνιση της αρχικής και των κβαντισμένων εικόνων
plt.figure(figsize=(20, 5))

# Σχεδιασμός της αρχικής εικόνας
plt.subplot(1, len(quantization_levels) + 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Αρχική Εικόνα')

# Σχεδιασμός των κβαντισμένων εικόνων
for i, levels in enumerate(quantization_levels, 1):
    quantized_image = quantize(image, levels)
    mse = calculate_mse(image, quantized_image)

    plt.subplot(1, len(quantization_levels) + 1, i + 1)
    plt.imshow(quantized_image, cmap='gray')
    plt.title(f'{levels} Στάθμες\nMSE: {mse:.2f}')

plt.show()
