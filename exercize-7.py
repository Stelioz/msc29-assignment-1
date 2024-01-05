import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_first_derivative_methods(image):
    # Convert image to 8-bit unsigned integer
    image_uint8 = cv2.convertScaleAbs(image)

    # Sobel
    sobel_x = cv2.Sobel(image_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert edges to 8-bit for Otsu's method
    sobel_edges_uint8 = cv2.convertScaleAbs(sobel_edges)
    _, sobel_thresholded = cv2.threshold(sobel_edges_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Roberts
    roberts_x = cv2.filter2D(image_uint8, cv2.CV_64F, np.array([[1, 0], [0, -1]]))
    roberts_y = cv2.filter2D(image_uint8, cv2.CV_64F, np.array([[0, 1], [-1, 0]]))
    roberts_edges = np.sqrt(roberts_x**2 + roberts_y**2)

    # Convert edges to 8-bit for Otsu's method
    roberts_edges_uint8 = cv2.convertScaleAbs(roberts_edges)
    _, roberts_thresholded = cv2.threshold(roberts_edges_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Prewitt
    prewitt_x = cv2.filter2D(image_uint8, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv2.filter2D(image_uint8, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    prewitt_edges = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Convert edges to 8-bit for Otsu's method
    prewitt_edges_uint8 = cv2.convertScaleAbs(prewitt_edges)
    _, prewitt_thresholded = cv2.threshold(prewitt_edges_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kirsch
    kirsch_edges = cv2.filter2D(image_uint8, cv2.CV_64F, np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]))

    # Convert edges to 8-bit for Otsu's method
    kirsch_edges_uint8 = cv2.convertScaleAbs(kirsch_edges)
    _, kirsch_thresholded = cv2.threshold(kirsch_edges_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return sobel_thresholded, roberts_thresholded, prewitt_thresholded, kirsch_thresholded

def apply_second_derivative_laplace(image, variance, threshold):
    # Apply LoG (Laplacian of Gaussian)
    blurred = cv2.GaussianBlur(image, (0, 0), variance)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    log_edges = np.zeros_like(laplacian, dtype=np.uint8)
    log_edges[laplacian > threshold] = 255

    return log_edges

def apply_canny(image):
    # Canny
    edges_canny = cv2.Canny(image, 50, 150)

    return edges_canny

def main():
    # Read the butterfly image
    butterfly_image = cv2.imread('images/butterfly_g.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply first derivative methods
    sobel, roberts, prewitt, kirsch = apply_first_derivative_methods(butterfly_image)

    # Display results for first derivative methods
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1), plt.imshow(butterfly_image, cmap='gray'), plt.title('Original')
    plt.subplot(2, 3, 2), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
    plt.subplot(2, 3, 3), plt.imshow(roberts, cmap='gray'), plt.title('Roberts')
    plt.subplot(2, 3, 4), plt.imshow(prewitt, cmap='gray'), plt.title('Prewitt')
    plt.subplot(2, 3, 5), plt.imshow(kirsch, cmap='gray'), plt.title('Kirsch')

    plt.show()

    # Experimentally determine optimal values for LoG
    variance_optimal = 1.5  # Adjust as needed
    threshold_optimal = 20  # Adjust as needed

    # Apply LoG
    log_edges = apply_second_derivative_laplace(butterfly_image, variance_optimal, threshold_optimal)

    # Display results for LoG
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1), plt.imshow(butterfly_image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(log_edges, cmap='gray'), plt.title('LoG')

    plt.show()

    # Apply Canny
    edges_canny = apply_canny(butterfly_image)

    # Display results for Canny
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1), plt.imshow(butterfly_image, cmap='gray'), plt.title('Original')
    plt.subplot(1, 2, 2), plt.imshow(edges_canny, cmap='gray'), plt.title('Canny')

    plt.show()

if __name__ == "__main__":
    main()
