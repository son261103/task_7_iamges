import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess(image_path):
    """
    Load and preprocess the image
    """
    # Read image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray, blurred


def apply_sobel(img):
    """
    Apply Sobel edge detection
    """
    # Calculate gradients in x and y directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Normalize to 0-255
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    return magnitude


def apply_prewitt(img):
    """
    Apply Prewitt edge detection
    """
    # Prewitt kernels
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    # Apply kernels
    prewittx = cv2.filter2D(img, -1, kernelx)
    prewitty = cv2.filter2D(img, -1, kernely)

    # Calculate magnitude
    magnitude = np.sqrt(prewittx ** 2 + prewitty ** 2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    return magnitude


def apply_roberts(img):
    """
    Apply Roberts edge detection
    """
    # Roberts kernels
    roberts_cross_v = np.array([[1, 0],
                                [0, -1]])
    roberts_cross_h = np.array([[0, 1],
                                [-1, 0]])

    # Apply kernels
    vertical = cv2.filter2D(img, -1, roberts_cross_v)
    horizontal = cv2.filter2D(img, -1, roberts_cross_h)

    # Calculate magnitude
    magnitude = np.sqrt(vertical ** 2 + horizontal ** 2)
    magnitude = np.uint8(magnitude * 255 / np.max(magnitude))

    return magnitude


def apply_canny(img, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection
    """
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges


def display_results(images, titles):
    """
    Display multiple images with their titles
    """
    n = len(images)
    plt.figure(figsize=(20, 4))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_results(images, titles, output_dir):
    """
    Save processed images
    """
    for img, title in zip(images, titles):
        output_path = f"{output_dir}/{title.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(output_path, img)