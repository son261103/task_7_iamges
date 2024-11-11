import os
from utils import *


def main():
    # Create output directory if it doesn't exist
    output_dir = "output_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process both images
    image_paths = ["image1.png", "image2.png"]

    for idx, image_path in enumerate(image_paths, 1):
        print(f"Processing image {idx}...")

        # Load and preprocess image
        original, gray, blurred = load_and_preprocess(image_path)

        # Apply different edge detection methods
        sobel_edges = apply_sobel(blurred)
        prewitt_edges = apply_prewitt(blurred)
        roberts_edges = apply_roberts(blurred)
        canny_edges = apply_canny(blurred)

        # Display results
        images = [gray, sobel_edges, prewitt_edges, roberts_edges, canny_edges]
        titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'Canny']

        display_results(images, titles)

        # Save results
        save_results(images, [f"image{idx}_{title}" for title in titles], output_dir)

        print(f"Results for image {idx} have been saved to {output_dir}")


if __name__ == "__main__":
    main()