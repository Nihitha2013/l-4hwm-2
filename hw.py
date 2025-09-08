import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image):
    """Utility function to display an image"""
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def interactive_edge_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image("Original Grayscale Image", gray_image)

    history = [gray_image.copy()]  # Initialize history

    print("\nSelect option:")
    print("1: Sobel Edge Detection")
    print("2: Canny Edge Detection")
    print("3: Laplacian Edge Detection")
    print("4: Gaussian Smoothing")
    print("5: Median Filtering")
    print("6: Reset")
    print("7: Undo")
    print("8: Exit")

    while True:
        choice = input("\nEnter your choice: ").strip()
        current_image = history[-1]

        if choice == "1":
            sobelx = cv2.Sobel(current_image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(current_image, cv2.CV_64F, 0, 1, ksize=5)
            combined_sobel = cv2.bitwise_or(
                np.abs(sobelx).astype(np.uint8), np.abs(sobely).astype(np.uint8)
            )
            history.append(combined_sobel)
            display_image("Sobel Edge Detection", combined_sobel)

        elif choice == "2":
            try:
                low_threshold = int(input("Enter the low threshold value: "))
                upper_threshold = int(input("Enter the upper threshold value: "))
                if low_threshold < 0 or upper_threshold < 0 or low_threshold >= upper_threshold:
                    print("Invalid threshold values.")
                    continue
                edges = cv2.Canny(current_image, low_threshold, upper_threshold)
                history.append(edges)
                display_image("Canny Edge Detection", edges)
            except ValueError:
                print("Please enter valid integers for thresholds.")

        elif choice == "3":
            laplacian = cv2.Laplacian(current_image, cv2.CV_64F)
            abs_laplacian = np.abs(laplacian).astype(np.uint8)
            history.append(abs_laplacian)
            display_image("Laplacian Edge Detection", abs_laplacian)

        elif choice == "4":
            try:
                kernel_size = int(input("Enter an odd kernel size (e.g., 3, 5, 7): "))
                if kernel_size % 2 == 0 or kernel_size < 1:
                    print("Kernel size must be a positive odd number.")
                    continue
                blurred_image = cv2.GaussianBlur(current_image, (kernel_size, kernel_size), 0)
                history.append(blurred_image)
                display_image("Gaussian Smoothing", blurred_image)
            except ValueError:
                print("Invalid input. Please enter an integer.")

        elif choice == "5":
            try:
                kernel_size = int(input("Enter an odd kernel size (e.g., 3, 5, 7): "))
                if kernel_size % 2 == 0 or kernel_size < 1:
                    print("Kernel size must be a positive odd number.")
                    continue
                median_image = cv2.medianBlur(current_image, kernel_size)
                history.append(median_image)
                display_image("Median Filtering", median_image)
            except ValueError:
                print("Invalid input. Please enter an integer.")

        elif choice == "6":
            history = [gray_image.copy()]
            display_image("Reset to Grayscale", gray_image)

        elif choice == "7":
            if len(history) > 1:
                history.pop()
                display_image("Undo Result", history[-1])
            else:
                print("Nothing to undo.")

        elif choice == "8":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")

# Example usage:
interactive_edge_detection("landscape.jpg")
