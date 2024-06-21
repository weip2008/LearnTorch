"""
pip install opencv-python
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "data/lady.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply different convolution filters
kernel_identity = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
kernel_edge_detection = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

filtered_identity = cv2.filter2D(gray_image, -1, kernel_identity)
filtered_edge_detection = cv2.filter2D(gray_image, -1, kernel_edge_detection)
filtered_sharpen = cv2.filter2D(gray_image, -1, kernel_sharpen)

# Display the original and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(filtered_identity, cmap='gray')
plt.title('Identity Filter')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(filtered_edge_detection, cmap='gray')
plt.title('Edge Detection Filter')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(filtered_sharpen, cmap='gray')
plt.title('Sharpen Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
