import cv2
import numpy as np

# Load the image
image = cv2.imread('data/watermark.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection using Canny
edges = cv2.Canny(gray, 50, 150)

# Dilate the edges to create a mask
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)

# Invert the mask
mask = cv2.bitwise_not(dilated_edges)

# Inpaint the image using the mask
result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

# Save the result
cv2.imwrite('data/without_watermark.jpg', result)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
