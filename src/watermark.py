import cv2
import numpy as np

# Load the image
image = cv2.imread('data/watermark.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding to create a binary image
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# Invert the binary image
binary = cv2.bitwise_not(binary)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for inpainting
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, contours, -1, 255, -1)

# Inpaint the image
result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Save the result
cv2.imwrite('data/without_watermark.jpg', result)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
