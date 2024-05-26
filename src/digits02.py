import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from digits01 import NeuralNetwork, test_data
import matplotlib.pyplot as plt
import numpy as np
import random 
from PIL import Image
from torchvision import datasets, transforms

# Define the function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image, dtype=np.float32)
    # image = 255 - image  # Invert colors (optional, depending on your training data)
    image = image / 255.0  # Normalize to [0, 1]
    image = image.reshape(1, 28,28)  # Flatten the image
    image = torch.tensor(image)
    return image

# Define the function to predict the digit
def predict_digit(image_path):
    x = preprocess_image(image_path)
    image = transforms.ToPILImage()(x)
    image.show()
    with torch.no_grad():
        output = model(x.unsqueeze(0))
        prediction = output[0].argmax(0)
    return prediction

# Example of how to use the predict_digit function
if __name__ == "__main__":
    model = NeuralNetwork()
    model.eval()
    model.load_state_dict(torch.load("handwritting_model_92.pth"))


    # Assuming the image is saved as "digit_2.png"
    digit_path = "digit_4.png"
    predicted_digit = predict_digit(digit_path)
    print(f"Predicted digit: {predicted_digit}")
