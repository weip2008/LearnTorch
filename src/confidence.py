import torch
import torch.nn.functional as F
from digits01 import NeuralNetwork
from digits02 import preprocess_image

def predict_digit(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1).values
    return prediction.item(), confidence.item()

# Example of how to use the predict_digit function
if __name__ == "__main__":
    model = NeuralNetwork()
    model.load_state_dict(torch.load("handwritting_model.pth"))

    model.eval()

    digit_path = "digit_8.png"
    predicted_digit, confidence = predict_digit(digit_path)
    print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}")
