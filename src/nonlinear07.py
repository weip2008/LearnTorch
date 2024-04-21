import torch
import torch.nn as nn
from nonlinear05 import Net

f = lambda x: x**3 + x**2/2 - 4*x - 2 # actual nonlinear function

model = Net()
model.load_state_dict(torch.load("relu_model.pth"))

x = torch.tensor([1.0], dtype=torch.float32)
pred = model(x)
actual = f(x)

# print("predicted", pred.item(), "; actual", actual.item())
print(f"Predicted value={pred.item():.1f}; Actual value={actual.item():.1f}")