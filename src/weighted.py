import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduction='none')
        weighted_loss = loss * self.weights
        return weighted_loss.mean()

# Generate weights
weights = np.linspace(0.1, 1.0, len(trainloader.dataset))
weights = torch.tensor(weights, dtype=torch.float32).to(device)

# Use the custom loss function
loss_fn = WeightedCrossEntropyLoss(weights)

# Train the model
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done with training!")

torch.save(model.state_dict(), "handwriting_model_with_weighting.pth")
print("Saved PyTorch Model State to handwriting_model_with_weighting.pth")
