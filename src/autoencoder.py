import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder_fc = nn.Linear(32*2*50, 256)
        
        # Decoder
        self.decoder_fc = nn.Linear(256, 32*2*50)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x_shape = x.size()  # Get the shape of the tensor
        x = x.view(x_shape[0], -1)  # Flatten the tensor to (batch_size, num_channels * height * width)
        x = F.relu(self.encoder_fc(x))
        
        # Decoder
        x = F.relu(self.decoder_fc(x))
        x = x.view(x_shape)  # Reshape back to original shape
        x = F.relu(self.decoder_conv2(x))
        x = torch.sigmoid(self.decoder_conv1(x))

        return x

# Example usage
input_size = (500, 1, 6, 200)
output_size = (500, 3)
x = torch.randn(input_size)
model = AutoEncoder()
output = model(x)
print(output.shape)  # Output shape: (500, 1, 6, 200)
