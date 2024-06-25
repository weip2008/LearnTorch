import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder_fc = nn.Linear(32 * 2 * 150, 256)  # Adjust this based on the flattened size after conv layers
        
        # Latent space
        self.latent_fc = nn.Linear(256, 2)  # Final layer to produce the 2-dimensional output
        
        # Decoder
        self.decoder_fc = nn.Linear(2, 256)
        self.decoder_fc2 = nn.Linear(256, 32 * 2 * 150)  # Adjust based on the shape before flattening
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x_shape = x.size()  # Get the shape of the tensor
        x = x.view(x_shape[0], -1)  # Flatten the tensor to (batch_size, num_channels * height * width)
        x = F.relu(self.encoder_fc(x))
        
        # Latent space
        latent = self.latent_fc(x)  # Produce the 2-dimensional output
        
        # Decoder
        x = F.relu(self.decoder_fc(latent))
        x = F.relu(self.decoder_fc2(x))
        x = x.view(x_shape[0], 32, 2, 150)  # Reshape back to the expected shape for ConvTranspose2d
        x = F.relu(self.decoder_conv2(x))
        x = torch.sigmoid(self.decoder_conv1(x))

        return latent, x  # Return both the latent and reconstructed output

# Example usage
x = torch.randn(500, 1, 1, 1200)  # Input shape: (500, 1, 1, 1200) to match the Conv2d requirements
model = AutoEncoder()
latent, output = model(x)
print(latent.shape)  # Output shape: (500, 2)
print(output.shape)  # Reconstructed output shape: (500, 1, 1, 1200)
