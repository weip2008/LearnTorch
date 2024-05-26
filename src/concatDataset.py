from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

# Load the MNIST dataset
mnist_train = datasets.MNIST('data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                             ]))
mnist_test = datasets.MNIST('data', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))

# Load your handwritten digit dataset (assuming it's stored as a PyTorch Dataset)
# Replace this with your actual dataset loading code
your_dataset = ...

# Combine the datasets
combined_train_dataset = ConcatDataset([mnist_train, your_dataset])
combined_test_dataset = ConcatDataset([mnist_test, your_dataset])

# Now you can use combined_train_dataset and combined_test_dataset for training and testing
