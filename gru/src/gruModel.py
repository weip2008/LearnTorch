import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from config import Config, execution_time
from logger import Logger
from generateDataset import StockDataset


# Custom dataset class for loading signals and data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.signals[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8*60, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def load_data(file_path):
    data = []
    signals = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            signals_part, data_part = line.strip().split(',[')
            
            signal = int(signals_part.strip())
            signals.append(signal)
            
            # Add the beginning bracket to the data part and opening bracket to the target part
            data_part = '[' + data_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            
            # Append to the respective lists
            data.append(data_row)
            #targets.append(target_row[0])  # Ensure target_row is a 1D array
    
    # Convert lists to numpy arrays
    data_np = np.array(data)
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    #signals_np = np.array(signals)
    
    return data_np, signals_np

class ModelGenerator:
    config = Config('gru/src/config.ini')
    log = Logger('gru/log/gru.log',logger_name='model')

    def __init__(self):
        self.loadData()
        self.defineModel("linear")
        self.train_test()
        self.save()
        ModelGenerator.log.info("ModelGenerator ================================ Done")

    def defineModel(self, model_type="gru"):
        ModelGenerator.log.info(f"2. Instantiate the {model_type} model")
        
        model_dict = {
            "gru": self.buildGRU,
            "linear": lambda: NeuralNetwork().to('cpu')
        }

        # Select and instantiate the model
        self.model = model_dict.get(model_type, self.buildDefaultModel)()

    def buildGRU(self):
        input_size = int(ModelGenerator.config.input_size)
        hidden_size = int(ModelGenerator.config.hidden_size)
        output_size = int(ModelGenerator.config.output_size)
        num_layers = int(ModelGenerator.config.num_layers)

        ModelGenerator.log.info(f"Number of layers: {num_layers}")
        return GRUModel(input_size, hidden_size, output_size, num_layers)

    def buildDefaultModel(self):
        ModelGenerator.log.warning("Unknown model type, falling back to default (NeuralNetwork).")
        return NeuralNetwork().to('cpu')
    
    def loadData(self):
        training_file_path = ModelGenerator.config.training_file_path
        testing_file_path = ModelGenerator.config.testing_file_path
        batch_size = int(ModelGenerator.config.batch_size)
        ModelGenerator.log.info(f"1. Load dataset from {training_file_path}")

        training_dataset = torch.load(training_file_path)
        self.train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
        testing_dataset =  torch.load(testing_file_path)
        self.test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
        ModelGenerator.log.info(f'Training data size: {len(training_dataset)}, {training_dataset.get_shapes()}')
        ModelGenerator.log.info(f'Testing data size: {len(testing_dataset)}, {testing_dataset.get_shapes()}')


    def buildDataLoader(self):
        # Instantiate the dataset
        ModelGenerator.log.info("2. Define dataset and dataloader")
        train_dataset = TimeSeriesDataset(self.training_data, self.training_signals)
        val_dataset = TimeSeriesDataset(self.testing_data, self.testing_signals)
        # Create DataLoader for batching
        batch_size = int(ModelGenerator.config.batch_size)
        # Training dataloader with shuffling
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        # Validation dataloader with shuffling
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    @execution_time
    def train(self, criterion):
        self.model.train()
        size = len(self.train_dataloader.dataset)
        for batch, (inputs, targets) in enumerate(self.train_dataloader):
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                    
            if batch % 32 == 0:
                loss, current = loss.item(), (batch + 1) * len(inputs)
                ModelGenerator.log.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, loss_fn):
        device = "cpu"
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                y = y.squeeze().to(torch.int64)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        self.accuracy = str(round(100 * correct, 1))+'%'
        ModelGenerator.log.info(f"Test result: Accuracy: {self.accuracy}, Avg loss: {test_loss:>8f} \n")

    def train_test(self):
        learning_rate = float(ModelGenerator.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

        epochs = int(ModelGenerator.config.num_epochs)
        for t in range(epochs):
            ModelGenerator.log.info(f"Epoch {t+1}\n-------------------------------")
            self.train(loss_fn)
            self.test(loss_fn)
        
    def save(self):
        save_path = ModelGenerator.config.model_save_path + self.accuracy + ".pth"
        # Save the model, optimizer state, and losses
        ModelGenerator.log.info(f"5. Save the model, optimizer state, and losses to {save_path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'train_losses': self.train_losses,
            # 'test_losses': self.val_losses
        }, save_path)


        ModelGenerator.log.info(f"Training model saved to {save_path}")


if __name__ == "__main__":
    ModelGenerator()
