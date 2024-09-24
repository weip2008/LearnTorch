import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from config import Config, execution_time
from gru import Logger


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

def load_data(training_file_path):
    data = []
    signals = []

    with open(training_file_path, 'r') as file:
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
    log = Logger('gru/log/gru.log')
    def __init__(self):
        self.loadData()
        self.buildDataLoader()
        self.defineModel()
        self.train()
        self.save()

    def loadData(self):
        training_file_path = ModelGenerator.config.training_file_path
        testing_file_path = ModelGenerator.config.testing_file_path

        ModelGenerator.log.info(f"1.1 Load training data from {training_file_path}")
        self.training_data, self.training_signals = load_data(training_file_path)
        ModelGenerator.log.info(f"Data shape: {self.training_data.shape}")
        ModelGenerator.log.info(f"Targets shape: {self.training_signals.shape}")
        ModelGenerator.log.info(f"1.2 Load testing data from {testing_file_path}")
        self.testing_data, self.testing_signals = load_data(testing_file_path)

    def buildDataLoader(self):
        # Instantiate the dataset
        ModelGenerator.log.info("2. Define dataset and dataloader")
        train_dataset = TimeSeriesDataset(self.training_data, self.training_signals)
        val_dataset = TimeSeriesDataset(self.testing_data, self.testing_signals)
        # Create DataLoader for batching
        batch_size = int(config.batch_size)
        # Training dataloader with shuffling
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Validation dataloader with shuffling
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    def defineModel(self):
        # Instantiate the model, define the loss function and the optimizer
        ModelGenerator.log.info("3. Instantiate the model, define the loss function and the optimize")

        # Define hyperparameters
        input_size = int(config.input_size)
        hidden_size = int(config.hidden_size)
        output_size = int(config.output_size)
        num_layers = int(config.num_layers)
        learning_rate = float(config.learning_rate)

        # Instantiate the model
        ModelGenerator.log.info(f"Number of layers: {num_layers}")
        self.model = GRUModel(input_size, hidden_size, output_size, num_layers)

    @execution_time
    def train(self):
        learning_rate = float(ModelGenerator.config.learning_rate)
        # Loss function: Binary Cross Entropy Loss
        #criterion = nn.BCEWithLogitsLoss()  # Use with sigmoid for binary classification
        criterion = nn.MSELoss()

        # Optimizer: Adam
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs

        # Training loop
        log.info("4. Start training loop")

        # Hyperparameters
        num_epochs = int(config.num_epochs)

        # List to store losses
        self.train_losses = []
        self.val_losses = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            epoch_loss = 0
            for inputs, targets in self.train_dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            self.train_losses.append(avg_epoch_loss)
            #print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time  # Duration in seconds
            ModelGenerator.log.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}, Duration: {epoch_duration:.2f} seconds')
            
            
            # Validation loss
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_inputs, val_targets in self.val_dataloader:
                    val_outputs = self.model(val_inputs)
                    val_loss += criterion(val_outputs, val_targets).item()
            avg_val_loss = val_loss / len(self.val_dataloader)
            self.val_losses.append(avg_val_loss)
            ModelGenerator.log.info(f' Validation Loss: {avg_val_loss:.6f}')
      
    def save(self):
        save_path = ModelGenerator.config.save_path
        # Save the model, optimizer state, and losses
        log.info(f"5. Save the model, optimizer state, and losses to {save_path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.val_losses
        }, save_path)


        log.info(f"Training model saved to {save_path}")


if __name__ == "__main__":
    ModelGenerator()
