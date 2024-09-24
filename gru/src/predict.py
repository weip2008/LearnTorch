import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
import numpy as np
import random
from config import Config, execution_time
from gruModel import load_data, GRUModel, TimeSeriesDataset
from gru import Logger

# Function to categorize the model output
def categorize_output(output):
    if 0.6 <= output <= 1.3:
        return 1.0
    elif -1.3 <= output <= -0.6:
        return -1.0
    else:
        return 0.0

class Predictor:
    config = Config('gru\src\config.ini')
    log = Logger('gru/log/gru.log')

    def __init__(self):
        self.load_data()
        self.buildLoader()
        self.defineModel()
        self.predict()
        Predictor.log.info("================================= Done")


    def load_data(self):
        testing_file_path = Predictor.config.testing_file_path
        Predictor.log.info(f"1. Load testing data from {testing_file_path}")
        self.testing_data, self.testing_signals = load_data(testing_file_path)
        Predictor.log.info(f"Data shape: {self.testing_data.shape}")
        Predictor.log.info(f"Targets shape: {self.testing_signals.shape}")

    def buildLoader(self):
        Predictor.log.info("2. Define dataset and dataloader")
        test_dataset = TimeSeriesDataset(self.testing_data, self.testing_signals)

        # Create DataLoader for batching
        batch_size = int(Predictor.config.batch_size)

        # Test dataloader with shuffling
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def defineModel(self):
        # Instantiate the model, define the loss function and the optimizer
        Predictor.log.info("3. Instantiate the model, define the loss function and the optimize")

        # Define hyperparameters
        input_size = int(Predictor.config.input_size)
        hidden_size = int(Predictor.config.hidden_size)
        output_size = int(Predictor.config.output_size)
        num_layers = int(Predictor.config.num_layers)

        # Instantiate the model
        Predictor.log.info(f"Number of layers: {num_layers}")
        self.model = GRUModel(input_size, hidden_size, output_size, num_layers)

        # Load the saved model state
        model_save_path = Predictor.config.model_save_path
        Predictor.log.info(f"4. Load trained model from {model_save_path}")
        checkpoint = torch.load(model_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self):
        sample_size = int(Predictor.config.sample_size)

        # Randomly select 10 rows from testing data
        random_indices = random.sample(range(len(self.testing_data)), sample_size)
        random_datas = self.testing_data[random_indices]
        random_targets = self.testing_signals[random_indices]

        # Print the output for each selected row
        Predictor.log.info("5. Randomly selected 10 rows and their corresponding outputs:")
        for i in range(sample_size):
            test_data = random_datas[i]
            test_target = random_targets[i].item()  # Get the actual target value
            
            # Call get_model_output to get the predicted output
            test_output = self.get_model_output(test_data)
            
            # Call categorize_output to categorize the predicted output
            categorized_output = categorize_output(test_output)
            
            # Print the test output, categorized output, and test target
            Predictor.log.info(f"Test Output: {test_output:7.4f} => Categorized Output: {categorized_output:4.1f}, \tTarget: {test_target:2.0f}")

    # Function to get the model output for a single input row
    def get_model_output(self,single_input):
        single_input_tensor = torch.tensor(single_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need for gradients during testing
            test_output = self.model(single_input_tensor)
        return test_output.item()  # Return the single output as a scalar

if __name__ == "__main__":
    Predictor()

