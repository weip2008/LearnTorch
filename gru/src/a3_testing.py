import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np

from a2_training import load_data, GRUModel, TimeSeriesDataset
from config import Config, execution_time
from logger import Logger
from a4_predict import categorize_output

class Tester:
    config = Config('gru/src/config.ini')
    log = Logger('gru/log/gru.log',logger_name='test')

    def __init__(self):
        self.load_data()
        self.buildLoader()
        self.defineModel()
        self.test()
        Tester.log.info("================================ Done")

    def load_data(self):
        testing_file_path = Tester.config.testing_file_path

        # Example usage
        Tester.log.info(f"1. Load testing data from {testing_file_path}")
        self.testing_data, self.testing_signals = load_data(testing_file_path)

        Tester.log.info(f"Data shape: {self.testing_data.shape}")
        Tester.log.info(f"Targets shape: {self.testing_signals.shape}")

    def buildLoader(self):
        # Instantiate the dataset
        Tester.log.info("2. Define dataset and dataloader")
        test_dataset = TimeSeriesDataset(self.testing_data, self.testing_signals)

        # Create DataLoader for batching
        batch_size = int(Tester.config.batch_size)
        # Test dataloader with shuffling
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def defineModel(self):
        # Instantiate the model, define the loss function and the optimizer
        Tester.log.info("3. Instantiate the model, define the loss function and the optimize")

        # Define hyperparameters
        input_size = int(Tester.config.input_size)
        hidden_size = int(Tester.config.hidden_size)
        output_size = int(Tester.config.output_size)
        num_layers = int(Tester.config.num_layers)

        # Instantiate the model
        Tester.log.info(f"Number of layers: {num_layers}")
        self.model = GRUModel(input_size, hidden_size, output_size, num_layers)

        # Load the saved model state
        model_save_path = Tester.config.model_save_path
        Tester.log.info(f"4. Load trained model from {model_save_path}")
        checkpoint = torch.load(model_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set the model to evaluation mode

    @execution_time
    def test(self):
        # Loss function: Binary Cross Entropy Loss
        #criterion = nn.BCEWithLogitsLoss()  # Use with sigmoid for binary classification
        criterion = nn.MSELoss()

        # Training loop
        Tester.log.info("5. Start testing loop")

        # Evaluate the model on the testing data
        test_loss = 0
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for test_inputs, test_targets in self.test_dataloader:
                test_outputs = self.model(test_inputs)
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()

                all_targets.extend(test_targets.numpy())
                all_outputs.extend(test_outputs.numpy())

        avg_test_loss = test_loss / len(self.test_dataloader)
        Tester.log.info(f'Test Loss (MSE): {avg_test_loss:.8f}')
        # Mean Squared Error (MSE) measures the average squared difference between the predicted values 
        # and the actual values.
        # A lower MSE indicates that the model’s predictions are closer to the actual values. 
        # Test Loss (MSE): 0.01045113 suggests that, on average, the squared difference between the 
        # predicted and actual values is quite small.

        # Calculate additional metrics manually
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)

        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(all_targets - all_outputs))
        Tester.log.info(f'Mean Absolute Error (MAE): {mae:.8f}')
        # MAE measures the average absolute difference between the predicted values and the actual values.
        # It gives an idea of how much the predictions deviate from the actual values on average. 
        # Mean Absolute Error (MAE): 0.07155589 means on average, the model’s predictions are off by about 0.0716 
        # units from the actual values.

        # R-squared (R2)
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        Tester.log.info(f'R-squared (R2): {r2:.8f}')
        # R-squared is a statistical measure that represents the proportion of the variance for a 
        # dependent variable that’s explained by an independent variable or variables in a regression model.
        # R-squared (R2): 0.89939589  indicates that approximately 89.94% of the variance in the target variable
        # is explained by the model. This is a high value, suggesting that the model fits the data well.

        # MSE and MAE are both measures of prediction error, with lower values indicating better performance.
        # R2 is a measure of how well the model explains the variability of the target data, 
        #    with values closer to 1 indicating a better fit
            
        # Open a text file in write mode
        output_results_path = Tester.config.output_results_path
        with open(output_results_path, 'w') as file:
            # Loop through all targets and outputs
            for target, output in zip(all_targets, all_outputs):
                # Apply the logic to categorize the output as either 1 or 0
                categorized_output = categorize_output(output)

                # Prepare the output string for each pair
                output_string = f"Target{target} : Output[{output[0]:.4f}] -> Signal[{categorized_output}]\n"
                
                # Print to the screen
                #print(output_string.strip())  # .strip() to avoid extra newlines

                # Write the same output to the file
                file.write(output_string)
                
        Tester.log.info(f'Saved categorized signals to file : {output_results_path}')       


if __name__ == "__main__":
    Tester()