def calculate_mse(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("Lists must have the same length")
    return sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)

# Example usage
actual_values = [1, 2, 3, 4, 5]
predicted_values = [1.1, 2.2, 3.3, 4.4, 5.5]

mse = calculate_mse(actual_values, predicted_values)
print("Mean Squared Error:", mse)
