import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Data Simulation
def generate_synthetic_stock_data(n=1000):
    np.random.seed(42)
    
    # Simulate dates
    dates = pd.date_range(start='2022-01-01', periods=n)
    
    # Simulate stock prices and other features
    open_prices = np.random.uniform(100, 200, n)
    high_prices = open_prices + np.random.uniform(0, 10, n)
    low_prices = open_prices - np.random.uniform(0, 10, n)
    close_prices = open_prices + (np.random.uniform(-5, 5, n) + 0.05 * (high_prices - low_prices))
    
    # Simulate volume (in millions)
    volume = np.random.uniform(1, 5, n) * 1e6
    
    # Simulate MACD (Moving Average Convergence Divergence) and Volatility
    macd = np.random.uniform(-2, 2, n)
    volatility = np.random.uniform(0.01, 0.05, n)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,  # Target variable
        'Volume': volume,
        'MACD': macd,
        'Volatility': volatility
    })
    
    return df

# Step 2: Preprocessing, Model Training, and SHAP Analysis
def run_xgboost_and_shap():
    # Generate synthetic stock data
    df = generate_synthetic_stock_data()
    
    # Define features (drop 'Date' and 'Close' as the latter is our target)
    X = df.drop(columns=['Date', 'Close'])
    y = df['Close']  # Target variable is the 'Close' price
    
    # Normalize features like MACD and Volatility
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['MACD', 'Volatility']] = scaler.fit_transform(X[['MACD', 'Volatility']])
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train an XGBoost regression model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    # Step 3: SHAP Explanation
    # Create a SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test)
    
    # Create SHAP summary plot with title customization
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)  # Disable automatic display
    plt.title("SHAP Summary Plot for Global Feature Importance")  # Add title
    plt.show()  # Now display the plot with the title

    # Create SHAP dependence plot with title customization
    most_important_feature_index = np.argmax(np.abs(shap_values).mean(axis=0))
    plt.figure(figsize=(10, 6))  # Create a new figure for the second plot
    shap.dependence_plot(most_important_feature_index, shap_values, X_test, show=False)  # Disable automatic display
    plt.title(f"SHAP Dependence Plot for Feature: {X_test.columns[most_important_feature_index]}")  # Add title
    plt.show()  # Now display the plot with the title
    
    # Step 4: Feature Importance
    # Compute the mean absolute SHAP values to determine top features
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    
    # Get the indices of the top 3 most important features
    top_3_indices = np.argsort(mean_abs_shap_values)[-3:]
    
    # Extract and print the top 3 feature names
    top_3_features = np.array(X_test.columns)[top_3_indices]
    print("Top 3 Features:", top_3_features)

# Run the model and SHAP analysis
run_xgboost_and_shap()
