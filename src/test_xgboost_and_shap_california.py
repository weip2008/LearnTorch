import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # This is the missing import

# Step 1: Load the California Housing Dataset
def load_california_data():
    # Load dataset from sklearn
    cali = fetch_california_housing()
    X = pd.DataFrame(cali.data, columns=cali.feature_names)
    y = pd.Series(cali.target, name="MedHouseVal")  # MedHouseVal is the target (median house value)
    
    return X, y

# Step 2: Preprocessing, Model Training, and SHAP Analysis
def run_xgboost_and_shap_california():
    # Load the dataset
    X, y = load_california_data()

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

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
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)  # Use actual feature names
    plt.title("SHAP Summary Plot for California Housing Dataset")  # Add title
    plt.show()  # Now display the plot with the title

    # Create SHAP dependence plot with title customization
    most_important_feature_index = np.argmax(np.abs(shap_values).mean(axis=0))  # Fixed the error here
    plt.figure(figsize=(10, 6))  # Create a new figure for the second plot
    shap.dependence_plot(most_important_feature_index, shap_values, X_test, feature_names=X.columns, show=False)
    plt.title(f"SHAP Dependence Plot for Feature: {X.columns[most_important_feature_index]}")  # Add title
    plt.show()  # Now display the plot with the title
    
    # Step 4: Feature Importance
    # Compute the mean absolute SHAP values to determine top features
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    
    # Get the indices of the top 3 most important features
    top_3_indices = np.argsort(mean_abs_shap_values)[-3:]
    
    # Extract and print the top 3 feature names
    top_3_features = X.columns[top_3_indices]
    print("Top 3 Features:", top_3_features)

# Run the model and SHAP analysis on the California Housing dataset
run_xgboost_and_shap_california()
