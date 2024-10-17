from sklearn.datasets import load_diabetes
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the diabetes dataset
def load_diabetes_data():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name="DiseaseProgression")  # Disease progression is the target
    return X, y

# Run XGBoost and SHAP analysis on the Diabetes dataset
def run_xgboost_and_shap_diabetes():
    X, y = load_diabetes_data()

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the XGBoost regression model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title("SHAP Summary Plot for Diabetes Dataset")
    plt.show()

    # SHAP Dependence Plot for most important feature
    most_important_feature_index = np.argmax(np.abs(shap_values).mean(axis=0))
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(most_important_feature_index, shap_values, X_test, feature_names=X.columns, show=False)
    plt.title(f"SHAP Dependence Plot for Feature: {X.columns[most_important_feature_index]}")
    plt.show()

# Run the SHAP analysis on Diabetes dataset
run_xgboost_and_shap_diabetes()
