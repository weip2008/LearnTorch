from sklearn.datasets import load_wine
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the wine dataset
def load_wine_data():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="WineType")  # WineType: classification target
    return X, y

# Run XGBoost and SHAP analysis on the Wine dataset
def run_xgboost_and_shap_wine():
    X, y = load_wine_data()

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the XGBoost classification model
    model = xgb.XGBClassifier(objective='multi:softprob', random_state=42, num_class=3)
    model.fit(X_train, y_train)

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    #plt.figure(figsize=(10, 6))
    #shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    #plt.title("SHAP Summary Plot for Wine Dataset")
    #plt.show()
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # SHAP Dependence Plot for most important feature
    most_important_feature_index = np.argmax(np.abs(shap_values[0]).mean(axis=0))  # shap_values[0] for class 0
    #plt.figure(figsize=(10, 6))
    #shap.dependence_plot(most_important_feature_index, shap_values[0], X_test, feature_names=X.columns, show=False)
    #plt.title(f"SHAP Dependence Plot for Feature: {X.columns[most_important_feature_index]}")
    #plt.show()
    shap.dependence_plot(most_important_feature_index, shap_values[0], X_test, feature_names=X.columns)

# Run the SHAP analysis on Wine dataset
run_xgboost_and_shap_wine()
