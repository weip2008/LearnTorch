from sklearn.datasets import load_breast_cancer
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the breast cancer dataset
def load_breast_cancer_data():
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="CancerType")  # CancerType: malignant or benign
    return X, y

# Run XGBoost and SHAP analysis on the Breast Cancer dataset
def run_xgboost_and_shap_breast_cancer():
    X, y = load_breast_cancer_data()

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the XGBoost classification model
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title("SHAP Summary Plot for Breast Cancer Dataset")
    plt.show()

    # SHAP Dependence Plot for most important feature
    most_important_feature_index = np.argmax(np.abs(shap_values).mean(axis=0))
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(most_important_feature_index, shap_values, X_test, feature_names=X.columns, show=False)
    plt.title(f"SHAP Dependence Plot for Feature: {X.columns[most_important_feature_index]}")
    plt.show()

# Run the SHAP analysis on Breast Cancer dataset
run_xgboost_and_shap_breast_cancer()
