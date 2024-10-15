import shap
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost regression model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.Explainer(model)

# Compute SHAP values for the test set
shap_values = explainer(X_test)

# Extract SHAP values and feature names
shap_values_matrix = shap_values.values
feature_names = shap_values.feature_names

# Compute the mean absolute SHAP values for each feature
mean_abs_shap_values = np.abs(shap_values_matrix).mean(axis=0)

# Visualize the SHAP values with a summary plot
shap.summary_plot(shap_values, X_test)

# Compute SHAP interaction values (optional)
shap_interaction_values = explainer.shap_interaction_values(X_test)

# Plot an interaction summary plot
shap.summary_plot(shap_interaction_values, X_test)

# Plot a dependence plot with interaction between two features
shap.dependence_plot(0, shap_values.values, X_test, interaction_index=1)
