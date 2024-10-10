import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

# Load your dataset
data = pd.read_csv('data/LULU_max_2024-05-28_2024-05-31_1m.csv')

# Convert the 'Date' column to datetime
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Extract useful features: day of the week and time in minutes
data['DayOfWeek'] = data['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
data['TimeInMinutes'] = data['Datetime'].dt.hour * 60 + data['Datetime'].dt.minute  # Total minutes since midnight

# Optionally drop the original date column if not needed
data.drop(columns=['Datetime','Adj Close'], inplace=True)


# Check data types
print(data.dtypes)

# Encode categorical features if necessary
data = pd.get_dummies(data, drop_first=True)  # This will convert categorical variables to dummy variables

# Ensure to use the correct target column name
y = data['Close']  # Use the actual column name for the stock price
X = data.drop(columns=['Close'])  # Drop the target column from features

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.Explainer(model)

# Compute SHAP values for the test set
shap_values = explainer(X_test)

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test)

