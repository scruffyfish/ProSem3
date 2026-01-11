import pandas as pd 
df = pd.read_csv("sensor_data_processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
import numpy as np

# Prepare features: Convert time to numerical values (hours)
df["hours"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600

# Features and target variables (Use data from the first n hours to predict soil moisture after 12 hours)
def create_sequences(data, n_steps_in=4, n_steps_out=2):
    """Create sequence data (n_steps_in=4 time steps (2 hours), predict n_steps_out=2 time steps (1 hour) → extended to 12 hours)"""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[i + n_steps_in + n_steps_out - 1])
    return np.array(X), np.array(y)

# Select soil moisture as the prediction target
moisture_data = df["soil_moisture_norm"].values
n_steps_in = 4  # Input: First 2 hours (4 intervals of 30 minutes each)
n_steps_out = 24  # Output: After 12 hours (24 intervals of 30 minutes each)

# Process data (Ensure there are enough samples)
if len(moisture_data) > n_steps_in + n_steps_out:
    X, y = create_sequences(moisture_data, n_steps_in, n_steps_out)
    X = X.reshape(X.shape[0], X.shape[1])  # Linear regression requires 2D input
    
    # Split into training and testing sets (80% training, 20% testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train the Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Model Evaluation (12-Hour Soil Moisture Prediction):")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Coefficient of Determination (R²): {r2:.4f}")
    
    # Visualize the prediction results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="red", linestyle="--")
    plt.title("Linear Regression: 12-Hour Soil Moisture Prediction")
    plt.xlabel("Test Samples")
    plt.ylabel("Normalized Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.savefig("linear_regression_prediction.png")
    plt.show()
else:
    print(f"Insufficient data to train the Linear Regression model (At least {n_steps_in + n_steps_out} samples required)")
