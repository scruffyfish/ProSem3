import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================================
# Step 1: Load preprocessed dataset (adapt to sensor_data_processed.csv)
# ======================================
try:
    # Load preprocessed data (contains denoised and normalized results)
    df = pd.read_csv("sensor_data_processed.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    print("✅ Dataset loaded successfully")
    print(f"✅ Data volume: {len(df)} records (expected 336 records)")
except FileNotFoundError:
    raise FileNotFoundError("❌ sensor_data_processed.csv not found. Please run the data preprocessing code first.")

# ======================================
# Step 2: Data preparation (separate scaler for soil moisture to fix denormalization)
# ======================================
# Select target: denoised soil moisture (raw range, for scaler fitting)
target_denoised = "soil_moisture_denoised"
target_normalized = "soil_moisture_norm"

# Extract denoised data (for scaler fitting and inverse normalization later)
denoised_soil_data = df[target_denoised].values.reshape(-1, 1)

# Extract normalized data (for LSTM training, range [0,1])
normalized_soil_data = df[target_normalized].values.reshape(-1, 1)

# Fit scaler EXCLUSIVELY on denoised soil moisture (fix range mismatch issue)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(denoised_soil_data)
print("\n✅ Scaler fitted exclusively on denoised soil moisture")
print(f"✅ Original data range: [{denoised_soil_data.min():.4f}, {denoised_soil_data.max():.4f}]")
print(f"✅ Normalized data range: [{normalized_soil_data.min():.4f}, {normalized_soil_data.max():.4f}]")

# ======================================
# Step 3: Build LSTM input sequences (time series dedicated format)
# ======================================
def create_lstm_sequences(data, time_step=24):
    """
    Build LSTM input sequences
    :param data: Normalized target data
    :param time_step: Time step (use previous 'time_step' data to predict the next 1 data point)
    :return: X (input features), y (output labels)
    """
    X, y = [], []
    for i in range(len(data) - time_step):
        # Take previous 'time_step' data as input
        X.append(data[i:(i + time_step), 0])
        # Take the (time_step+1)th data as label (prediction target)
        y.append(data[i + time_step, 0])
    # Convert to numpy array to adapt to LSTM input format
    X = np.array(X)
    y = np.array(y)
    # Reshape X to LSTM required format: [samples, time steps, features] (features=1 for univariate prediction)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

# Configure time step (adapt to 7-day data, 24 steps = 1 day of data)
time_step = 24
X, y = create_lstm_sequences(normalized_soil_data, time_step)
print(f"\n✅ LSTM sequences built successfully")
print(f"✅ Input feature shape (X): {X.shape} (Samples: {X.shape[0]}, Time steps: {X.shape[1]}, Features: {X.shape[2]})")
print(f"✅ Output label shape (y): {y.shape}")

# Split into training and test sets (8:2 split, keep time series order, no shuffling)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"\n✅ Training/test set split completed")
print(f"✅ Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# ======================================
# Step 4: Build LSTM model (lightweight and efficient, avoid overfitting)
# ======================================
def build_lstm_model(input_shape):
    """Build lightweight LSTM model"""
    model = Sequential()
    # 1st LSTM layer: 64 neurons, return sequences (for multi-layer LSTM)
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape, activation='tanh'))
    # Dropout layer: prevent overfitting, drop 20% of neurons
    model.add(Dropout(0.2))
    # 2nd LSTM layer: 32 neurons, do not return sequences (last LSTM layer)
    model.add(LSTM(32, activation='tanh'))
    # Dropout layer: further prevent overfitting
    model.add(Dropout(0.2))
    # Fully connected layer: output 1 value (soil moisture prediction result)
    model.add(Dense(1))
    # Compile model: select adam optimizer, loss function uses MSE (regression problem)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build model (input shape = (time steps, features))
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
# Print model structure
model.summary()
print("\n✅ LSTM model built successfully")

# ======================================
# Step 5: Model training (with early stopping to prevent overfitting)
# ======================================
# Configure early stopping: stop training if validation loss does not decrease for 3 epochs, save the best model
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation set loss
    patience=3,          # Tolerate 3 epochs of no decrease
    restore_best_weights=True,  # Restore the best model weights during training
    verbose=1
)

# Start training
print("\n✅ Start training LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=50,            # Maximum training epochs
    batch_size=32,        # Batch size
    validation_data=(X_test, y_test),  # Validation set
    callbacks=[early_stopping],        # Early stopping callback
    verbose=1
)

# ======================================
# Step 6: Model prediction and evaluation (fixed denormalization)
# ======================================
# Model prediction (training set + test set)
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Inverse normalization (restore to original soil moisture range, FIXED: use soil moisture exclusive scaler)
# Reshape data to adapt to scaler requirements
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Inverse normalize all results (ensure consistent range)
y_train_original = scaler.inverse_transform(y_train_reshaped)
y_train_pred_original = scaler.inverse_transform(y_train_pred)
y_test_original = scaler.inverse_transform(y_test_reshaped)
y_test_pred_original = scaler.inverse_transform(y_test_pred)

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train_original, y_train_pred_original)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train_original, y_train_pred_original)

test_mse = mean_squared_error(y_test_original, y_test_pred_original)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_original, y_test_pred_original)

# Print evaluation results
print("\n" + "="*60)
print("✅ Model evaluation results (original data range)")
print("="*60)
print(f"Training set - MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | R²: {train_r2:.6f}")
print(f"Test set - MSE: {test_mse:.6f} | RMSE: {test_rmse:.6f} | R²: {test_r2:.6f}")

# ======================================
# Step 7: Visualize results (training loss trend + true vs predicted values)
# ======================================
# Create figure with 2 subplots
plt.figure(figsize=(12, 10))

# Plot 1: Training loss and validation loss trend
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], color='blue', label='Training Loss')
plt.plot(history.history['val_loss'], color='red', label='Validation Loss')
plt.title('LSTM Model Training & Validation Loss Trend')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend(loc='best')
plt.grid(True, alpha=0.5)

# Plot 2: Test set true vs predicted values (fixed range)
plt.subplot(2, 1, 2)
plt.plot(y_test_original, color='darkgreen', linewidth=2, label='True Soil Moisture (Test Set)')
plt.plot(y_test_pred_original, color='orange', linewidth=2, label='Predicted Soil Moisture (Test Set)')
plt.title('LSTM Model: True vs Predicted Soil Moisture (Test Set)')
plt.xlabel('Sample Index')
plt.ylabel('Soil Moisture (0-1)')
plt.legend(loc='best')
plt.grid(True, alpha=0.5)

# Save visualization results (high resolution, no white edges)
plt.tight_layout()
plt.savefig("lstm_prediction_result.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("\n✅ LSTM prediction result visualization saved as: lstm_prediction_result.png")

# ======================================
# Step 8: Save model (for direct loading and use later, no need to retrain)
# ======================================
model.save("lstm_soil_moisture_model.h5")
print("✅ LSTM model saved as: lstm_soil_moisture_model.h5")
print("\n" + "="*60)
print("✅ All tasks completed! LSTM model training & evaluation finished")
