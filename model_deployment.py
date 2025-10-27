# --- MODEL DEPLOYMENT SCRIPT ---
# This script is designed to run in a production environment.
# It loads the saved model weights and scaler object to make predictions
# on new, unseen data without needing to retrain the model.

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---

# IMPORTANT: Ensure this matches the lookback used during training (Section 11)
LOOKBACK = 48 
# Number of features used in the model (4 traffic metrics + Pkt_Ratio + Total_Mb)
NUM_FEATURES = 6
MODEL_PATH = 'lstm_congestion_model.h5'
SCALER_PATH = 'scaler.pkl'

# --- 2. SEQUENCE CREATION FUNCTION (ESSENTIAL) ---

def create_sequences(data, lookback):
    """
    Converts a time-series array into the 3D structure required by LSTMs:
    (samples, timesteps, features)
    """
    X, Y = [], []
    for i in range(len(data) - lookback):
        # Current 48 timesteps (4 hours)
        X.append(data[i:(i + lookback), :])
        # Target: the next single timestep (t+48)
        Y.append(data[i + lookback, -1]) 
    return np.array(X), np.array(Y)

# --- 3. LOAD MODEL AND SCALER ---

try:
    # Load the trained LSTM model (architecture and weights)
    loaded_model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")

    # Load the fitted scaler object (critical for consistency)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Scaler loaded successfully from: {SCALER_PATH}")

except FileNotFoundError:
    print(f"❌ ERROR: One or both files ({MODEL_PATH}, {SCALER_PATH}) not found.")
    print("Please ensure you ran Section 11 in your Jupyter Notebook and saved the model/scaler.")
    # Exit gracefully if files are missing
    exit() 

# --- 4. GENERATE/LOAD NEW DATA ---

# In a real scenario, this would be a fresh feed of 5-minute data.
# This sample simulates a new 4-hour window (48 timesteps) of data, 
# with 6 features (scaled values expected). 
# We simulate the features being scaled in the range [0, 1].

# Create sample data (48 timesteps x 6 features)
np.random.seed(42)
new_data_unscaled = np.zeros((48, NUM_FEATURES))
# Simulate a low traffic period with a spike near the end
new_data_unscaled[:40, :] = np.random.rand(40, NUM_FEATURES) * 50 
new_data_unscaled[40:, :] = np.random.rand(8, NUM_FEATURES) * 500 # Simulate a peak

# The data must contain the same 6 features, so we scale the simulated unscaled data
new_data_scaled = scaler.transform(new_data_unscaled)

print(f"\nLoaded {len(new_data_unscaled)} new data points for prediction.")

# --- 5. PREPARE INPUT AND FORECAST ---

# Since we only want to predict the *next* single time step (t+1),
# we need one sequence: the last 48 points.
# X_new shape should be (1, 48, 6)
X_new, _ = create_sequences(new_data_scaled, LOOKBACK)

if X_new.shape[0] == 0:
    print("❌ ERROR: Not enough new data points. Need at least 48 timesteps to form one sequence.")
else:
    # Predict the next time step (t+1)
    scaled_prediction = loaded_model.predict(X_new)

    # --- 6. INVERSE TRANSFORM AND REPORT ---

    # We must inverse transform the scaled prediction back to Mb/s.
    # The target (Total_Mb) is the LAST column (index -1 in Python).
    
    # 1. Create a dummy array matching the input feature shape (1, 6)
    dummy_array = np.zeros(shape=(len(scaled_prediction), NUM_FEATURES))
    
    # 2. Insert the single scaled prediction into the last column of the dummy array
    dummy_array[:, -1] = scaled_prediction[:, 0]
    
    # 3. Inverse transform the entire dummy array
    prediction_unscaled = scaler.inverse_transform(dummy_array)[:, -1]
    
    forecast_mb = prediction_unscaled[0]

    # --- FINAL OUTPUT ---
    print("\n=============================================")
    print("      CONGESTION FORECAST (NEXT 5 MINUTES)     ")
    print("=============================================")
    print(f"Model: {MODEL_PATH}")
    print(f"Input: Last {LOOKBACK} Timesteps (4 Hours)")
    print(f"Forecasted Traffic: {forecast_mb:,.2f} Mb/s")
    
    # Assuming your 90th percentile threshold was ~106.31 Mb/s:
    CONGESTION_THRESHOLD = 106.31 
    
    if forecast_mb >= CONGESTION_THRESHOLD:
        print(f"⚠️ **STATUS: HIGH RISK OF CONGESTION**")
        print(f"Policy Action: Recommended to increase VLAN 106 bandwidth.")
    else:
        print(f"✅ STATUS: NORMAL TRAFFIC EXPECTED")
    print("=============================================")
