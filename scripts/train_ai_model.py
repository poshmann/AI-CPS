import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Load the training data
data_path = os.path.join("data", "trainingdata.csv")
data = pd.read_csv(data_path)

# Step 2: Separate features and target
X = data.drop(columns=["Severity", "P_CODE", "VDC_NAME", "DISTRICT", "REGION", "Severity category"])  # Drop irrelevant columns
y = data["Severity"]  # Replace with your target column name

# Step 3: Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Build the AI model
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 5: Train the model
history = model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 6: Save the trained model
model_save_path = os.path.join("models", "currentAiSolution.h5")
os.makedirs("models", exist_ok=True)
model.save(model_save_path)

print(f"AI model saved to {model_save_path}")
