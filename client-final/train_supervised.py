import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
CSV = "../complete_driving_data.csv"
BC_MODEL = "bc_model.keras"
SCALER_X = "bc_scaler.gz"
SCALER_Y = "bc_output_scaler.gz"

# Load and clean data
df = pd.read_csv(CSV).dropna()
df.columns = df.columns.str.strip().str.lower()  # Normalize column names to lowercase
print("Available columns in CSV:", df.columns.tolist())

# Feature columns (lowercase to match normalized CSV headers)
feature_columns = [
    'angle', 'trackpos', 'speedx', 'speedy', 'speedz', 'rpm', 'z',
    'wheel_0', 'wheel_1', 'wheel_2', 'wheel_3',
    *[f"track_{i}" for i in range(19)],
    *[f"opp_{i}" for i in range(36)]
]

# Ensure all features are present, add defaults if missing
available_features = [col for col in feature_columns if col in df.columns]
missing_features = [col for col in feature_columns if col not in df.columns]
print("Using features:", available_features)
if missing_features:
    print("Warning: Missing features:", missing_features)
    for col in missing_features:
        df[col] = 0.0
        print(f"Added default value 0.0 for missing feature: {col}")

# Outputs (also lowercase)
output_columns = ['steer', 'accel', 'brake', 'clutch', 'gear', 'meta']

# Check and add missing outputs with defaults
available_outputs = [col for col in output_columns if col in df.columns]
missing_outputs = [col for col in output_columns if col not in df.columns]
print("Using outputs:", available_outputs)
if missing_outputs:
    print("Warning: Missing outputs:", missing_outputs)
    for col in missing_outputs:
        df[col] = 0.0
        print(f"Added default value 0.0 for missing output: {col}")

# Inputs and targets
X = df[feature_columns].values
y = df[output_columns].values

# Normalize inputs and outputs
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)
joblib.dump(scaler_X, SCALER_X)

scaler_y = StandardScaler().fit(y)
y_scaled = scaler_y.transform(y)
joblib.dump(scaler_y, SCALER_Y)

# Train/test split
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

# Model
model = models.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(y.shape[1], activation='linear')  # 6 outputs
])
model.compile(optimizer=optimizers.Adam(1e-3), loss='mean_squared_error')

# Save best model only
checkpoint = ModelCheckpoint(
    BC_MODEL,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    Xtr, ytr,
    validation_data=(Xte, yte),
    epochs=50,
    batch_size=256,
    callbacks=[checkpoint],
    verbose=1
)

print(f"\u2713 Best model saved to {BC_MODEL} with lowest val_loss: {min(history.history['val_loss']):.4f}")