import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths and constants
CSV = "G-Speedway.csv"  # Path to the driving data CSV
BC_MODEL = "bc_model.keras"           # File to save the trained model
SCALER_X = "bc_scaler.gz"             # File to save the input scaler
SCALER_Y = "bc_output_scaler.gz"      # File to save the output scaler
FEATURES_FILE = "bc_features.txt"     # File to save the feature columns
SEQUENCE_LENGTH = 5                   # Number of time steps to use for sequence model

# 1. Load data from CSV and clean it
df = pd.read_csv(CSV).dropna()  # Read and drop any rows with NaNs
print("Available columns in CSV:", df.columns.tolist())

# 2. Define feature (input) columns and output columns
feature_columns = [
    'Angle', 'TrackPosition', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'Z'
]

if 'FuelLevel' in df.columns:
    feature_columns.append('FuelLevel')

if 'RacePosition' in df.columns:
    feature_columns.append('RacePosition')

# Add WheelSpinVelocity_1 to 4
feature_columns += [f'WheelSpinVelocity_{i}' for i in range(1, 5)]

# Add Track_1 to Track_19
feature_columns += [f'Track_{i}' for i in range(1, 20)]

# Add Opponent_1 to Opponent_36
feature_columns += [f'Opponent_{i}' for i in range(1, 37)]

# Output target columns
output_columns = ['Steering', 'Acceleration', 'Braking', 'Clutch', 'Gear']

# 3. Ensure all expected columns are present
available_features = [col for col in feature_columns if col in df.columns]
missing_features = [col for col in feature_columns if col not in df.columns]
print("Using features:", available_features)
print(f"Total number of features: {len(available_features)}")
if missing_features:
    print("Warning: Missing features in data:", missing_features)
    for col in missing_features:
        df[col] = 0.0
        print(f"Added missing feature column '{col}' with default 0.0")

available_outputs = [col for col in output_columns if col in df.columns]
missing_outputs = [col for col in output_columns if col not in df.columns]
print("Using outputs:", available_outputs)
if missing_outputs:
    print("Warning: Missing outputs in data:", missing_outputs)
    for col in missing_outputs:
        df[col] = 0.0
        print(f"Added missing output column '{col}' with default 0.0")

# 4. Feature engineering: derive average track sensor readings
left_indices = [f"Track_{i}" for i in range(1, 7) if f"Track_{i}" in df.columns]
mid_indices  = [f"Track_{i}" for i in range(7, 14) if f"Track_{i}" in df.columns]
right_indices= [f"Track_{i}" for i in range(14, 20) if f"Track_{i}" in df.columns]

df['Track_Left_Avg']   = df[left_indices].mean(axis=1)
df['Track_Middle_Avg'] = df[mid_indices].mean(axis=1)
df['Track_Right_Avg']  = df[right_indices].mean(axis=1)

feature_columns += ['Track_Left_Avg', 'Track_Middle_Avg', 'Track_Right_Avg']
print(f"Total number of features after engineering: {len(feature_columns)}")

# Save feature columns to file
with open(FEATURES_FILE, 'w') as f:
    for col in feature_columns:
        f.write(col + '\n')
print(f"✓ Saved feature columns to '{FEATURES_FILE}'")

# 5. Prepare input and output arrays
X = df[feature_columns].values
y = df[output_columns].values

# 6. Normalize input and output
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)
joblib.dump(scaler_X, SCALER_X)

scaler_y = StandardScaler().fit(y)
y_scaled = scaler_y.transform(y)
joblib.dump(scaler_y, SCALER_Y)

# 7. Convert into sequence format
X_seq = []
y_seq = []
for i in range(SEQUENCE_LENGTH - 1, len(X_scaled)):
    X_seq.append(X_scaled[i - SEQUENCE_LENGTH + 1: i + 1])
    y_seq.append(y_scaled[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"Prepared sequence data: X_seq shape = {X_seq.shape}, y_seq shape = {y_seq.shape}")

# 8. Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# 9. Define the model
model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                        input_shape=(SEQUENCE_LENGTH, X_train.shape[2])))
model.add(layers.GRU(64, return_sequences=False))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(y_train.shape[1], activation='linear'))

# 10. Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
model.summary()
print("Model is compiled and ready to train.")

# 11. Callbacks
checkpoint_cb = ModelCheckpoint(BC_MODEL, monitor='val_loss', save_best_only=True, verbose=1)
#earlystop_cb = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

# 12. Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    #callbacks=[checkpoint_cb, earlystop_cb],
    callbacks=[checkpoint_cb],
    verbose=1
)

# 13. Evaluate model accuracy on validation set
# Predict on validation set
y_val_pred = model.predict(X_val)

# Inverse transform predictions and true values to original scale
y_val_pred_orig = scaler_y.inverse_transform(y_val_pred)
y_val_orig = scaler_y.inverse_transform(y_val)

# Calculate metrics
mse = mean_squared_error(y_val_orig, y_val_pred_orig)
mae = mean_absolute_error(y_val_orig, y_val_pred_orig)
r2 = r2_score(y_val_orig, y_val_pred_orig)

# Print overall metrics
print("\nModel Evaluation on Validation Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Calculate and print MSE and MAE for each output variable
print("\nPer-Output Metrics:")
for i, output_name in enumerate(output_columns):
    mse_output = mean_squared_error(y_val_orig[:, i], y_val_pred_orig[:, i])
    mae_output = mean_absolute_error(y_val_orig[:, i], y_val_pred_orig[:, i])
    print(f"{output_name}: MSE = {mse_output:.4f}, MAE = {mae_output:.4f}")

# 14. Calculate and display accuracy percentages
# Define typical ranges for each output (in original scale)
output_ranges = {
    'Steering': 2.0,      # [-1, 1]
    'Acceleration': 1.0,  # [0, 1]
    'Braking': 1.0,      # [0, 1]
    'Clutch': 1.0,       # [0, 1]
    'Gear': 7.0          # [-1, 6]
}

# Calculate per-output R² scores
r2_per_output = {}
for i, output_name in enumerate(output_columns):
    r2_per_output[output_name] = r2_score(y_val_orig[:, i], y_val_pred_orig[:, i])

# Calculate accuracy percentages
print("\nAccuracy Percentages:")
# Overall accuracy based on R²
overall_accuracy_r2 = r2 * 100
print(f"Overall Model Accuracy (R²-based): {overall_accuracy_r2:.2f}%")

# Per-output accuracies
print("\nPer-Output Accuracies:")
for i, output_name in enumerate(output_columns):
    # R²-based accuracy
    r2_accuracy = r2_per_output[output_name] * 100
    # MAE-based accuracy: 1 - (MAE / range)
    mae = mean_absolute_error(y_val_orig[:, i], y_val_pred_orig[:, i])
    output_range = output_ranges.get(output_name, 1.0)  # Default range 1 if unknown
    mae_accuracy = (1 - mae / output_range) * 100
    print(f"{output_name}:")
    print(f"  R²-based Accuracy: {r2_accuracy:.2f}%")
    print(f"  MAE-based Accuracy: {mae_accuracy:.2f}%")

# 15. Report
best_val_loss = min(history.history['val_loss'])
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"Training complete. Best validation loss: {best_val_loss:.4f} on epoch {best_epoch}.")
print(f"✓ Best model saved to '{BC_MODEL}'")