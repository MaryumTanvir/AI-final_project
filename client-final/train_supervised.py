
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter

# Paths and constants
CSV = "e-track.csv"
BC_MODEL = "bc_model.keras"
SCALER_X = "bc_scaler.gz"
SCALER_Y = "bc_output_scaler.gz"
SEQUENCE_LENGTH = 5

# 1. Load and clean data
df = pd.read_csv(CSV).dropna()
df.columns = df.columns.str.strip()  # Ensure no leading/trailing spaces
df = df.loc[:, ~df.columns.duplicated()]
print("✅ Cleaned DataFrame Columns:", df.columns.tolist())


# 2. Normalize extra features if present
if 'CurrentLapTime' in df.columns:
    df['CurrentLapTime'] /= df['CurrentLapTime'].max()

if 'DistanceCovered' in df.columns:
    df['DistanceCovered'] /= df['DistanceCovered'].max()

if 'DistanceFromStart' in df.columns:
    df['DistanceFromStart'] /= df['DistanceFromStart'].max()

# 3. Define feature columns
feature_columns = [
    'Angle', 'TrackPosition', 'SpeedX', 'SpeedY', 'SpeedZ', 'RPM', 'Z'
]

optional_features = [
    'FuelLevel', 'Clutch',
    *[f'WheelSpinVelocity_{i}' for i in range(1, 5)],
    *[f'Track_{i}' for i in range(1, 20)],
    *[f'Opponent_{i}' for i in range(1, 37)],
    'RacePosition',
]

for col in optional_features:
    if col in df.columns:
        feature_columns.append(col)

# Derived track averages
left_indices = [f"Track_{i}" for i in range(1, 7) if f"Track_{i}" in df.columns]
mid_indices  = [f"Track_{i}" for i in range(7, 14) if f"Track_{i}" in df.columns]
right_indices= [f"Track_{i}" for i in range(14, 20) if f"Track_{i}" in df.columns]

df['Track_Left_Avg'] = df[left_indices].mean(axis=1)
df['Track_Middle_Avg'] = df[mid_indices].mean(axis=1)
df['Track_Right_Avg'] = df[right_indices].mean(axis=1)

feature_columns += ['Track_Left_Avg', 'Track_Middle_Avg', 'Track_Right_Avg']

# Time & distance
for col in ['CurrentLapTime', 'DistanceCovered', 'DistanceFromStart']:
    if col in df.columns:
        feature_columns.append(col)

# Gear (make sure only added once)
if 'Gear' in df.columns and 'Gear' not in feature_columns:
    feature_columns.append('Gear')

# ✅ Remove duplicates
feature_columns = list(dict.fromkeys(feature_columns))

# 🔍 Debug: check for duplicates
dupes = [col for col, count in Counter(feature_columns).items() if count > 1]
if dupes:
    print("❌ Duplicate features found:", dupes)
else:
    print("✅ No duplicates in feature_columns.\n")

# ✅ Final debug
print("✅ Final feature count:", len(feature_columns))
for i, col in enumerate(feature_columns, start=1):
    print(f"{i:2}: {col}")

# 4. Define output columns
output_columns = ['Steering', 'Acceleration', 'Braking', 'Clutch', 'Gear']

# 5. Add missing columns to df if needed
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0.0
        print(f"🟡 Added missing feature column '{col}' with default 0.0")

for col in output_columns:
    if col not in df.columns:
        df[col] = 0.0
        print(f"🟡 Added missing output column '{col}' with default 0.0")

# 6. Create X and y
X = df[feature_columns].values
y = df[output_columns].values

print("\n📊 Final shapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# 7. Normalize inputs and outputs
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)
joblib.dump(scaler_X, SCALER_X)
print("✅ Saved scaler with features:", scaler_X.n_features_in_)
print("\n📦 Comparing feature_columns to df columns:")
actual_cols_used = df[feature_columns].columns.tolist()
print(f"🟡 len(feature_columns): {len(feature_columns)}")
print(f"🟢 len(df[feature_columns].columns): {len(actual_cols_used)}")





scaler_y = StandardScaler().fit(y)
y_scaled = scaler_y.transform(y)
joblib.dump(scaler_y, SCALER_Y)

# 8. Create sequences
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH - 1, len(X_scaled)):
    X_seq.append(X_scaled[i - SEQUENCE_LENGTH + 1: i + 1])
    y_seq.append(y_scaled[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)
print("🧪 Prepared sequences: X_seq shape =", X_seq.shape, ", y_seq shape =", y_seq.shape)

# 9. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
print("📚 Training samples:", X_train.shape[0], ", Validation samples:", X_val.shape[0])

# 10. Define model
model = models.Sequential()
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(SEQUENCE_LENGTH, X_train.shape[2])))
model.add(layers.GRU(64, return_sequences=False))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(y_train.shape[1], activation='linear'))
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
model.summary()

# 11. Train with callbacks
checkpoint_cb = ModelCheckpoint(BC_MODEL, monitor='val_loss', save_best_only=True, verbose=1)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # change to 50 for real training
    batch_size=256,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

# 12. Final Report
best_val_loss = min(history.history['val_loss'])
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f} on epoch {best_epoch}.")
print(f"✓ Best model saved to '{BC_MODEL}'")
