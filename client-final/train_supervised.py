import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths and constants
CSV = "../complete_driving_data.csv"  # Path to the driving data CSV
BC_MODEL = "bc_model.keras"          # File to save the trained model
SCALER_X = "bc_scaler.gz"           # File to save the input scaler
SCALER_Y = "bc_output_scaler.gz"    # File to save the output scaler
SEQUENCE_LENGTH = 5                 # Number of time steps to use for sequence model

# 1. Load data from CSV and clean it
df = pd.read_csv(CSV).dropna()               # read data and drop any rows with NaNs
df.columns = df.columns.str.strip().str.lower()  # normalize column names to lowercase (no leading/trailing spaces)
print("Available columns in CSV:", df.columns.tolist())

# 2. Define feature (input) columns and output columns
#    Include all relevant sensor and state telemetry as features.
feature_columns = [
    'angle', 'trackpos', 'speedx', 'speedy', 'speedz', 'rpm', 'z'
]
# Include fuel if present (car state might include fuel level)
if 'fuel' in df.columns:
    feature_columns.append('fuel')
# Include wheel speed sensors, track sensors, opponent sensors
feature_columns += [f'wheel_{i}' for i in range(4)]
feature_columns += [f'track_{i}' for i in range(19)]
feature_columns += [f'opp_{i}' for i in range(36)]
# (Note: we do not include 'gear' here as an input feature to avoid using the target gear directly as input)

output_columns = ['steer', 'accel', 'brake', 'clutch', 'gear', 'meta']  # target control actions to predict

# 3. Ensure all feature and output columns are present in the DataFrame
#    If any expected feature or output column is missing, add it with default value 0.0.
available_features = [col for col in feature_columns if col in df.columns]
missing_features = [col for col in feature_columns if col not in df.columns]
print("Using features:", available_features)
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

# 4. Feature engineering: derive average left, middle, and right track sensor values
#    (These provide summarized track distance information to help the model)
left_indices = [f"track_{i}" for i in range(0, 6) if f"track_{i}" in df.columns]    # first 6 track sensors (left side)
mid_indices  = [f"track_{i}" for i in range(6, 13) if f"track_{i}" in df.columns]   # middle 7 track sensors
right_indices= [f"track_{i}" for i in range(13, 19) if f"track_{i}" in df.columns]  # last 6 track sensors (right side)
df['track_left_avg']   = df[left_indices].mean(axis=1)
df['track_middle_avg'] = df[mid_indices].mean(axis=1)
df['track_right_avg']  = df[right_indices].mean(axis=1)
# Add the new derived feature names to the feature list
feature_columns += ['track_left_avg', 'track_middle_avg', 'track_right_avg']

# 5. Prepare input features (X) and target outputs (y) as NumPy arrays
X = df[feature_columns].values
y = df[output_columns].values

# 6. Normalize inputs and outputs using sklearn StandardScaler
#    This scales each feature to zero-mean, unit-variance based on the training data distribution.
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)
joblib.dump(scaler_X, SCALER_X)  # save the input scaler to file

scaler_y = StandardScaler().fit(y)
y_scaled = scaler_y.transform(y)
joblib.dump(scaler_y, SCALER_Y)  # save the output scaler to file

# 7. Convert the dataset into time-sequence format to capture the last SEQUENCE_LENGTH steps
#    Each training sample will consist of `SEQUENCE_LENGTH` consecutive time steps of sensor data.
seq_length = SEQUENCE_LENGTH
X_seq = []
y_seq = []
# Loop over the data to create sequences of length 5
for i in range(seq_length - 1, len(X_scaled)):
    # Take the sequence of features from time (i-seq_length+1) to time i (inclusive)
    X_seq.append(X_scaled[i-seq_length+1 : i+1])
    # The target for this sequence will be the control actions at time i
    y_seq.append(y_scaled[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"Prepared sequence data: X_seq shape = {X_seq.shape}, y_seq shape = {y_seq.shape}")
# (If the data contains multiple independent driving sessions, ensure sequences do not cross session boundaries.)

# 8. Split into training and validation sets for model evaluation
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# 9. Define the neural network model (Conv1D + GRU for sequence modeling)
#    The model takes a sequence of the last 5 time steps of sensor data and outputs the 6 control values.
model = models.Sequential()
# First layer: 1D convolution over time axis to extract short-term temporal features
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same',
                        input_shape=(seq_length, X_train.shape[2])))
# Second layer: GRU (Gated Recurrent Unit) to capture longer-term temporal dependencies from the conv features
model.add(layers.GRU(64, return_sequences=False))  # GRU outputs a single vector (final state) for the sequence
# Additional dense layers for processing the GRU output
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
# Output layer: linear outputs for steer, accel, brake, clutch, gear, meta (6 values)
model.add(layers.Dense(y_train.shape[1], activation='linear'))

# 10. Compile the model with an optimizer and loss function for regression
model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')
model.summary()  # print model architecture summary
print("Model is compiled and ready to train.")

# 11. Set up callbacks for early stopping and model checkpointing
#     - ModelCheckpoint will save the best model (lowest validation loss) to disk.
#     - EarlyStopping will halt training early if validation loss stops improving.
checkpoint_cb = ModelCheckpoint(BC_MODEL, monitor='val_loss', save_best_only=True, verbose=1)
earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 12. Train the model on the training set, with validation on the hold-out set
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

# 13. Training finished - print final loss values and confirm model saving
best_val_loss = min(history.history['val_loss'])
best_epoch = np.argmin(history.history['val_loss']) + 1
print(f"Training complete. Best validation loss: {best_val_loss:.4f} on epoch {best_epoch}.")
print(f"✓ Best model saved to '{BC_MODEL}'")
