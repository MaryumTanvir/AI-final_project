import os
import numpy as np
import joblib
import tensorflow as tf

import msgParser
import carState
import carControl
from collections import deque

# Constants for filenames
BC_MODEL_FILE = "bc_model.keras"
BC_SCALER_FILE = "bc_scaler.gz"
BC_OUTPUT_SCALER_FILE = "bc_output_scaler.gz"
FEATURES_FILE = "bc_features.txt"

class Driver:
    def __init__(self, stage: int = 3, train: bool = False):
        """Initialize the Driver with or without training mode."""
        self.stage = stage        # 0: Warm-Up, 1: Qualifying, 2: Race, 3: Unknown
        self.train_mode = train   # Unused (no RL training logic)
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.max_gear = 6
        self.min_gear = -1
        self.seq_len = 5
        self.seq_buffer = deque(maxlen=self.seq_len)
        self.prev_rpm = None
        self.steer_lock = 0.785398
        self.max_speed = 100.0
        self.off_track_steps = 0
        self.max_off_track_steps = 20  # Reset after 20 steps stuck
        self.use_bc_steering = False   # Set to True to test BC steering
        self.prev_track_pos = 0.0      # Track position history for BC stability

        # Load feature columns
        self.feature_columns = []
        if os.path.exists(FEATURES_FILE):
            with open(FEATURES_FILE, 'r') as f:
                self.feature_columns = [line.strip() for line in f]
            print(f"✓ Loaded {len(self.feature_columns)} feature columns from '{FEATURES_FILE}'")
        else:
            print(f"Error: Feature columns file '{FEATURES_FILE}' not found.")
            raise FileNotFoundError(f"Feature columns file '{FEATURES_FILE}' missing.")

        # Load BC model and scalers
        self.use_bc = False
        self.bc_model = None
        self.bc_scaler = None
        self.bc_output_scaler = None
        self.expected_feature_count = len(self.feature_columns)
        if os.path.exists(BC_MODEL_FILE) and os.path.exists(BC_SCALER_FILE) and os.path.exists(BC_OUTPUT_SCALER_FILE):
            try:
                self.bc_model = tf.keras.models.load_model(BC_MODEL_FILE)
                self.bc_scaler = joblib.load(BC_SCALER_FILE)
                self.bc_output_scaler = joblib.load(BC_OUTPUT_SCALER_FILE)
                self.expected_feature_count = self.bc_scaler.n_features_in_
                if len(self.feature_columns) != self.expected_feature_count:
                    print(f"Warning: Feature columns count ({len(self.feature_columns)}) does not match scaler ({self.expected_feature_count}).")
                self.use_bc = True
                print(f"✓ Loaded behavior cloning model and scalers. Expecting {self.expected_feature_count} features.")
            except Exception as e:
                print(f"Warning: Failed to load BC model/scalers: {e}")
                self.use_bc = False
        else:
            print("Warning: BC model or scalers not found. Using rule-based controls.")

    def init(self) -> str:
        """Configure sensors with rule-based angles."""
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        self.angles[9] = 0
        return self.parser.stringify({'init': self.angles})

    def _get_state_vector(self) -> np.ndarray:
        """Compile the current car state into a feature vector."""
        track = self.state.track if self.state.track is not None else [200.0] * 19
        opponents = self.state.opponents if self.state.opponents is not None else [200.0] * 36
        wheel_spin = self.state.wheelSpinVel if self.state.wheelSpinVel is not None else [0.0] * 4

        # Fix invalid track sensor values
        track = [200.0 if x <= 0 else x for x in track]

        features = {
            'Angle': self.state.angle or 0.0,
            'TrackPosition': self.state.trackPos or 0.0,
            'SpeedX': self.state.getSpeedX() or 0.0,
            'SpeedY': self.state.getSpeedY() or 0.0,
            'SpeedZ': self.state.getSpeedZ() or 0.0,
            'RPM': self.state.getRpm() or 0.0,
            'Z': self.state.z or 0.0,
            'FuelLevel': getattr(self.state, "fuel", 0.0),
            'RacePosition': getattr(self.state, "racePos", 1.0),
        }

        # Log FuelLevel and Track statistics
        print(f"Parsed FuelLevel: {features['FuelLevel']:.3f}")
        print(f"Track Stats: min={min(track):.1f}, max={max(track):.1f}, mean={np.mean(track):.1f}")

        for i in range(4):
            features[f'WheelSpinVelocity_{i+1}'] = wheel_spin[i]
        for i in range(19):
            features[f'Track_{i+1}'] = track[i]
        for i in range(36):
            features[f'Opponent_{i+1}'] = opponents[i]

        features['Track_Left_Avg'] = float(np.mean(track[0:6]))
        features['Track_Middle_Avg'] = float(np.mean(track[6:13]))
        features['Track_Right_Avg'] = float(np.mean(track[13:19]))

        state_vector = []
        missing_features = []
        for col in self.feature_columns:
            if col in features:
                state_vector.append(features[col])
                print(f"Feature '{col}': {features[col]:.3f}")
            else:
                print(f"Warning: Feature '{col}' not available in state. Using 0.0.")
                state_vector.append(0.0)
                missing_features.append(col)

        if missing_features:
            print(f"Missing features: {missing_features}")

        print(f"Generated feature vector length: {len(state_vector)}")
        if len(state_vector) != self.expected_feature_count:
            print(f"Error: Feature vector length mismatch. Got {len(state_vector)}, expected {self.expected_feature_count}.")
            print(f"Features included: {[col for col in self.feature_columns if col in features]}")

        return np.array(state_vector, dtype=np.float32)

    def bc_action(self):
        """Predict control actions, using rule-based steering and gear."""
        rpm = self.state.getRpm() or 0.0
        track_pos = self.state.trackPos or 0.0

        if not self.use_bc:
            return self.rule_based_action()

        try:
            raw_vec = self._get_state_vector().reshape(1, -1)
            scaled_vec = self.bc_scaler.transform(raw_vec)[0]

            if len(self.seq_buffer) == 0:
                for _ in range(self.seq_len):
                    self.seq_buffer.append(scaled_vec.copy())
            else:
                self.seq_buffer.append(scaled_vec)

            seq_input = np.stack(self.seq_buffer, axis=0).reshape(1, self.seq_len, -1)
            y_scaled = self.bc_model.predict(seq_input, verbose=0)
            action = self.bc_output_scaler.inverse_transform(y_scaled)[0]

        except Exception as e:
            print(f"[BC] Prediction error: {e}. Using rule-based action.")
            self.prev_rpm = rpm
            return self.rule_based_action()

        steer_bc, accel_bc, brake_bc, clutch_bc, gear_bc = action

        # Rule-based steering with damping
        angle = self.state.angle or 0.0
        steer = (angle - track_pos * 0.5) / self.steer_lock
        steer *= 0.8
        steer = float(np.clip(steer, -1.0, 1.0))

        # Use BC steering if enabled, fallback if trackPos worsens
        if self.use_bc_steering:
            if abs(track_pos) > abs(self.prev_track_pos) and abs(track_pos) > 0.5:
                print(f"BC steering unstable (trackPos={track_pos:.3f}). Falling back to rule-based.")
                steer_final = steer
            else:
                steer_final = float(np.clip(steer_bc, -1.0, 1.0))
                print(f"Using BC steer: {steer_final:.3f}")
        else:
            steer_final = steer
            print(f"BC steer prediction: {steer_bc:.3f}, Using rule-based steer: {steer_final:.3f}")

        self.prev_track_pos = track_pos

        # Off-track recovery
        speed = self.state.getSpeedX() or 0.0
        accel = self.control.getAccel() or 0.0
        gear = self.state.getGear() or 1
        brake = 0.0
        damage = getattr(self.state, "damage", 0)
        if (abs(track_pos) > 1.2 and speed < 5.0) or damage > 2500:
            self.off_track_steps += 1
            if self.off_track_steps > self.max_off_track_steps:
                print("Car stuck off-track or heavily damaged. Requesting reset.")
                self.off_track_steps = 0
                self.prev_rpm = rpm
                return None, None, None, None, None
            print("Off-track detected. Reversing.")
            gear = -1
            accel = 0.3  # Reduced for controlled reversing
            brake = 0.0
            steer_final = -angle * 2 / self.steer_lock
            steer_final = float(np.clip(steer_final, -1.0, 1.0))
        else:
            self.off_track_steps = max(0, self.off_track_steps - 1)
            # Rule-based accel
            if speed < self.max_speed:
                accel += 0.1
                if accel > 1.0:
                    accel = 1.0
            else:
                accel -= 0.1
                if accel < 0.0:
                    accel = 0.0
            # Rule-based gear
            if speed < 30.0:
                gear = 1
            else:
                up = True if self.prev_rpm is None or (self.prev_rpm - rpm) < 0 else False
                gear = self.state.getGear() or 1
                if up and rpm > 7000:
                    gear += 1
                if not up and rpm < 3000:
                    gear -= 1
                gear = max(1, min(self.max_gear, gear))

        clutch = 0.0
        self.prev_rpm = rpm

        return steer_final, accel, brake, clutch, gear

    def rule_based_action(self):
        """Full rule-based control logic."""
        rpm = self.state.getRpm() or 0.0
        angle = self.state.angle or 0.0
        track_pos = self.state.trackPos or 0.0
        steer = (angle - track_pos * 0.5) / self.steer_lock
        steer *= 0.8
        steer = float(np.clip(steer, -1.0, 1.0))

        speed = self.state.getSpeedX() or 0.0
        accel = self.control.getAccel() or 0.0
        gear = self.state.getGear() or 1
        brake = 0.0
        damage = getattr(self.state, "damage", 0)

        if (abs(track_pos) > 1.2 and speed < 5.0) or damage > 2500:
            self.off_track_steps += 1
            if self.off_track_steps > self.max_off_track_steps:
                print("Car stuck off-track or heavily damaged. Requesting reset.")
                self.off_track_steps = 0
                self.prev_rpm = rpm
                return None, None, None, None, None
            print("Off-track detected. Reversing.")
            gear = -1
            accel = 0.3  # Reduced for controlled reversing
            brake = 0.0
            steer = -angle * 2 / self.steer_lock
            steer = float(np.clip(steer, -1.0, 1.0))
        else:
            self.off_track_steps = max(0, self.off_track_steps - 1)
            if speed < self.max_speed:
                accel += 0.1
                if accel > 1.0:
                    accel = 1.0
            else:
                accel -= 0.1
                if accel < 0.0:
                    accel = 0.0
            if speed < 30.0:
                gear = 1
            else:
                up = True if self.prev_rpm is None or (self.prev_rpm - rpm) < 0 else False
                if up and rpm > 7000:
                    gear += 1
                if not up and rpm < 3000:
                    gear -= 1
                gear = max(1, min(self.max_gear, gear))

        clutch = 0.0
        self.prev_rpm = rpm
        return steer, accel, brake, clutch, gear

    def drive(self, msg: str) -> str:
        """Main driving function."""
        self.state.setFromMsg(msg)
        print(f"Raw TORCS message: {msg[:500]}...")
        steer, accel, brake, clutch, gear = self.bc_action()
        if steer is None:  # Reset requested
            return '(meta 1)'
        self.control.steer = steer
        self.control.accel = accel
        self.control.brake = brake
        self.control.clutch = clutch
        self.control.gear = gear
        track = self.state.track if self.state.track is not None else [200.0] * 19
        track = [200.0 if x <= 0 else x for x in track]  # Fix for logging
        opponents = self.state.opponents if self.state.opponents is not None else [200.0] * 36
        print(f"Sensors: angle={self.state.angle or 0.0:.3f}, trackPos={self.state.trackPos or 0.0:.3f}, "
              f"speedX={self.state.getSpeedX() or 0.0:.3f}, rpm={self.state.getRpm() or 0.0:.3f}, "
              f"damage={getattr(self.state, 'damage', 0):.1f}, "
              f"track=[{', '.join([f'{x:.1f}' for x in track])}], "
              f"opponents=[{', '.join([f'{x:.1f}' for x in opponents])}]")
        print(f"Controls: steer={steer:.3f}, accel={accel:.3f}, brake={brake:.3f}, clutch={clutch:.3f}, gear={gear}")
        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        self.prev_rpm = None
        self.off_track_steps = 0
        self.prev_track_pos = 0.0

    def reward(self):
        return 0.0