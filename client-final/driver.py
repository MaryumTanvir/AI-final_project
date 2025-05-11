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


class Driver:
    def __init__(self, stage: int = 3, train: bool = False):
        """Initialize the Driver with or without training mode. 
        Loads the behavior cloning model and scalers if available."""
        self.stage = stage        # Unused in behavior cloning, but kept for compatibility
        self.train_mode = train   # Unused (no RL training logic in this driver)
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.max_gear = 6  # maximum forward gear
        self.seq_len = 5                   # model expects 5 frames
        self.seq_buffer = deque(maxlen=self.seq_len)

        # Load Behavior Cloning model and scalers
        self.use_bc = False
        self.bc_model = None
        self.bc_scaler = None
        self.bc_output_scaler = None
        if os.path.exists(BC_MODEL_FILE) and os.path.exists(BC_SCALER_FILE) and os.path.exists(BC_OUTPUT_SCALER_FILE):
            try:
                # Load trained Keras model
                self.bc_model = tf.keras.models.load_model(BC_MODEL_FILE)
                # Load pre-fitted scalers (using joblib for pickled StandardScaler objects)
                self.bc_scaler = joblib.load(BC_SCALER_FILE)
                self.bc_output_scaler = joblib.load(BC_OUTPUT_SCALER_FILE)
                self.use_bc = True
                print("✓ Loaded behavior cloning model and scalers.")
            except Exception as e:
                print(f"Warning: Failed to load BC model/scalers: {e}")
                self.use_bc = False
        else:
            print("Warning: BC model or scalers not found. Running without behavior cloning.")
            self.use_bc = False

    def init(self) -> str:
        """Build the init string to configure sensors."""
        angles = [-90 + i * 10 for i in range(19)]
        return self.parser.stringify({
            'init': angles
              
        })


    def _get_state_vector(self) -> np.ndarray:
        """Compile the current car state into the 76-dim feature vector the BC model expects."""

        try:
            # 1. Sensor arrays with safety defaults
            track = self.state.getTrack() or [0.0] * 19
            opponents = self.state.getOpponents() or [0.0] * 36
            wheel_spin = self.state.getWheelSpinVel() or [0.0] * 4

            # 2. Core scalars with fallback to 0.0
            angle = self.state.getAngle() or 0.0
            track_pos = self.state.getTrackPos() or 0.0
            speed_x = self.state.getSpeedX() or 0.0
            speed_y = self.state.getSpeedY() or 0.0
            speed_z = self.state.getSpeedZ() or 0.0
            rpm = self.state.getRpm() or 0.0
            z = self.state.getZ() or 0.0
            fuel = self.state.getFuel() or 0.0
            clutch = self.state.clutch or 0.0  # no getter, uses direct access
            race_pos = self.state.getRacePos() or 0.0
            gear = self.state.getGear() or 1

            # 3. Optional scalars (normalized)
            current_lap_time = self.state.getCurLapTime() or 0.0
            distance_covered = self.state.getDistRaced() or 0.0
            distance_from_start = self.state.getDistFromStart() or 0.0

            # Normalize time and distances
            current_lap_time /= 100.0
            distance_covered /= 10000.0
            distance_from_start /= 10000.0

            # 4. Derived averages
            left_avg = float(np.mean(track[0:6]))
            mid_avg = float(np.mean(track[6:13]))
            right_avg = float(np.mean(track[13:19]))

            # 5. Final state vector
            state_vector = [
                angle, track_pos, speed_x, speed_y, speed_z, rpm, z, fuel, clutch,
                *wheel_spin, *track, *opponents,
                race_pos,
                left_avg, mid_avg, right_avg,
                current_lap_time, distance_covered, distance_from_start,
                gear
            ]

            if len(state_vector) != 76:
                raise ValueError(f"Expected 76 features, got {len(state_vector)}")

            return np.array(state_vector, dtype=np.float32)

        except Exception as e:
            print(f"[ERROR] _get_state_vector failed: {e}")
            raise



    def bc_action(self):
        """
        Use the behavior-cloning network to predict (steer, accel, brake, clutch, gear).
        Returns five values ready to be copied into self.control.* (meta = 0 by default).
        """
        if not self.use_bc:
            print("not loaded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return 0.0, 0.0, 0.0, 0.0, 1  # default values if model is not loaded

        try:
            # Step 1: Get current feature vector and scale it
            raw_vec = self._get_state_vector().reshape(1, -1)  # shape (1, 70)
            scaled_vec = self.bc_scaler.transform(raw_vec)[0]  # shape (70,)

            # Step 2: Fill sequence buffer with scaled input
            if len(self.seq_buffer) == 0:
                for _ in range(self.seq_len):
                    self.seq_buffer.append(scaled_vec.copy())
            else:
                self.seq_buffer.append(scaled_vec)

            # Step 3: Format sequence input to (1, 5, 70) and predict
            seq_input = np.stack(self.seq_buffer, axis=0).reshape(1, self.seq_len, -1)
            y_scaled = self.bc_model.predict(seq_input, verbose=0)  # shape (1, 5)
            action = self.bc_output_scaler.inverse_transform(y_scaled)[0]  # shape (5,)

        except Exception as e:
            print(f"[BC] Prediction error: {e}. Using fallback control.")
            return 0.0, 0.0, 0.0, 0.0, 1

        # Unpack predicted outputs
        steer, accel, brake, clutch, gear = action

        # Post-process each control
        steer  = float(np.clip(steer, -1.0, 1.0))  # negate because training used reversed steering
        accel  = float(np.clip(accel, 0.0, 1.0))
        brake  = float(np.clip(brake, 0.0, 1.0))
        clutch = float(np.clip(clutch, 0.0, 1.0))

        # Gear: clean rounding + safety
        if not np.isfinite(gear):
            gear = 1
        try:
            gear = int(round(gear))
        except Exception:
            gear = int(gear) if isinstance(gear, (int, np.integer)) else 1
        gear = max(1, min(self.max_gear, gear))

        return steer, accel, brake, clutch, gear


    def drive(self, msg: str) -> str:
        """Main driving function called every time step with sensor data `msg`.
        Parses the sensor data, computes control actions, and returns a command string."""
        # Update the CarState from the incoming message string
        self.state.setFromMsg(msg)

        # Get the model-predicted action (or default if no model)
        steer, accel, brake, clutch, gear = self.bc_action()

        # Apply the predicted control values to the CarControl object
        self.control.steer = steer
        self.control.accel = accel
        self.control.brake = brake
        self.control.clutch = clutch
        self.control.gear = gear
        
        # (Optional) Print or log some info for debugging/monitoring
        # print(f"steer={steer:.3f}, accel={accel:.3f}, brake={brake:.3f}, gear={gear}, meta={meta}")

        # Return the control action as a message string for TORCS
        return self.control.toMsg()

    def onShutDown(self):
        """Called when the simulation is shutting down. Clean up if needed."""
        # No special cleanup required (we could close files or save logs here if any)
        pass

    def onRestart(self):
        """Called when an episode restarts. Reset any necessary state."""
        # Nothing to reset in pure BC mode (no persistent episode state kept)
        pass

    def reward(self):
        """[Optional] Compute a reward for training (not used in BC inference mode)."""
        # In pure behavior cloning, we don't use a reward function.
        # This is just a placeholder to avoid errors if called in training mode.
        return 0.0
