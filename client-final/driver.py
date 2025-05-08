import os
import numpy as np
import joblib
import csv
import tensorflow as tf
from tensorflow.keras import layers, models

import msgParser
import carState
import carControl

# Constants
LOG_FILE = "complete_driving_data.csv"
BC_MODEL = "bc_model.keras"
BC_SCALER = "bc_scaler.gz"
BC_OUTPUT_SCALER = "bc_output_scaler.gz"

# Driver Class
class Driver:
    def __init__(self, stage: int, train: bool = False):
        self.stage = stage
        self.train_mode = train
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        self.max_gear = 6

        # Load Behavior Cloning model and scalers
        self.use_bc = False
        self.bc_net = None
        self.bc_scaler = None
        self.bc_output_scaler = None
        if os.path.exists(BC_MODEL) and os.path.exists(BC_SCALER) and os.path.exists(BC_OUTPUT_SCALER):
            try:
                self.bc_net = tf.keras.models.load_model(BC_MODEL)
                self.bc_scaler = joblib.load(BC_SCALER)
                self.bc_output_scaler = joblib.load(BC_OUTPUT_SCALER)
                self.use_bc = True
                print("\u2713 Behavior-cloning model and scalers loaded")
            except Exception as e:
                print(f"Warning: Failed to load BC model or scalers: {e}")
                self.use_bc = False

        # State tracking
        self.start_steps = 0
        self.initial_dist_raced = None
        self.prev_steer = 0.0
        self.stuck_steps = 0

    def get_state_vec(self):
        track = self.state.track or [0.0] * 19
        opponents = self.state.opponents or [0.0] * 36
        wheelSpinVel = self.state.wheelSpinVel or [0.0] * 4
        state_vec = [
            self.state.angle or 0.0,
            self.state.trackPos or 0.0,
            self.state.getSpeedX() or 0.0,
            self.state.getSpeedY() or 0.0,
            self.state.getSpeedZ() or 0.0,
            self.state.getRpm() or 0.0,
            self.state.z or 0.0,
            *wheelSpinVel,
            *track,
            *opponents
        ]
        return np.array(state_vec, dtype=np.float32)

    def bc_action(self):
        if not self.use_bc:
            return 0.0, 0.0, 0.0, 0.0, 1, 0.0

        s_vec = self.get_state_vec()
        try:
            X = self.bc_scaler.transform(s_vec[None, :])
            prediction_scaled = self.bc_net.predict(X, verbose=0)[0]
            prediction = self.bc_output_scaler.inverse_transform([prediction_scaled])[0]
            steer, accel, brake, clutch, gear, meta = prediction

            steer = -float(np.clip(steer, -1, 1))
            accel = float(np.clip(accel, 0, 1))
            brake = float(np.clip(brake, 0, 1))
            clutch = float(np.clip(clutch, 0, 1))
            gear = int(round(max(1, min(self.max_gear, gear))))
            meta = float(np.clip(meta, 0, 1))

            steer = self.prev_steer * 0.95 + steer * 0.05
            self.prev_steer = steer

            if self.state.track and self.state.track[9] < 4.0:
                steer = 0.5 if (self.state.trackPos or 0) > 0 else -0.5
                accel = 0.3
                meta = 1.0

            return steer, accel, brake, clutch, gear, meta
        except Exception as e:
            print(f"Error in bc_action: {e}. Using default actions.")
            return 0.0, 0.0, 0.0, 0.0, 1, 0.0

    def init(self):
        angles = [-90 + i * 10 for i in range(19)]
        return self.parser.stringify({'init': angles})

    def drive(self, msg: str):
        self.state.setFromMsg(msg)

        if self.initial_dist_raced is None:
            self.initial_dist_raced = self.state.distRaced or 0.0

        dist_moved = (self.state.distRaced or 0.0) - self.initial_dist_raced
        if dist_moved < 1.0 and self.start_steps < 100:
            self.control.clutch = 1.0
            self.control.gear = 1
            self.start_steps += 1
        else:
            steer, accel, brake, clutch, gear, meta = self.bc_action()
            self.control.clutch = clutch
            self.control.gear = gear

            track_pos = self.state.trackPos or 0.0
            speed_x = self.state.getSpeedX() or 0.0
            if (abs(speed_x) < 1.0 and accel > 0.5 and abs(track_pos) > 0.8) or meta > 0.5:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0

            if self.stuck_steps > 20:
                accel = -0.5
                brake = 0.0
                steer = -0.5 if track_pos > 0 else 0.5
                self.control.gear = -1
                self.stuck_steps = 0

            self.control.steer = steer
            self.control.accel = max(accel, 0.0)
            self.control.brake = max(brake, 0.0)
            self.control.meta = int(meta)

        track_sensors = self.state.track or [0.0] * 19
        print(f"Step {self.start_steps}: distRaced={self.state.distRaced:.4f}, speedX={self.state.getSpeedX():.4f}, "
              f"steer={self.control.steer:+.3f}, accel={self.control.accel:.3f}, gear={self.control.gear}, clutch={self.control.clutch:.3f}, "
              f"trackPos={self.state.trackPos or 0.0:+.3f}, angle={self.state.angle or 0.0:+.3f}, "
              f"trackSensors={track_sensors[:5]}")

        log_row(self.state, self.control)
        return self.control.toMsg()

    def save_model(self):
        pass

    def onShutDown(self):
        pass

    def onRestart(self):
        self.start_steps = 0
        self.initial_dist_raced = None
        self.prev_steer = 0.0
        self.stuck_steps = 0

# Logging
def log_row(state_obj, control_obj):
    s = state_obj.get_all_state_data()
    a = {
        'steer': control_obj.steer,
        'gear': control_obj.gear,
        'accel': control_obj.accel,
        'brake': control_obj.brake,
        'clutch': control_obj.clutch,
        'meta': control_obj.meta
    }
    row = {**s, **a}
    file_exists = os.path.exists(LOG_FILE)
    file_empty = file_exists and os.path.getsize(LOG_FILE) == 0
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists or file_empty:
            writer.writeheader()
        writer.writerow(row)
