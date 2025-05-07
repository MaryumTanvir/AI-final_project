import msgParser
import carState
import carControl
import csv
import os
from datetime import datetime
import keyboard

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        # for manual driving
        self.atsteer = 0.0
        self.ataccel = 0.0
        self.atbrake = 0.0

        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 110
        self.prev_rpm = None
        self.reverse_gear = False  # Flag for reverse mode
        self.init_data_logging()
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def init_data_logging(self):

        self.log_file = 'complete_driving_data.csv'

        if os.path.exists(self.log_file):
            self.log_headers_written = True
        else:
            self.log_headers_written = False
            
    def log_data(self):
        state_data = self.state.get_all_state_data()
        control_data = {
            'steer': self.control.getSteer(),
            'gear': self.control.getGear(),
            'accel': self.control.getAccel(),
            'brake': self.control.getBrake(),
            'clutch': self.control.getClutch(),
            'focus': self.control.getFocus(),
            'meta': self.control.getMeta()
        }
        
        combined_data = {**state_data, **control_data}

        if not self.log_headers_written:
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=combined_data.keys())
                writer.writeheader()
            self.log_headers_written = True
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=combined_data.keys())
            writer.writerow(combined_data)
    
    def reset_player_controls(self):
        '''Reset player control variables to default values'''
        self.atsteer = 0.0
        self.ataccel = 0.0
        self.atbrake = 0.0

    def drive(self, msg):
        # Reset player control variables when using autonomous drive
        self.reset_player_controls()
        
        self.state.setFromMsg(msg)
        
        self.steer()
        self.gear()
        self.speed()
        
        self.log_data()
        
        return self.control.toMsg()
    

    def drive_manual(self, msg):
        # Update car state using the server's response
        self.state.setFromMsg(msg)

        self.handle_keyboard_input()
        self.gear()

        self.log_data()
        return self.control.toMsg()

    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        # If reverse mode is active, force reverse gear (-1)
        if self.reverse_gear:
            self.control.setGear(-1)
            return
        
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        speed = self.state.getSpeedX()
        
        if self.prev_rpm is None:
            up = True
        else:
            up = (self.prev_rpm - rpm) < 0
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000 and gear > 1:
            gear -= 1  # Downshift if RPM is too low and not already in 1st gear
        
        # If reverse mode has been canceled and gear is still reverse, force it to 1
        if not self.reverse_gear and gear < 1:
            gear = 1
        
        self.control.setGear(gear)
        self.prev_rpm = rpm
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
            
    
    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass
        
    # Player control variables
    atsteer = 0.0
    ataccel = 0.0
    atbrake = 0.0

    def handle_keyboard_input(self):
        speed = self.state.getSpeedX()
        gear = self.state.getGear()
        
        # When down arrow is pressed:
        if keyboard.is_pressed('down'):
            # If car is stopped (or nearly stopped) and in 1st gear, activate reverse mode.
            if abs(speed) < 0.1 and gear == 1:
                self.reverse_gear = True
            if self.reverse_gear:
                # In reverse mode, if the up arrow is pressed, we apply braking to decelerate smoothly
                if keyboard.is_pressed('up'):
                    # Apply brake until speed is nearly zero before canceling reverse mode
                    if abs(speed) < 0.5:
                        self.reverse_gear = False
                        self.ataccel = 0.0
                    else:
                        self.ataccel = 0.0
                        self.atbrake = min(self.atbrake + 0.1, 1.0)
                else:
                    # Otherwise, use down arrow to accelerate in reverse
                    self.ataccel = min(self.ataccel + 0.1, 1.0)
                    self.atbrake = 0.0
            else:
                # If not in reverse mode, down arrow applies normal braking
                self.ataccel = 0.0
                self.atbrake = min(self.atbrake + 0.1, 1.0)
        elif keyboard.is_pressed('up'):
            # If up arrow is pressed and not in reverse mode, accelerate forward
            if self.reverse_gear:
                # If still in reverse mode, apply braking to stop the reverse motion gradually
                if abs(speed) < 0.5:
                    self.reverse_gear = False
                    self.ataccel = 0.0
                else:
                    self.ataccel = 0.0
                    self.atbrake = min(self.atbrake + 0.1, 1.0)
            else:
                self.ataccel = min(self.ataccel + 0.1, 1.0)
                self.atbrake = 0.0
        else:
            self.ataccel = max(self.ataccel - 0.1, 0.0)
            self.atbrake = 0.0

        if keyboard.is_pressed('right'):
            self.atsteer = max(self.atsteer - 0.1, -1.0)
        elif keyboard.is_pressed('left'):
            self.atsteer = min(self.atsteer + 0.1, 1.0)
        else:
            self.atsteer = 0.0

        self.control.setAccel(self.ataccel)
        self.control.setBrake(self.atbrake)
        self.control.setSteer(self.atsteer)


# ORIGINALLLL HEREE
#HEHE