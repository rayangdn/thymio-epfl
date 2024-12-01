# Local navigation system for autonomous robot path following and obstacle avoidance
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

# Angular control parameters
ANGLE_TOLERANCE = 5 # Maximum acceptable angle error in degrees
KP_ROTATION = 90   # Proportional gain for rotation control
MAX_ROTATION_SPEED = 150 # Maximum rotation speed in motor units

# Linear motion control parameters
DISTANCE_TOLERANCE = 50 # Maximum acceptable position error in mm
KP_TRANSLATION = 0.7   # Proportional gain for translation control
MIN_TRANSLATION_SPEED = 100 # Minimum forward speed in motor units
MAX_TRANSLATION_SPEED = 200 # Maximum forward speed in motor units

# Obstacle avoidance parameters
OBSTACLES_MAX_ITER = 1 # Number of iterations to perform avoidance after detection
OBSTACLES_SPEED = 100  # Base speed during obstacle avoidance
SCALE_SENSOR = 200    # Scaling factor for sensor readings
# Sensor weights for left and right motor adjustments
WEIGHT_LEFT = [ 5,  8, -10,  -8, -5]  # Positive weights favor right turn
WEIGHT_RIGHT = [-5, -8, -10, 8,  5]   # Positive weights favor left turn

class LocalNav():
    def __init__(self):
        # Current waypoint index in trajectory
        self.current_checkpoint = 0
        # Robot position in global coordinates [x, y]
        self.thymio_pos = None
        # Robot orientation in radians
        self.thymio_orientation = 0
        # Counter for obstacle avoidance iterations
        self.obstacles_iter = 0
        # Flag to request trajectory recomputation
        self.needs_recompute = False
        
        print("LocalNav initialized correctly.")
        
    def _detect_obstacles(self,sensor_data):
        # Return True if any proximity sensor detects an obstacle
        return sum(sensor_data) > 0
    
    def _calculate_angle_to_target(self, current_pos, target_pos):
        # Calculate angle between robot position and target
        delta = target_pos - current_pos
        target_angle = np.arctan2(delta[1], delta[0])
        return target_angle
    
    def _calculate_motion_commands(self, angle_diff, distance):
        # Calculate rotation speed proportional to angle error
        if abs(angle_diff) < np.deg2rad(ANGLE_TOLERANCE):
            rotation_speed = 0
        else: 
            rotation_speed = np.clip(KP_ROTATION*angle_diff, -MAX_ROTATION_SPEED, MAX_ROTATION_SPEED)
        
        # Forward speed decreases with angle error and increases with distance
        angle_factor = np.cos(angle_diff) 
        distance_factor = np.clip(KP_TRANSLATION*distance, MIN_TRANSLATION_SPEED, 
                                    MAX_TRANSLATION_SPEED)
        
        forward_speed = distance_factor * max(0, angle_factor)
        
        return int(forward_speed), int(rotation_speed)
    
    def _avoid_obstacles(self, sensor_data):
        # Initialize speeds with base obstacle avoidance speed
        left_speed = OBSTACLES_SPEED
        right_speed = OBSTACLES_SPEED

        # Updates speed based on sensor data and their corresponding weights
        for i in range(len(sensor_data)):
            left_speed += sensor_data[i] * WEIGHT_LEFT[i] / SCALE_SENSOR
            right_speed += sensor_data[i] * WEIGHT_RIGHT[i] / SCALE_SENSOR

        command = {
                'action': 'avoid_obstacles',
                'left_speed': int(left_speed),
                'right_speed': int(right_speed)
            }
        return command, False
    
    def _trajectory_following(self, trajectory_points):
        # Process waypoints until trajectory completion
        while self.current_checkpoint < len(trajectory_points):
            target_pos = trajectory_points[self.current_checkpoint]

            # Calculate error metrics to target
            target_angle = self._calculate_angle_to_target(self.thymio_pos, target_pos)
            distance = utils.distance(self.thymio_pos, target_pos)
            
            # Normalize angle difference to [-π, π]
            angle_diff = (target_angle - self.thymio_orientation + np.pi) % (2 * np.pi) - np.pi

            # Check waypoint completion
            if distance < DISTANCE_TOLERANCE:
                self.current_checkpoint += 1
                if self.current_checkpoint >= len(trajectory_points):
                    command = {
                        'action': 'stop',
                        'message': 'Trajectory completed'
                    }
                    self.current_checkpoint = 0
                    return command, True
            
            # Convert angle and distance errors to motor commands
            forward_speed, rotation_speed = self._calculate_motion_commands(angle_diff, distance)
            
            # Convert to differential drive commands
            left_speed = forward_speed + rotation_speed
            right_speed = forward_speed - rotation_speed

            command = {
                'action': 'follow_path',
                'left_speed': left_speed,
                'right_speed': right_speed,
                'current_checkpoint': self.current_checkpoint
            }
            return command, False
        
    def get_command(self, trajectory_points, thymio_pos, sensor_data):
        # Update robot state from sensor data
        self.thymio_pos = np.array(thymio_pos[:2])
        self.thymio_orientation = thymio_pos[2]
        
        # Extract front proximity sensor readings
        sensor_data = sensor_data[:5]
        
        if self._detect_obstacles(sensor_data):
            # Activate obstacle avoidance behavior
            self.obstacles_iter = OBSTACLES_MAX_ITER
            self.needs_recompute = True
            command, goal_reached = self._avoid_obstacles(sensor_data) 
        else:
            # Decrement obstacle avoidance counter
            self.obstacles_iter = max(self.obstacles_iter - 1, 0)
            
            if self.obstacles_iter == 0:
                if self.needs_recompute:
                    # Request new trajectory from current position
                    command = {
                        'action': 'recompute_trajectory',
                        'current_pos': self.thymio_pos.tolist(),
                        'current_orientation': self.thymio_orientation
                    }
                    self.needs_recompute = False
                    return command, False
                else:
                    # Resume normal trajectory following
                    command, goal_reached = self._trajectory_following(trajectory_points)
            else:
                # Continue obstacle avoidance
                command, goal_reached = self._avoid_obstacles(sensor_data)
                
        return command, goal_reached