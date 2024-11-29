import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

# Rotation parameters
ANGLE_TOLERANCE = 3 # degrees
KP_ROTATION = 90
MAX_ROTATION_SPEED = 150

# Translation parameters
DISTANCE_TOLERANCE = 50 # mm
KP_TRANSLATION = 0.7 
MIN_TRANSLATION_SPEED = 100
MAX_TRANSLATION_SPEED = 200

# Obstacle avoidance parameters
OBSTACLES_MAX_ITER = 7
OBSTACLES_SPEED = 100
SCALE_SENSOR = 100
WEIGHT_LEFT = [ 5,  8, -10,  -8, -5]
WEIGHT_RIGHT = [-5, -8, -10, 8,  5]

class LocalNav():
    
    def __init__(self):
        self.current_checkpoint = 0
        self.thymio_pos = None
        self.thymio_orientation = 0
        self.obstacles_iter = 0
        
        print("LocalNav initialized correctly.")
        
    #----------------------------------------#
    
    def _detect_obstacles(self,sensor_data):
        return sum(sensor_data) > 0
    
    def _calculate_angle_to_target(self, current_pos, target_pos):
        delta = target_pos - current_pos
        target_angle = np.arctan2(delta[1], delta[0])
        return target_angle
    
    def _calculate_motion_commands(self, angle_diff, distance):
        
        # Calculate rotation speed based on angle difference
        # The bigger the difference, the slower the forward motion
        if abs(angle_diff) < np.deg2rad(ANGLE_TOLERANCE):
            rotation_speed = 0
        else: 
            rotation_speed = np.clip(KP_ROTATION*angle_diff, -MAX_ROTATION_SPEED, MAX_ROTATION_SPEED)
        
        # Calculate forward speed based on both distance and angle difference
        # Reduce forward speed when angle difference is large or distance is small
        angle_factor = np.cos(angle_diff) 
        distance_factor = np.clip(KP_TRANSLATION*distance, MIN_TRANSLATION_SPEED, 
                                    MAX_TRANSLATION_SPEED)
        
        forward_speed = distance_factor * max(0, angle_factor)
        
        return int(forward_speed), int(rotation_speed)
    
    def _avoid_obstacles(self, sensor_data):
        left_speed = 0
        right_speed = 0

        # Updates speed based on sensor data and their corresponding weights
        for i in range(len(sensor_data) - 2):
            left_speed = left_speed + sensor_data[i] * WEIGHT_LEFT[i] / SCALE_SENSOR
            right_speed = right_speed + sensor_data[i] * WEIGHT_RIGHT[i] / SCALE_SENSOR

        command = {
                'action': 'avoid_obstacles',
                'left_speed': int(left_speed + OBSTACLES_SPEED),
                'right_speed': int(right_speed + OBSTACLES_SPEED)
            }
        return command, False
    
    def _trajectory_following(self, trajectory_points):
        
        # While there are still points in the trajectory, keep navigating
        while self.current_checkpoint < len(trajectory_points):
            target_pos = trajectory_points[self.current_checkpoint]

            # Calculate the distance and the angle to the target
            target_angle = self._calculate_angle_to_target(self.thymio_pos, target_pos)
            distance = utils.distance(self.thymio_pos, target_pos)
            # Calculate angle difference and normalize angle difference to [-pi, pi]
            angle_diff = (target_angle - self.thymio_orientation + np.pi) % (2 * np.pi) - np.pi

            # Check if we've reached the target
            if distance < DISTANCE_TOLERANCE:
                self.current_checkpoint += 1
                if self.current_checkpoint >= len(trajectory_points):
                    command = {
                        'action': 'stop',
                        'message': 'Trajectory completed'
                    }
                    self.current_checkpoint = 0
                    return command, True
            
            # Calculate combined motion commands
            forward_speed, rotation_speed = self._calculate_motion_commands(angle_diff, distance)
            # Print status
            print(f"\nCurrent status:")
            print(f"Distance to target: {distance:.2f}mm")
            print(f"Angle difference: {np.degrees(angle_diff):.2f}°")
            print(f"Forward speed: {forward_speed:.2f}")
            print(f"Rotation speed: {rotation_speed:.2f}")
            print(f"Current checkpoint: {self.current_checkpoint}")
                
            left_speed = forward_speed + rotation_speed
            right_speed = forward_speed -rotation_speed

            command = {
                'action': 'follow_path',
                'left_speed': left_speed,
                'right_speed': right_speed,
                'current_checkpoint': self.current_checkpoint
            }
            return command, False
    
    #----------------------------------------#
        
    def get_command(self, trajectory_points, thymio_pos, sensor_data):
        
        # Update position and orientation
        self.thymio_pos = np.array(thymio_pos[:2])
        self.thymio_orientation = thymio_pos[2]
        
        if self._detect_obstacles(sensor_data):
            # If obstacles are detected, start avoiding them
            self.obstacles_iter = OBSTACLES_MAX_ITER
            command, goal_reached =self._avoid_obstacles(sensor_data) 
        else:
            # If no obstacles are detected, follow the trajectory
            self.obstacles_iter = max(self.obstacles_iter - 1, 0)
            command, goal_reached = (
                self._trajectory_following(trajectory_points) if self.obstacles_iter == 0
                else self._avoid_obstacles(sensor_data)
            )
        return command, goal_reached
    
