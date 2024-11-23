import os
import yaml
import numpy as np
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

class LocalNav():
    def __init__(self, angle_threshold, distance_threshold, scale_rotation_speed, scale_translation_speed, 
                    max_translation_speed, min_translation_speed, max_rotation_speed, obstacles_max_iter,
                    obstacles_speed, scale_sensor, weight_left, weight_right):

        self.current_checkpoint = 0
        self.thymio_pos = None
        self.thymio_orientation = 0
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold
        self.scale_rotation_speed = scale_rotation_speed
        self.max_rotation_speed = max_rotation_speed
        self.scale_translation_speed = scale_translation_speed
        self.max_translation_speed = max_translation_speed
        self.min_translation_speed = min_translation_speed
        self.obstacles_iter = 0
        self.obstacles_max_iter = obstacles_max_iter
        self.obstacles_speed = obstacles_speed
        self.scale_sensor = scale_sensor
        self.weight_left = weight_left
        self.weight_right = weight_right
        
        print("LocalNav Initialized")

    def _detect_obstacles(self,sensor_data):
        return sum(sensor_data) > 0
    
    def _calculate_angle_to_target(self, current_pos, target_pos):
        delta = target_pos - current_pos
        target_angle = np.arctan2(delta[1], delta[0])
        return target_angle
    
    def _calculate_motion_commands(self, angle_diff, distance):
        
        # Calculate rotation speed based on angle difference
        # The bigger the difference, the slower the forward motion
        if abs(angle_diff) < self.angle_threshold:
            rotation_speed = 0
        else: 
            rotation_speed = np.clip(self.scale_rotation_speed*angle_diff, -self.max_rotation_speed, self.max_rotation_speed)
        
        # Calculate forward speed based on both distance and angle difference
        # Reduce forward speed when angle difference is large
        angle_factor = np.cos(angle_diff) 
        distance_factor = np.clip(self.scale_translation_speed*distance, self.min_translation_speed, 
                                    self.max_translation_speed)
        
        forward_speed = distance_factor * max(0, angle_factor)
        
        return int(forward_speed), int(rotation_speed)
    
    def _avoid_obstacles(self, sensor_data):
        left_speed = 0
        right_speed = 0

        # Updates speed based on sensor data and their corresponding weights
        for i in range(len(sensor_data) - 2):
            left_speed = left_speed + sensor_data[i] * self.weight_left[i] / self.scale_sensor
            right_speed = right_speed + sensor_data[i] * self.weight_right[i] / self.scale_sensor

        command = {
                'action': 'avoid_obstacles',
                'left_speed': int(left_speed + self.obstacles_speed),
                'right_speed': int(right_speed + self.obstacles_speed)
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
            if distance < self.distance_threshold:
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
            # # Print status
            # print(f"\nCurrent status:")
            # print(f"Distance to target: {distance:.2f}mm")
            # print(f"Angle difference: {np.degrees(angle_diff):.2f}°")
            # print(f"Forward speed: {forward_speed:.2f}")
            # print(f"Rotation speed: {rotation_speed:.2f}")
            # print(f"Current checkpoint: {self.current_checkpoint}")
                
            left_speed = forward_speed + rotation_speed
            right_speed = forward_speed - rotation_speed

            command = {
                'action': 'follow_path',
                'left_speed': left_speed,
                'right_speed': right_speed,
                'current_checkpoint': self.current_checkpoint
            }
            return command, False

    def get_command(self, trajectory_points, thymio_pos, thymio_orientation, sensor_data):
        
        # Update position and orientation
        self.thymio_pos = np.array(thymio_pos)
        self.thymio_orientation = thymio_orientation
        
        if self._detect_obstacles(sensor_data):
            self.obstacles_iter = min(self.obstacles_iter + 1, self.obstacles_max_iter)
            command, goal_reached = (
                self._avoid_obstacles(sensor_data) if self.obstacles_iter >= self.obstacles_max_iter
                else self._trajectory_following(trajectory_points)
            )
        else:
            self.obstacles_iter = max(self.obstacles_iter - 1, 0)
            command, goal_reached = (
                self._trajectory_following(trajectory_points) if self.obstacles_iter == 0
                else self._avoid_obstacles(sensor_data)
            )
        return command, goal_reached
    
