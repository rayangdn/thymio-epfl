import os
import yaml
import numpy as np
import time

class LocalNav():
        def __init__(self):
            # Load config
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)             
            config_path = os.path.join(parent_dir, 'config', 'config.yaml')
            
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)

            self.current_checkpoint = 1
            self.thymio_position = None
            self.thymio_orientation = 0
            self.angle_threshold = np.radians(self.config['controller']['angle_tolerance'])
            self.distance_threshold = self.config['controller']['distance_tolerance']
            self.scale_rotation_speed = self.config['controller']['scale_rotation_speed']
            self.scale_translation_speed = self.config['controller']['scale_translation_speed']
            self.max_translation_speed = self.config['controller']['max_translation_speed']
            self.min_translation_speed = self.config['controller']['min_translation_speed']
            self.max_rotation_speed = self.config['controller']['max_rotation_speed']
            
            print("LocalNav Initialized")

        def calculate_angle_to_target(self, current_pos, target_pos):
            delta = target_pos - current_pos
            target_angle = np.arctan2(delta[1], delta[0])
            return target_angle
        
        def calculate_distance_to_target(self, current_pos, target_pos):
            return np.linalg.norm(target_pos - current_pos)
        
        def calculate_motion_commands(self, angle_diff, distance):
            
            # Calculate rotation speed based on angle difference
            # The bigger the difference, the slower the forward motion
            rotation_speed = np.clip(self.scale_rotation_speed*angle_diff, -self.max_rotation_speed, self.max_rotation_speed)
            
            # Calculate forward speed based on both distance and angle difference
            # Reduce forward speed when angle difference is large
            angle_factor = np.cos(angle_diff)  # Will be 1 when aligned, less when not
            distance_factor = np.clip(self.scale_translation_speed*distance, self.min_translation_speed, 
                                      self.max_translation_speed)
            
            forward_speed = distance_factor * max(0, angle_factor)
            
            return int(forward_speed), int(rotation_speed)
        
        def trajectory_following(self, trajectory_points):
            
            # While there are still points in the trajectory, keep navigating
            while self.current_checkpoint < len(trajectory_points):
                target_pos = trajectory_points[self.current_checkpoint]

                # Calculate the distance and the angle to the target
                target_angle = self.calculate_angle_to_target(self.thymio_position, target_pos)
                distance = self.calculate_distance_to_target(self.thymio_position, target_pos)
                # Calculate angle difference and normalize angle difference to [-pi, pi]
                angle_diff = (target_angle - self.thymio_orientation + np.pi) % (2 * np.pi) - np.pi

                # Check if we've reached the target
                if distance < self.distance_threshold:
                    print(f"\nReached checkpoint {self.current_checkpoint}")
                    self.current_checkpoint += 1
                    if self.current_checkpoint >= len(trajectory_points):
                        print("\nTrajectory completed!")
                        command = {
                            'action': 'stop',
                            'message': 'Trajectory completed'
                        }
                        return command, True
                
                # Calculate combined motion commands
                forward_speed, rotation_speed = self.calculate_motion_commands(angle_diff, distance)
                # # Print status
                # print(f"\nCurrent status:")
                # print(f"Distance to target: {distance:.2f}mm")
                # print(f"Angle difference: {np.degrees(angle_diff):.2f}°")
                # print(f"Forward speed: {forward_speed:.2f}")
                # print(f"Rotation speed: {rotation_speed:.2f}")
                # print(f"Current checkpoint: {self.current_checkpoint}")
                
                command = {
                    'action': 'move_and_rotate',
                    'forward_speed': forward_speed,
                    'rotation_speed': rotation_speed,
                    'current_checkpoint': self.current_checkpoint
                }
                return command, False

        def navigate(self, trajectory_points, thymio_position, thymio_orientation):
            # Update position and orientation
            self.thymio_position = np.array(thymio_position)
            self.thymio_orientation = thymio_orientation
            command, goal_reached = self.trajectory_following(trajectory_points)

            return command, goal_reached
            