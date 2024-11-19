import os
import yaml
import numpy as np
import time

class LocalNav():
        def __init__(self, angle_threshold, distance_threshold, scale_rotation_speed, scale_translation_speed, 
                     max_translation_speed, min_translation_speed, max_rotation_speed):

            self.current_checkpoint = 0
            self.thymio_position = None
            self.thymio_orientation = 0
            self.angle_threshold = angle_threshold
            self.distance_threshold = distance_threshold
            self.scale_rotation_speed = scale_rotation_speed
            self.max_rotation_speed = max_rotation_speed
            self.scale_translation_speed = scale_translation_speed
            self.max_translation_speed = max_translation_speed
            self.min_translation_speed = min_translation_speed
           
            print("LocalNav Initialized")

        def _calculate_angle_to_target(self, current_pos, target_pos):
            delta = target_pos - current_pos
            target_angle = np.arctan2(delta[1], delta[0])
            return target_angle
        
        def _calculate_distance_to_target(self, current_pos, target_pos):
            return np.linalg.norm(target_pos - current_pos)
        
        def _calculate_motion_commands(self, angle_diff, distance):
            
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
        
        def _trajectory_following(self, trajectory_points):
            
            # While there are still points in the trajectory, keep navigating
            while self.current_checkpoint < len(trajectory_points):
                target_pos = trajectory_points[self.current_checkpoint]

                # Calculate the distance and the angle to the target
                target_angle = self._calculate_angle_to_target(self.thymio_position, target_pos)
                distance = self._calculate_distance_to_target(self.thymio_position, target_pos)
                # Calculate angle difference and normalize angle difference to [-pi, pi]
                angle_diff = (target_angle - self.thymio_orientation + np.pi) % (2 * np.pi) - np.pi

                # Check if we've reached the target
                if distance < self.distance_threshold:
                    self.current_checkpoint += 1
                    if self.current_checkpoint >= len(trajectory_points):
                        print("\nTrajectory completed!")
                        command = {
                            'action': 'stop',
                            'message': 'Trajectory completed'
                        }
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
                left_speed = forward_speed
                right_speed = forward_speed
                if angle_diff >= 0:
                    left_speed  += rotation_speed
                    right_speed -= rotation_speed
                else:
                    left_speed  -= rotation_speed
                    right_speed += rotation_speed
                command = {
                    'action': 'move_and_rotate',
                    'left_speed': left_speed,
                    'right_speed': right_speed,
                    'current_checkpoint': self.current_checkpoint
                }
                return command, False

        def navigate(self, trajectory_points, thymio_position, thymio_orientation):
            
            # Update position and orientation
            self.thymio_position = np.array(thymio_position)
            self.thymio_orientation = thymio_orientation
            command, goal_reached = self._trajectory_following(trajectory_points)

            return command, goal_reached
        
