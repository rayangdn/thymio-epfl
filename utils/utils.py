import numpy as np
import cv2

def add_label(image, text):
    # Add black background for text
    cv2.rectangle(image, (0, 0), (200, 30), (0, 0, 0), -1)
    # Add white text
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    return image

def mm_to_pixels(position_mm):
    # Convert from mm to pixels
    x_px = int(position_mm[0] * 1080/300)
    y_px = int(position_mm[1] * 1080/300)
    return np.array([x_px, y_px])

def simulate_robot_movement(position, orientation, command, dt=0.001):
    new_position = position.copy()
    new_orientation = orientation
    if command['action'] == 'move_and_rotate':
        # Update orientation
        new_orientation += command['rotation_speed'] * dt
        # Update position based on forward movement in the current orientation
        new_position[0] += command['forward_speed'] * np.cos(new_orientation) * dt  
        new_position[1] += command['forward_speed'] * np.sin(new_orientation) * dt
    
    return new_position, new_orientation