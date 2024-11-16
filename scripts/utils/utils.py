import numpy as np

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