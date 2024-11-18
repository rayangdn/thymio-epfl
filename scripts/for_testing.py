import cv2
import matplotlib.pyplot as plt
import numpy as np

from global_nav import GlobalNav
from local_nav import LocalNav

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

def draw_robot(img, position, orientation, size=10):
    # Convert position from mm to pixels before drawing
    pos_px = mm_to_pixels(position)
    
    # Draw robot body
    cv2.circle(img, (pos_px[0], pos_px[1]), size, (255, 0, 255), -1)
    
    # Draw orientation line
    end_x = int(pos_px[0] + size * 1.5 * np.cos(orientation))
    end_y = int(pos_px[1] + size * 1.5 * np.sin(orientation))
    cv2.line(img, (pos_px[0], pos_px[1]), (end_x, end_y), (0, 0, 255), 2)
    
    return img

def main():
    global_nav = GlobalNav()
    local_nav = LocalNav()
    img_path = "frame.png"
    img = cv2.imread(img_path)
    
    thymio_goal_positions = {
        "thymio": np.array([30, 30]),
        "goal": np.array([900, 550])
    }
    
    thymio_orientation = 0  # rads
    goal_reached = False
    
    trajectory_img, trajectory_points_mm, thymio_position_mm = global_nav.get_trajectory(img, thymio_goal_positions)
    display_img = trajectory_img.copy()
    
    # Initialize position and orientation
    current_position = thymio_position_mm['thymio'].copy()
    current_orientation = thymio_orientation
    
    # Get initial command
    command, goal_reached = local_nav.navigate(trajectory_points_mm, current_position, current_orientation)
    
    plt.figure(figsize=(12, 8))
    
    while not goal_reached:
        # Simulate movement
        new_position, new_orientation = simulate_robot_movement(
            current_position, current_orientation, command
        )
        
        # Update current position and orientation
        current_position = new_position
        current_orientation = new_orientation
        
        # Get next command using updated position and orientation
        command, goal_reached = local_nav.navigate(trajectory_points_mm, current_position, current_orientation)
        display_img = trajectory_img.copy()
        display_img = draw_robot(display_img, current_position, current_orientation)
        
        plt.clf()
        plt.imshow(display_img)
        plt.pause(0.02)  # Add small delay for visualization
    
    # Show final state
    plt.imshow(display_img)
    plt.show()

if __name__ == "__main__":
    import time  # Add import if using time.sleep
    main()