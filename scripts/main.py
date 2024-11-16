from vision import Vision
from interface import Interface
from global_nav import GlobalNav
from local_nav import LocalNav
import numpy as np
from utils.utils import simulate_robot_movement

def main():
    # Initialize Vision and Interface systems
    vision = Vision()
    interface = Interface()
    global_nav = GlobalNav()
    local_nav = LocalNav()
    
    # Try connecting to the webcam
    print(f"Trying to connect to device {vision.device_id}...")
    if not vision.connect_webcam():
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print(f"Successfully connected to device {vision.device_id}")
    
    try:
        while interface.is_window_open():
            frame, process_frame, _ = vision.get_frame()
            thymio_goal_positions = {
                "thymio": np.array([30, 30]),
                "goal": np.array([800, 600])
            }
            goal_reached = False
            thymio_orientation = 0  # rads
            trajectory_frame = None
            if process_frame is not None:
                trajectory_frame, trajectory_points_mm, thymio_position_mm = global_nav.get_trajectory(process_frame, thymio_goal_positions) 
                current_position = thymio_position_mm['thymio'].copy()
                current_orientation = thymio_orientation
                command, goal_reached = local_nav.navigate(trajectory_points_mm, current_position, current_orientation)
                
                while not goal_reached:
                    frame, process_frame, _ = vision.get_frame()
                    
                    # Simulate movement
                    new_position, new_orientation = simulate_robot_movement(
                        current_position, current_orientation, command
                    )
                    
                    # Update current position and orientation
                    current_position = new_position
                    current_orientation = new_orientation
                    
                    # Get next command using updated position and orientation
                    command, goal_reached = local_nav.navigate(trajectory_points_mm, current_position, current_orientation)
                    
                    # Update display with current position
                    interface.update_display(frame, process_frame, trajectory_frame, current_position, current_orientation)
            
            interface.update_display(frame, process_frame, trajectory_frame)
    finally:
        vision.cleanup_webcam()
        interface.cleanup()

if __name__ == "__main__":
    main()