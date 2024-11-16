from vision import Vision
from interface import Interface
from global_nav import GlobalNav
import numpy as np

def main():
    # Initialize Vision and Interface systems
    vision = Vision()
    interface = Interface()
    global_nav = GlobalNav()
    
    # Try connecting to the webcam
    print(f"Trying to connect to device {vision.device_id}...")
    if not vision.connect_webcam():
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print(f"Successfully connected to device {vision.device_id}")
    
    try:
        while interface.is_window_open():
            frame, process_frame, thymio_goal_position  = vision.get_frame()
            trajectory_frame = None
            if process_frame is not None:
                thymio_goal_positions = {
                    "thymio": np.array([30, 30]),
                    "goal": np.array([900, 550])
                    }
                trajectory_frame = global_nav.get_trajectory(process_frame, thymio_goal_positions)
            interface.update_display(frame, process_frame, trajectory_frame)
            
    finally:
        vision.cleanup_webcam()
        interface.cleanup()

if __name__ == "__main__":
    main()