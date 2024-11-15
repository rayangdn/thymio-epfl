from vision import Vision
from interface import Interface
from global_nav import GlobalNav
import cv2

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
            frame, process_frame, _, _, _, _ = vision.get_frame()
            trajectory_frame = None
            obstacle_corners = None
            if process_frame is not None:
                cv2.imwrite('frame.jpg', process_frame)
                trajectory_frame, _ = global_nav.detect_contours(process_frame)
            interface.update_display(frame, process_frame, trajectory_frame)
            
    finally:
        vision.cleanup_webcam()
        interface.cleanup()

if __name__ == "__main__":
    main()