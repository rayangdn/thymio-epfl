import os
import yaml
import cv2
import numpy as np
import cv2.aruco as aruco

class Vision:
    # Class-level constant for corner mapping
    CORNER_MAPPING = {
        0: "bottom_left",
        1: "bottom_right",
        2: "top_right",
        3: "top_left",
        4: "tyhmio",
        5: "goal"
    }

    def __init__(self):
        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)             
        config_path = os.path.join(parent_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize camera
        self.cap = None
        self.device_id = self.config['webcam']['device_id']
        
        # Initialize ArUco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def connect_webcam(self):
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True

    def detect_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def draw_markers(self, frame, corners, ids):
        if ids is not None:
            # Draw the detected markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw the corner name and ID for each marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                center = np.mean(c, axis=0).astype(int)
                
                # Get corner name from mapping, or use ID if not mapped
                corner_name = self.CORNER_MAPPING.get(marker_id, f"Unknown ID: {marker_id}")
                
                # Draw background rectangle for better text visibility
                text_size = cv2.getTextSize(corner_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, 
                            (center[0] - 5, center[1] - text_size[1] - 5),
                            (center[0] + text_size[0] + 5, center[1] + 5),
                            (0, 0, 0),
                            -1)
                
                # Draw corner name
                cv2.putText(frame, corner_name, 
                           (center[0], center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        return frame

    def print_corner_mappings(self):
        print("Detected corner mappings:")
        for marker_id, corner_name in self.CORNER_MAPPING.items():
            print(f"Marker ID {marker_id} -> {corner_name}")

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame.")
                return None
            
            corners, ids = self.detect_aruco_markers(frame)
            frame = self.draw_markers(frame, corners, ids)
            
            return frame
        return None

def main():
    # Initialize Vision system with smaller display size
    vision = Vision()  # Half of 1920x1080
    
    # Try connecting to the webcam
    print(f"Trying to connect to device {vision.device_id}...")
    if not vision.connect_webcam():
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print(f"Successfully connected to device {vision.device_id}")
    vision.print_corner_mappings()
    
    try:
        while True:
            # Get frame (already processed and resized)
            frame = vision.get_frame()
            if frame is None:
                break
            
            # Display the frame
            cv2.imshow('ArUco Corner Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        vision.cleanup()

if __name__ == "__main__":
    main()