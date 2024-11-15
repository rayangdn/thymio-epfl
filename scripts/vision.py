import os
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt

class Interface:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(2, 2)
        gs.update(left=0.01, right=0.99, bottom=0.05, top=0.95, wspace=0.02, hspace=0.15)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        
        self.fig.canvas.manager.set_window_title('Vision Interface')
        
        self.plots = [None] * 4

    def update_display(self, original_frame, process_frame):
        frames = [original_frame, process_frame, original_frame, original_frame]
        titles = ['Webcam View', 'Processing View', 'Processing View', 'Result View']
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        
        for i, (frame, title, ax) in enumerate(zip(frames, titles, axes)):
            if frame is None:
                continue
            if self.plots[i] is None:
                self.plots[i] = ax.imshow(frame)
                ax.set_title(title, fontsize=12, pad=10)
                ax.axis('off')
            else:
                self.plots[i].set_data(frame)
        
        plt.draw()
        plt.pause(0.001)

    def is_window_open(self):
        return bool(plt.get_fignums())

    def cleanup(self):
        plt.close('all')
        
class Vision:
    
    # Class-level constant for corner mapping
    # USe order bf, tl, tr, br
    MAPPING = {
        0: "bottom_left",
        1: "bottom_right",
        2: "top_left",
        3: "top_right",
        4: "thymio",
        5: "goal"
    }
    
    def __init__(self):
        print("Initializing Vision")
        
        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)             
        config_path = os.path.join(parent_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Initialize camera
        self.cap = None
        self.device_id = self.config['webcam']['device_id']
        self.camera_matrix = np.array(self.config['webcam']['matrix'])
        self.dist_coeffs = np.array(self.config['webcam']['distortion'])
        self.resolution = self.config['webcam']['resolution']
        
        # Perspective transform matrix
        self.perspective_matrix = None
        
        # Initialize World
        self.world_width = self.config['world']['width']
        self.world_height = self.config['world']['height']
        
        # Initialize ArUco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Initalize interface
        self.interface = Interface()

    def connect_webcam(self):
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True
    
    def cleanup_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.interface.cleanup()
        
    def process_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Create dictionaries to store corner, thymio and goal positions
        corner_positions = {
            "bottom_left": None, 
            "bottom_right": None,
            "top_left": None,
            "top_right": None
        }
        thymio_goal_positions = {
            "thymio": None,
            "goal": None
        }
        
        found_corners = False
        found_thymio_goal = False
        
        if ids is None:
            return frame, corner_positions, thymio_goal_positions, found_corners, found_thymio_goal
        
        # Draw markers
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            
            # Get center of marker
            center = np.mean(c, axis=0).astype(int)
            
            # Get name from mapping
            name = self.MAPPING[marker_id]
            
            # Store corner positions
            if marker_id in [0, 1, 2, 3]:
                corner_positions[name] = (int(c[0][0]), int(c[0][1])) 
            
            # Store thymio and goal positions
            if marker_id in [4, 5]:
                thymio_goal_positions[name] = (int(center[0][0]), int(center[0][1]))
            
            # Draw background rectangle for better text visibility
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, 
                        (center[0] - 5, center[1] - text_size[1] - 5),
                        (center[0] + text_size[0] + 5, center[1] + 5),
                        (0, 0, 0), -1)
            
            # Draw marker name 
            cv2.putText(frame, name, 
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)
            
            # Draw lines between corners if all are detected to define the world
            if len(corner_positions) == 4 and all(pos is not None for pos in corner_positions.values()):
                corners_order = ["bottom_left", "top_left", "top_right", "bottom_right"]
                found_corners = True
                for i in range(4):
                    start = corner_positions[corners_order[i]]
                    end = corner_positions[corners_order[(i + 1) % 4]]
                    frame = cv2.line(frame, start, end, (0, 255, 0), 2)
            if len(thymio_goal_positions) == 2 and all(pos is not None for pos in thymio_goal_positions.values()):
                found_thymio_goal = True
                
        return frame, corner_positions, thymio_goal_positions, found_corners, found_thymio_goal
    
    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        
        # Undistort frame
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Crop the frame
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        
        return frame
    
    def compute_perspective_transform(self, source_points, frame):
        
        scale_factor = self.resolution[1] / self.world_width
        dest_width = self.world_width * scale_factor
        dest_height = self.world_height * scale_factor
        # Define destination points
        dest_points = np.float32([
            [0, dest_height],  # bottom-left
            [dest_width, dest_height],  # bottom-right
            [0, 0],  #top-left
            [dest_width, 0] #top-right  
        ])

        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)

        return np.array([dest_width, dest_height]).astype(int)

    def get_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame.")
                return None
            
            # Undistort the frame
            frame = self.undistort_frame(frame)
            
            # Detect markers
            frame, corner_positions, thymio_goal_position, found_corners, found_thymio_goal = self.process_aruco_markers(frame)
            
            process_frame = None
            roi = None
            
            # Compute perspective transform if we have all corner positions
            if found_corners:
                source_points = np.array(list(corner_positions.values()), dtype=np.float32)
                roi = self.compute_perspective_transform(source_points, frame)
                if roi is not None:
                    # Get the top-down view of the map
                    process_frame = cv2.warpPerspective(frame, self.perspective_matrix, roi)
                    process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite('map_view.png', process_frame)
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.interface.update_display(frame, process_frame)
            return frame
        return None
            
        
def main():
    # Initialize Vision system with smaller display size
    vision = Vision()  
    
    # Try connecting to the webcam
    print(f"Trying to connect to device {vision.device_id}...")
    if not vision.connect_webcam():
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print(f"Successfully connected to device {vision.device_id}")
    
    try:
        while True:
            frame = vision.get_frame()
            if frame is None:
                break
            
            if not vision.interface.is_window_open():
                break
        
    finally:
        vision.cleanup_webcam()

if __name__ == "__main__":
    main() 