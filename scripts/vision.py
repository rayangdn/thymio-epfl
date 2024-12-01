import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

# Camera parameters
CAM_MATRIX = np.array([[1484.14935, 0, 964.220337],
                       [0, 1490.50225, 539.531044],
                       [0, 0, 1]])

CAM_DISTORTION = np.array([0.154056967, -1.04865202, -0.000689137957, -0.00132248869, 1.48905183])
  
CAM_RESOLUTION = (1920, 1080) 

OBSTACLE_MIN_AREA = 500 # mm²

# Aruco definitions
# MAPPING = {
#     0: "bottom_left",
#     1: "bottom_right",
#     2: "top_left",
#     3: "top_right",
#     4: "thymio",
#     5: "goal"
# }
    
class Vision():
    
    def __init__(self, device_id):
        # Initialize the camera
        self.device_id = device_id 
        self.cap = None
        
        # Intialize perspective params
        self.perspective_matrix = None
        self.process_roi = None
        self.scale_factor = None
        
        # Initialize the aruco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        print("Vision initialized correctly.")
    
    #-------------------------------------------------#    
    
    def _undistort_frame(self, frame):
        h, w = frame.shape[:2]
        
        # Undistort the frame
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAM_MATRIX, CAM_DISTORTION, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, CAM_MATRIX, CAM_DISTORTION, None, newcameramtx)
        
        # Crop the frame
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        return frame
    
    def _get_original_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            
            # Undistort the frame based on the camera matrix and distortion coefficients
            frame = self._undistort_frame(frame)
            return frame
    
    def _detect_corner_markers(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Define the corner positions
        corner_positions = np.zeros((4, 2), dtype=int)  # 4 corners, each with x,y coordinates
        found_corners = False
        
        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Store corner positions
                if marker_id in [0, 1, 2, 3]:
                    corner_positions[marker_id] = np.array([c[0][0], c[0][1]])
                    
                # Check if all corners have been detected
                if np.all(corner_positions):  
                    found_corners = True
                    
        return frame, corner_positions, found_corners
    
    def _detect_thymio_marker(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Define thymio position
        thymio_pos = np.zeros(3, dtype=float)  # x, y, orientation
        found_thymio = False
        
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Get center of marker
                center = np.mean(c, axis=0).astype(int)
                
                # Store thymio position 
                if marker_id == 4:
                    found_thymio = True
                    thymio_pos[:2] = center # x, y
                    
                   # Calculate orientation using the front edge of the marker
                    front_edge = c[1] - c[0]  # Vector from corner 0 to corner 1
                                
                    # Calculate orientation
                    orientation = np.arctan2(front_edge[1], front_edge[0]) 
                    
                    # Normalize angle to [-pi, pi]
                    orientation = np.arctan2(np.sin(orientation), np.cos(orientation))   
                    thymio_pos[2] = orientation
                    
                    # Draw orientation arrow for visualization
                    start_point = (int(center[0]), int(center[1]))
                    end_point = (int(center[0] + 50 * np.cos(orientation)), 
                               int(center[1] + 50 * np.sin(orientation)))
                    
                    cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)
                    
                    # Draw a circle at the center for visualization
                    cv2.circle(frame, start_point, 3, (0, 0, 255), -1)
                    
                    # Draw a line from the center to the front edge for visualization
                    cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)
                    cv2.circle(frame, start_point, 3, (0, 0, 255), -1)
        
        return frame, thymio_pos, found_thymio

    def _detect_goal_marker(self, frame):
         # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Define goal position
        goal_pos = np.zeros(2, dtype=float)  # x, y
        found_goal = False
        
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Get center of marker
                center = np.mean(c, axis=0).astype(int)
                # Store thymio position 
                if marker_id == 5:
                    found_goal = True
                    goal_pos = center # x, y
                    
        return frame, goal_pos, found_goal
    
    def _compute_perspective_transform(self, source_points, world_width, 
                                       world_height):
        self.scale_factor = CAM_RESOLUTION[1] / world_width
        
        # define destination points
        dest_width = world_width * self.scale_factor 
        dest_height = world_height * self.scale_factor 
        
        # Define destination points
        dest_points = np.float32([
            [0, dest_height],  # bottom-left
            [dest_width, dest_height],  # bottom-right
            [0, 0],  #top-left
            [dest_width, 0] #top-right  
        ])
        
        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        self.process_roi = np.array([dest_width, dest_height]).astype(int)
    
    def _filter_close_corners(self, corners, min_distance=10):
        # If there are no corners, return an empty list
        if len(corners) == 0:
            return corners
        
        # Convert to numpy array
        corners = np.array(corners)
        
        # List to keep track of corners to keep
        filtered_corners = [corners[0]]
        
        for corner in corners[1:]:
            if all(utils.distance(corner, kept_corner) >= min_distance for kept_corner in filtered_corners):
                filtered_corners.append(corner)
        
        return np.array(filtered_corners)
        
    def _detect_obstacles(self, frame):
        # Store intermediate results
        visualization_steps = {}
        
        # Original image
        frame = frame.copy()
        visualization_steps['Original image'] = frame.copy()

        # Grayscale conversion
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Gaussian blur
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Thresholding
        _, threshold_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        visualization_steps['Threshold image'] = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)
        
        # Edge detection
        edges_frame = cv2.Canny(threshold_frame, 50, 150)
        visualization_steps['Edges image'] = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2RGB)

        # Find and draw contours
        contours, _ = cv2.findContours(edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Frame to be returned
        obstacles_frame = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)
        
        obstacles_corners = {}
        
        for i, contour in enumerate(contours):
            
            # Skip small contours
            if cv2.contourArea(contour) <= OBSTACLE_MIN_AREA:
                continue
            
            # Draw the contour
            cv2.drawContours(obstacles_frame, [contour], -1, (255, 0, 0), 2)
            
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter close corners
            corners = approx.reshape(-1, 2).astype(np.float32)
            filtered_corners = self._filter_close_corners(corners)
            
            # Draw the corners
            obstacle_name = f"obstacle{i+1}"
            obstacles_corners[obstacle_name] = np.array(filtered_corners)
            for corner in filtered_corners:
                cv2.circle(obstacles_frame, tuple(corner.astype(np.int32)), 5, (0, 0, 255), -1)
        
        visualization_steps['Obstacles image'] = obstacles_frame
        
        return obstacles_corners, obstacles_frame, visualization_steps
    
    #-------------------------------------------------#
    
    def get_perspective_parameters(self, world_width, world_height):
        
        # Stop when a perspective matrix has been find
        while self.perspective_matrix is None:
            found_corners = False
            while not found_corners:
                
                # Get original frame
                original_frame = self._get_original_frame()
                
                # Get corner aruco markers
                original_frame, corner_positions, found_corners = self._detect_corner_markers(original_frame)
            
            print("Find all four corners!")
            # Define source points for the perspective transform
            source_points = corner_positions.astype(np.float32)
            
            # Calculate perspective transform
            self._compute_perspective_transform(source_points, world_width, world_height)
            
        print("Got a valid perspective matrix and a defined roi!")
        
        
    def get_frame(self):
        # Get original frame
        original_frame = self._get_original_frame()
        
        # Transform in process frame with the calculated perspective matrix and the defined roi
        process_frame = cv2.warpPerspective(original_frame, self.perspective_matrix, self.process_roi)  
        return original_frame, process_frame  
        
    def get_obstacles_position(self, frame):
        
        # Detect obstacles
        obstacles_pos, obstacles_frame, visualization_steps = self._detect_obstacles(frame)
        if len(obstacles_pos) == 0:
            print("No obstacles detected!")
        else:
            print("Obstacles detected! number of obstacles: ", len(obstacles_pos))
            obstacles_pos = {key: utils.pixels_to_mm(value, self.scale_factor ) for key, value in obstacles_pos.items()}
            obstacles_rounded = {k: np.round(v, 1) for k, v in obstacles_pos.items()}
            obstacles_formatted = '\n '.join([f"{k}: {v}" for k, v in obstacles_rounded.items()])
            print(f"Obstacles [mm]: \n {obstacles_formatted}")
            utils.display_processing_steps(visualization_steps)
            
        return obstacles_pos, obstacles_frame
    
    def get_thymio_position(self, frame):
        
        # Detect thymio marker
        frame, thymio_pos, found_thymio = self._detect_thymio_marker(frame)
        
        # Check if thymio position is found
        if found_thymio:
            # Convert to mm
            thymio_pos[:2] = utils.pixels_to_mm(thymio_pos[:2], self.scale_factor)
        
        return frame, thymio_pos, found_thymio
        
    def get_goal_position(self, frame):
        
        # Detect goal marker
        frame, goal_pos, found_goal = self._detect_goal_marker(frame)
        
        # Check if goal position is found
        if found_goal:
            # Convert to mm
            goal_pos = utils.pixels_to_mm(goal_pos, self.scale_factor)
            
        return frame, goal_pos, found_goal
    
    def flush(self):
        if self.cap is not None:
            # Flush the buffer by reading multiple frames
            for _ in range(10):  # Read 5 frames to clear buffer
                self.cap.grab()
                
    def connect_webcam(self):
        # Connect to the webcam
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        
        # Check if the webcam is opened correctly
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        print("Webcam connected correctly.")
        return True
    
    def disconnect_webcam(self):
        if self.cap is not None:
            self.cap.release()
        print("Webcam disconnected correctly.")
        cv2.destroyAllWindows()
    
    
        
