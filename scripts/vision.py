# Vision system for ArUco marker detection and obstacle recognition using OpenCV
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

# Intrinsic camera parameters from calibration
CAM_MATRIX = np.array([[1484.14935, 0, 964.220337],
                       [0, 1490.50225, 539.531044],
                       [0, 0, 1]])

# Lens distortion coefficients from calibration
CAM_DISTORTION = np.array([0.154056967, -1.04865202, -0.000689137957, -0.00132248869, 1.48905183])
  
# Input resolution for camera capture
CAM_RESOLUTION = (1920, 1080) 

# Minimum area threshold for valid obstacle detection
OBSTACLE_MIN_AREA = 500 # mm²

# Minimum distance between corner markers
MIN_CORNER_DISTANCE = 10 #pixels
    
class Vision():    
    def __init__(self, device_id):
        # Camera device identifier
        self.device_id = device_id 
        self.cap = None
        
        # Matrix for perspective transform to top-down view
        self.perspective_matrix = None
        # Dimensions of the transformed image
        self.process_roi = None
        # Conversion factor from pixels to millimeters
        self.scale_factor = None
        
        # Initialize 4x4 ArUco marker detector with 50 unique markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        print("Vision initialized correctly.")
    
    def _undistort_frame(self, frame):
        # Calculate optimal camera matrix for undistortion
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(CAM_MATRIX, CAM_DISTORTION, (w, h), 1, (w, h))
        
        # Apply undistortion using calibration parameters
        frame = cv2.undistort(frame, CAM_MATRIX, CAM_DISTORTION, None, newcameramtx)
        
        # Crop to valid pixel region after undistortion
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        return frame
    
    def _get_original_frame(self):
        # Capture and undistort a single frame from camera
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            frame = self._undistort_frame(frame)
            return frame
    
    def _detect_corner_markers(self, frame):
        # Convert to grayscale for ArUco detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in frame
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Initialize array for 4 corner marker positions
        corner_positions = np.zeros((4, 2), dtype=int)
        found_corners = False
        
        if ids is not None:
            # Draw detected markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Process each detected marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Store corner markers (IDs 0-3) positions
                if marker_id in [0, 1, 2, 3]:
                    corner_positions[marker_id] = np.array([c[0][0], c[0][1]])
                    
                # Check if all 4 corners are detected
                if np.all(corner_positions):  
                    found_corners = True
                    # Reorder corners to [0, 2, 3, 1]
                    ordered_corners = corner_positions[[0, 2, 3, 1]]
                    
                    # Convert ordered corners to the format needed by polylines
                    corners_poly = ordered_corners.reshape((-1, 1, 2))
                    
                    # Add the first point at the end to close the rectangle
                    corners_poly = np.append(corners_poly, [corners_poly[0]], axis=0)
                    
                    # Draw the rectangle
                    cv2.polylines(frame, [corners_poly], isClosed=True, color=(0, 255, 0), thickness=2)
                    
        return frame, corner_positions, found_corners
    
    def _detect_thymio_marker(self, frame):
        # Convert to grayscale for ArUco detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in frame
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Initialize Thymio pose array [x, y, orientation]
        thymio_pos = np.zeros(3, dtype=float)
        found_thymio = False
        
        if ids is not None:
            # Draw detected markers on frame
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Process each detected marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Calculate Thymio position from marker ID 4
                if marker_id == 4:
                    center = np.mean(c, axis=0).astype(int)
                    found_thymio = True
                    thymio_pos[:2] = center
                    
                    # Calculate orientation from marker front edge vector
                    front_edge = c[1] - c[0]
                    orientation = np.arctan2(front_edge[1], front_edge[0])
                    
                    # Normalize angle to [-π, π]
                    thymio_pos[2] = np.arctan2(np.sin(orientation), np.cos(orientation))
                    
                    # Draw orientation arrow and center point
                    start_point = (int(center[0]), int(center[1]))
                    end_point = (int(center[0] + 50 * np.cos(orientation)), 
                               int(center[1] + 50 * np.sin(orientation)))
                    
                    cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2)
                    cv2.circle(frame, start_point, 3, (0, 0, 255), -1)
        
        return frame, thymio_pos, found_thymio

    def _detect_goal_marker(self, frame):
        # Convert to grayscale for ArUco detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers in frame
        corners, ids, _ = self.detector.detectMarkers(gray_frame)
        
        # Initialize goal position array [x, y]
        goal_pos = np.zeros(2, dtype=float)
        found_goal = False
        
        if ids is not None:
            # Draw detected markers on frame
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Process each detected marker
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Calculate goal position from marker ID 5
                if marker_id == 5:
                    center = np.mean(c, axis=0).astype(int)
                    found_goal = True
                    goal_pos = center
                    
        return frame, goal_pos, found_goal
    
    def _compute_perspective_transform(self, source_points, world_width, world_height):
        # Calculate pixel to millimeter scale factor
        self.scale_factor = CAM_RESOLUTION[1] / world_width
        
        # Calculate dimensions of transformed image
        dest_width = world_width * self.scale_factor 
        dest_height = world_height * self.scale_factor 
        
        # Define destination points for top-down view
        dest_points = np.float32([
            [0, dest_height],          # bottom-left
            [dest_width, dest_height], # bottom-right
            [0, 0],                    # top-left
            [dest_width, 0]            # top-right  
        ])
        
        # Calculate perspective transformation matrix
        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        self.process_roi = np.array([dest_width, dest_height]).astype(int)
    
    def _filter_close_corners(self, corners):
        # Return empty list if no corners
        if len(corners) == 0:
            return corners
        
        corners = np.array(corners)
        filtered_corners = [corners[0]]
        
        # Keep corners that are at least min_distance away from all kept corners
        for corner in corners[1:]:
            if all(utils.distance(corner, kept_corner) >= MIN_CORNER_DISTANCE for kept_corner in filtered_corners):
                filtered_corners.append(corner)
        
        return np.array(filtered_corners)
        
    def _detect_obstacles(self, frame):
        # Store processing steps for visualization
        visualization_steps = {'Original image': frame.copy()}
        
        # Convert to grayscale and apply Gaussian blur
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Apply Otsu's thresholding
        _, threshold_frame = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        visualization_steps['Threshold image'] = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)
        
        # Apply Canny edge detection
        edges_frame = cv2.Canny(threshold_frame, 50, 150)
        visualization_steps['Edges image'] = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2RGB)

        # Find contours of obstacles
        contours, _ = cv2.findContours(edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles_frame = cv2.cvtColor(threshold_frame, cv2.COLOR_GRAY2RGB)
        obstacles_corners = {}
        
        # Process each contour above minimum area threshold
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) <= OBSTACLE_MIN_AREA:
                continue
            
            # Draw contour outline
            cv2.drawContours(obstacles_frame, [contour], -1, (255, 0, 0), 2)
            
            # Approximate contour with polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            corners = approx.reshape(-1, 2).astype(np.float32)
            
            # Filter and store corner points
            obstacle_name = f"obstacle{i+1}"
            filtered_corners = self._filter_close_corners(corners)
            obstacles_corners[obstacle_name] = filtered_corners
            
            # Draw corner points
            for corner in filtered_corners:
                cv2.circle(obstacles_frame, tuple(corner.astype(np.int32)), 5, (0, 0, 255), -1)
        
        visualization_steps['Obstacles image'] = obstacles_frame
        return obstacles_corners, obstacles_frame, visualization_steps
    
    def get_perspective_parameters(self, world_map):
        # Get world map dimensions
        world_width, world_height = world_map[0], world_map[1]
        # Calculate perspective transform until successful
        while self.perspective_matrix is None:
            found_corners = False
            while not found_corners:
                # Get frame and detect corner markers
                original_frame = self._get_original_frame()
                original_frame, corner_positions, found_corners = self._detect_corner_markers(original_frame)
            
            # Calculate transform from corner positions
            source_points = corner_positions.astype(np.float32)
            self._compute_perspective_transform(source_points, world_width, world_height)
        
        print("Got a valid perspective matrix and a defined roi!")
        
    def get_frame(self):
        # Capture frame and apply perspective transform
        original_frame = self._get_original_frame()
        process_frame = cv2.warpPerspective(original_frame, self.perspective_matrix, self.process_roi)  
        return original_frame, process_frame  
        
    def get_obstacles_position(self, frame):
        # Detect and process obstacles in frame
        obstacles_pos, obstacles_frame, visualization_steps = self._detect_obstacles(frame)
        
        if len(obstacles_pos) == 0:
            print("No obstacles detected!")
        else:
            # Convert obstacle positions from pixels to millimeters
            print(f"Obstacles detected! number of obstacles: {len(obstacles_pos)}")
            obstacles_pos = {key: utils.pixels_to_mm(value, self.scale_factor) 
                           for key, value in obstacles_pos.items()}
            
            # Format and print obstacle positions
            obstacles_rounded = {k: np.round(v, 1) for k, v in obstacles_pos.items()}
            obstacles_formatted = '\n '.join([f"{k}: {v}" for k, v in obstacles_rounded.items()])
            print(f"Obstacles [mm]: \n {obstacles_formatted}")
            utils.display_processing_steps(visualization_steps)
            
        return obstacles_pos, obstacles_frame
    
    def get_thymio_position(self, frame):
        # Detect Thymio marker and convert position to millimeters
        frame, thymio_pos, found_thymio = self._detect_thymio_marker(frame)
        if found_thymio:
            thymio_pos[:2] = utils.pixels_to_mm(thymio_pos[:2], self.scale_factor)
        return frame, thymio_pos, found_thymio
        
    def get_goal_position(self, frame):
        # Detect goal marker and convert position to millimeters
        frame, goal_pos, found_goal = self._detect_goal_marker(frame)
        if found_goal:
            goal_pos = utils.pixels_to_mm(goal_pos, self.scale_factor)
        return frame, goal_pos, found_goal
    
    def flush(self):
        # Clear camera buffer by reading multiple frames
        if self.cap is not None:
            for _ in range(10):
                self.cap.grab()
                
    def connect_webcam(self):
        # Initialize camera connection with specified resolution
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RESOLUTION[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RESOLUTION[1])
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
            
        print("Webcam connected correctly.")
        return True
    
    def disconnect_webcam(self):
        # Release camera resources and close windows
        if self.cap is not None:
            self.cap.release()
        print("Webcam disconnected correctly.")
        cv2.destroyAllWindows()