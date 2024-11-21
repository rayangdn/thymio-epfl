import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

class Vision():
    # Class-level constant for corner mapping
    MAPPING = {
        0: "bottom_left",
        1: "bottom_right",
        2: "top_left",
        3: "top_right",
        4: "thymio",
        5: "goal"
    }
    
    def __init__(self, device_id, camera_matrix, dist_coeffs, resolution, padding, scale_factor, world_width, world_height):
            
        # Initialize camera
        self.cap = None
        self.device_id = device_id
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.resolution = resolution
        self.padding = padding
        self.scale_factor = scale_factor
        
        # Perspective transform matrix
        self.perspective_matrix = None
        
        # Initialize World
        self.world_width = world_width
        self.world_height = world_height
        
        # Initialize ArUco detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        print("Vision Initialized")
        
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
        cv2.destroyAllWindows()
        
    def _calculate_marker_orientation(self, corners):
        # Get the first and second corners of the marker
        # ArUco corners are ordered clockwise from top-left
        corner0 = corners[0]  # Top-left corner
        corner1 = corners[1]  # Top-right corner
        
        # Calculate the vector between corners
        dx = corner1[0] - corner0[0]
        dy = corner1[1] - corner0[1]
        
        # Calculate angle in radians relative to positive x-axis
        angle = np.arctan2(dy, dx)
        
        # Normalize angle to be between -π and π
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        return angle

    def _process_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Create array to store corner, thymio and goal positions
        corner_positions = np.zeros((4, 2), dtype=int)  # 4 corners, each with x,y coordinates
        thymio_corners = np.zeros((4, 2), dtype=int)  # Store the full corners for thymio marker
        thymio_position = np.zeros(3, dtype=int)  # x, y, orientation
        goal_position = np.zeros(2, dtype=int)  # x, y
        
        found_corners = False
        
        if ids is not None:
            # Draw markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Get center of marker
                center = np.mean(c, axis=0).astype(int)
            
                # Store corner positions
                if marker_id in [0, 1, 2, 3]:
                    corner_positions[marker_id] = np.array([c[0][0], c[0][1]])
                
                # Store thymio position and corners
                if marker_id == 4:
                    #Get corners for orientation calculation
                    thymio_corners = c
                    thymio_position[:2] = center
                    
                # Store goal position
                if marker_id == 5:
                    goal_position = center
                
                # Get name from mapping
                name = self.MAPPING.get(marker_id, f"Unknown Marker: {marker_id}")
                
                # Draw background rectangle for better text visibility
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, 
                            (center[0] - 5, center[1] - text_size[1] - 5),
                            (center[0] + text_size[0] + 5, center[1] + 5),
                            (0, 0, 0), -1)
                
                cv2.putText(frame, name, 
                            (center[0], center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 0, 255), 2)
                
                # Draw lines between corners if all are detected to define the world
                if np.all(corner_positions):  # Check if all corners have been detected
                    corners_order = [0, 2, 3, 1] # bottom_left, top_left, top_right, bottom_right
                    found_corners = True
                    for i in range(4):
                        start = tuple(map(int, corner_positions[corners_order[i]]))
                        end = tuple(map(int,corner_positions[corners_order[(i + 1) % 4]]))
                        frame = cv2.line(frame, start, end, (0, 255, 0), 2)
        
        return frame, corner_positions, thymio_position, thymio_corners, goal_position, found_corners
    
    def _undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        
        # Undistort frame
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Crop the frame
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        
        return frame
    
    def _compute_perspective_transform(self, source_points):
        
        # define destination points
        dest_width = self.world_width * self.scale_factor
        dest_height = self.world_height * self.scale_factor
        
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
                return None, None, None, None, None, False, False
            
            found_thymio = False
            found_goal = False
            
            # Undistort the frame
            frame = self._undistort_frame(frame)
            
            # Detect markers
            frame, corner_positions, thymio_position, thymio_corners, goal_position, found_corners = self._process_aruco_markers(frame)
            
            process_frame = None
            roi = None
            
            # Compute perspective transform if we have all corner positions
            if found_corners:
                source_points = corner_positions.astype(np.float32)
                roi = self._compute_perspective_transform(source_points)
                if roi is not None:
                    # Get the top-down view of the map
                    process_frame = cv2.warpPerspective(frame, self.perspective_matrix, roi)
                    
                    if np.all(thymio_corners): # Check if thymio has been detected
                        
                        # Transform thymio corners to process_frame space
                        thymio_corners_reshaped = thymio_corners.reshape(-1, 1, 2)
                        transformed_corners = cv2.perspectiveTransform(thymio_corners_reshaped, self.perspective_matrix)
                        
                        # Calculate orientation in transformed space
                        orientation = self._calculate_marker_orientation(transformed_corners.reshape(-1, 2))
                        
                        # Calculate center position
                        center = np.mean(transformed_corners, axis=0)[0]
                        thymio_position = np.array([center[0], center[1], orientation])
                        
                        # Draw orientation arrow in process_frame
                        end_point = (
                            int(center[0] + 30 * np.cos(orientation)),
                            int(center[1] + 30 * np.sin(orientation))
                        )
                        cv2.arrowedLine(process_frame, 
                                      (int(center[0]), int(center[1])), 
                                      end_point, 
                                      (0, 255, 0), 2)
                        
                        #Pixels to mm
                        thymio_position = np.array([utils.pixels_to_mm(thymio_position[0], self.scale_factor), 
                            utils.pixels_to_mm(thymio_position[1], self.scale_factor)])
                        found_thymio = True
                    
                    # Check if goal position is detected
                    if np.all(goal_position):  
                        transformed_point = cv2.perspectiveTransform(goal_position, self.perspective_matrix)[0][0]
                        goal_position = np.array([utils.pixels_to_mm(transformed_point[0], self.scale_factor), 
                                                  utils.pixels_to_mm(transformed_point[1], self.scale_factor)])
                        found_goal = True
                    
                    process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thymio_orientation = thymio_position[2] # Get orientation 
            thymio_position = thymio_position[:2] # Get x, y position in pixels
                
            return frame, process_frame, thymio_position, thymio_orientation, goal_position, found_thymio, found_goal
        
        return None, None, None, None, False, False