import numpy as np
import cv2
import cv2.aruco as aruco
   
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
        
    def _process_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Create dictionaries to store corner, thymio and goal positions
        corner_positions = {
            "bottom_left": None, 
            "bottom_right": None,
            "top_left": None,
            "top_right": None
        }
        # Thymio position
        thymio_position = {"thymio": None}
        
        # Goal position
        goal_position = {"goal": None}
        
        found_corners = False
        
        if ids is not None:
            # Draw markers
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids)):
                marker_id = ids[i][0]
                c = corners[i][0]
                
                # Get center of marker
                center = np.mean(c, axis=0).astype(int)
                
                # Get name from mapping
                name = self.MAPPING.get(marker_id, f"Unknown Marker: {marker_id}")
                
                # Store corner positions
                if marker_id in [0, 1, 2, 3]:
                    corner_positions[name] = np.array([c[0][0], c[0][1]]).astype(int)
                
                # Store thymio position
                if marker_id in [4]:
                    thymio_position[name] = np.array([center[0][0], center[0][1]]).astype(int)
                    
                # Store goal position
                if marker_id in [5]:
                    goal_position[name] = np.array([center[0][0], center[0][1]]).astype(int)
                
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
                        start = tuple(map(int, corner_positions[corners_order[i]]))
                        end = tuple(map(int, corner_positions[corners_order[(i + 1) % 4]]))
                        frame = cv2.line(frame, start, end, (0, 255, 0), 2)
        
        return frame, corner_positions, thymio_position, goal_position, found_corners
    
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
                return None
            
            # Undistort the frame
            frame = self._undistort_frame(frame)
            
            # Detect markers
            frame, corner_positions, thymio_position, goal_position, found_corners = self._process_aruco_markers(frame)
            
            process_frame = None
            roi = None
            
            # Compute perspective transform if we have all corner positions
            if found_corners:
                source_points = np.array(list(corner_positions.values()), dtype=np.float32)
                roi = self._compute_perspective_transform(source_points)
                if roi is not None:
                    # Get the top-down view of the map
                    process_frame = cv2.warpPerspective(frame, self.perspective_matrix, roi)
                    
                    # add padding to the frame to avoid edge cases
                    process_frame = process_frame[self.padding:-self.padding, self.padding:-self.padding]
                    process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame, process_frame, thymio_position, goal_position
        
        return None, None, None, None
            
        
