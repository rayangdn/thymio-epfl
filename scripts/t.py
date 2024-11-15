import os
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco

class Vision:
    MAPPING = {
        0: "bottom_left",
        1: "bottom_right",
        2: "top_left",
        3: "top_right",
        4: "thymio",
        5: "goal"
    }

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)             
        config_path = os.path.join(parent_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.cap = None
        self.device_id = self.config['webcam']['device_id']
        self.camera_matrix = np.array(self.config['webcam']['matrix'])
        self.dist_coeffs = np.array(self.config['webcam']['distortion'])
        self.resolution = self.config['webcam']['resolution']
        self.world_width = self.config['width']
        self.world_height = self.config['height']
        self.scale = self.resolution[1] / self.world_width
        self.perspective_matrix = None
        self.scale_factor = 21/1080
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detect_edges_realtime(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_with_markers = frame.copy()
        coordinates_cm = []
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points_cm = [(point[0][0] * self.scale_factor, point[0][1] * self.scale_factor) 
                        for point in approx]
            coordinates_cm.append(points_cm)
            
            points = np.array(approx).reshape(-1, 2)
            for i in range(len(points)):
                next_i = (i + 1) % len(points)
                cv2.line(frame_with_markers, 
                        tuple(points[i]), 
                        tuple(points[next_i]), 
                        (0, 0, 255), 2)
                cv2.circle(frame_with_markers, tuple(points[i]), 5, (255, 0, 0), -1)
                
                x, y = points[i]
                x_cm, y_cm = x * self.scale_factor, y * self.scale_factor
                cv2.putText(frame_with_markers, 
                           f'({x_cm:.1f}, {y_cm:.1f}) cm',
                           (x + 10, y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           (0, 255, 0),
                           2)
        
        return frame_with_markers, coordinates_cm

    def connect_webcam(self):
        self.cap = cv2.VideoCapture(self.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False
        return True
    
    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, self.camera_matrix, 
                                    self.dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w]

    def detect_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, ids

    def draw_markers(self, frame, corners, ids):
        if ids is None:
            return frame, None
        
        coordinates = []
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        for i in range(len(ids)):
            marker_id = ids[i][0]
            c = corners[i][0]
            center = np.mean(c, axis=0).astype(int)
            name = self.MAPPING.get(marker_id, f"Unknown ID: {marker_id}")
            
            if (i <= 3) and (0 in ids) and (1 in ids) and (2 in ids) and (3 in ids):
                coordinates.append(corners[i][0][0].astype(int))

            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, 
                        (center[0] - 5, center[1] - text_size[1] - 5),
                        (center[0] + text_size[0] + 5, center[1] + 5),
                        (0, 0, 0), -1)
            
            cv2.putText(frame, name, 
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)
            
        if(len(coordinates) == 4):
            cv2.line(frame, (coordinates[0][0], coordinates[0][1]), (coordinates[1][0], coordinates[1][1]), (0, 255, 255), 2)
            cv2.line(frame, (coordinates[1][0], coordinates[1][1]), (coordinates[3][0], coordinates[3][1]), (0, 255, 255), 2)
            cv2.line(frame, (coordinates[2][0], coordinates[2][1]), (coordinates[3][0], coordinates[3][1]), (0, 255, 255), 2)
            cv2.line(frame, (coordinates[2][0], coordinates[2][1]), (coordinates[0][0], coordinates[0][1]), (0, 255, 255), 2)
        else:
            coordinates = None
        return frame, coordinates
    
    def compute_perspective_transform(self, source_points, frame):
        source_points = np.float32(source_points)
        dest_width = self.world_width * self.scale
        dest_height = self.world_height * self.scale
        
        dest_points = np.float32([
            [0, dest_height],
            [dest_width, dest_height],
            [0, 0],
            [dest_width, 0]
        ])
        
        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        return int(dest_width), int(dest_height)

    def get_2d_map(self, frame, dimensions):
        if self.perspective_matrix is not None:
            return cv2.warpPerspective(frame, self.perspective_matrix, dimensions)
        return None
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't read frame.")
                return None, None, None
            
            frame = self.undistort_frame(frame)
            
            # Detect ArUco markers
            corners, ids = self.detect_aruco_markers(frame)
            frame_with_markers, world_coordinates = self.draw_markers(frame.copy(), corners, ids)
            
            # Detect edges and draw coordinates
            
            
            map_view = None
            if world_coordinates is not None:
                dimensions = self.compute_perspective_transform(world_coordinates, frame)
                if dimensions is not None:
                    map_view = self.get_2d_map(frame, dimensions)
                    map_view, _= self.detect_edges_realtime(map_view)
            
            return frame_with_markers,  map_view
        
        return None, None, None

def main():
    vision = Vision()
    
    print(f"Trying to connect to device {vision.device_id}...")
    if not vision.connect_webcam():
        print("Could not find webcam on any device ID. Please check connection.")
        return
    
    print(f"Successfully connected to device {vision.device_id}")
    
    try:
        while True:
            frame_markers, map_view = vision.get_frame()
            if frame_markers is None:
                break
            
            cv2.imshow('ArUco Markers', frame_markers)
            
            if map_view is not None:
                cv2.imshow('2D World Map', map_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and map_view is not None:
                cv2.imwrite('map_view.png', map_view)
                print("Map view saved as 'map_view.png'")
                
    finally:
        vision.cleanup()

if __name__ == "__main__":
    main()