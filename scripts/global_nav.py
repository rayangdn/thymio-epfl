import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2

class GlobalNav:
    def __init__(self):
        print("Initializing GlobalNav")
        
        # Load config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)             
        config_path = os.path.join(parent_dir, 'config', 'config.yaml')
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Initialize World
        self.world_width = self.config['world']['width']
        self.world_height = self.config['world']['height']
        
        # Initialize Thymio
        self.thymio_size = self.config['thymio']['size']
        # Compute scale factor
        self.scale_factor = self.config['webcam']['resolution'][1] / self.world_width
        
    def distance(self, p1, p2):
        return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
    def convert_to_cm(self, points):
        points_cm = points.copy()
        points_cm[:, 0] = points[:, 0] / self.scale_factor  # x coordinates
        points_cm[:, 1] = points[:, 1] / self.scale_factor  # y coordinates
        return points_cm
    
    def convert_to_pixels(self, points):
        points_pixels = points.copy()
        points_pixels[:, 0] = points[:, 0] * self.scale_factor
        points_pixels[:, 1] = points[:, 1] * self.scale_factor
        return points_pixels
        
    def filter_close_corners(self, corners, min_distance=10):
        if len(corners) == 0:
            return corners
        
        # Convert to numpy array if it's not already
        corners = np.array(corners)
        
        # List to keep track of corners to keep
        filtered_corners = []
        
        # Add first corner
        filtered_corners.append(corners[0])
        
        # Check each corner against the kept corners
        for corner in corners[1:]:
            # Flag to check if current corner is far enough from all kept corners
            is_far_enough = True
            
            for kept_corner in filtered_corners:
                if self.distance(corner, kept_corner) < min_distance:
                    is_far_enough = False
                    break
            
            if is_far_enough:
                filtered_corners.append(corner)
        
        return np.array(filtered_corners)
    
    def extend_obstacles(self, corners):
        corners = np.array(corners)
        
        # Calculate the safety margin (half the robot size plus a small buffer)
        safety_margin = (self.thymio_size['width'] / 2) + 0.5  # cm
        
        extended_corners = []
        n = len(corners)
        
        for i in range(n):
            current = corners[i]
            prev = corners[(i - 1) % n]  # Previous point
            next = corners[(i + 1) % n]  # Next point
            
            # Calculate vectors to previous and next points
            v1 = current - prev
            v2 = next - current
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Calculate bisector vector
            bisector = v1_norm + v2_norm
            
            # Normalize bisector
            if np.linalg.norm(bisector) > 0:
                bisector = bisector / np.linalg.norm(bisector)
                
                # Calculate the scaling factor for the bisector
                # The more acute the angle, the larger the extension needed
                angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
                scale = safety_margin / np.sin(angle/2) if angle != 0 else safety_margin
                
                # Extend the corner point
                extended_point = current + bisector * scale
                extended_corners.append(extended_point)
            else:
                # If vectors are opposite, extend perpendicular to both
                perp = np.array([-v1_norm[1], v1_norm[0]])
                extended_corners.append(current + perp * safety_margin)
                extended_corners.append(current - perp * safety_margin)
        
        return np.array(extended_corners)

    def detect_contours(self, img, min_corner_distance=10):
        img = img.copy()

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for noise reduction
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        _, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply Canny edge detection
        edges_img = cv2.Canny(threshold_img, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the threshold image for drawing
        contour_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)

        # Dictionary to store corners for each obstacle
        obstacles_corners = {}

        for i, contour in enumerate(contours):
            # Draw the contour
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
            
            contour = contour.astype(np.float32)
            
            # Approximate the contour to find corners
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Precision parameter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get corners for this contour and filter them
            corners = approx.reshape(-1, 2).astype(np.float32)
            filtered_corners = self.filter_close_corners(corners, min_corner_distance)
            
            corners_cm = self.convert_to_cm(filtered_corners)
            extended_corners_cm = self.extend_obstacles(corners_cm)
            extended_corners = self.convert_to_pixels(extended_corners_cm)
            
            # Store corners in dictionary
            obstacle_name = f"Obstacle{i+1}"
            obstacles_corners[obstacle_name] = [[round(x, 3), round(y, 3)] for x, y in extended_corners_cm]

            for corner, extended_corners in zip(filtered_corners, extended_corners):
                corner = corner.astype(np.int32)
                extended_corners = extended_corners.astype(np.int32)
                cv2.circle(contour_img, tuple(corner), 5, (0, 0, 255), -1)
                cv2.circle(contour_img, tuple(extended_corners), 5, (255, 0, 0), -1)
        
        # Draw the border of the world
        h, w = contour_img.shape[:2]
        border_thickness = 2
        cv2.rectangle(contour_img, (border_thickness, border_thickness), (w - border_thickness, h - border_thickness), (0, 0, 0), border_thickness)
        return contour_img, obstacles_corners