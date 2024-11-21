import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvisgraph as vg
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

class GlobalNav:
    def __init__(self, world_width, world_height, obstacle_min_area, thymio_size, security_margin, scale_factor, aruco_size):
        
        # Initialize World
        self.world_width = world_width
        self.world_height = world_height
        
        # Initialize Thymio
        self.thymio_width = utils.mm_to_pixels(thymio_size['width'], scale_factor)
        self.security_margin = utils.mm_to_pixels(security_margin, scale_factor)
        
        # Compute scale factor
        self.scale_factor = scale_factor
        self.obstale_min_area = utils.mm_to_pixels(obstacle_min_area, scale_factor**2)
        
        # Add mask size for ArUco markers
        self.aruco_mask_size = utils.mm_to_pixels(aruco_size, scale_factor)
        print("GlobalNav Initialized")
    
    def _create_aruco_mask(self, img, thymio_pos, goal_pos):
    
        pos = tuple(map(int, thymio_pos))
        cv2.circle(img, pos, self.aruco_mask_size, 255, -1)
            
        pos = tuple(map(int, goal_pos))
        cv2.circle(img, pos, self.aruco_mask_size, 255, -1)
            
        return img
        
    def _filter_close_corners(self, corners, min_distance=10):
        if len(corners) == 0:
            return corners
        
        # Convert to numpy array if it's not already
        corners = np.array(corners)
        
        # List to keep track of corners to keep
        filtered_corners = [corners[0]]
        
        for corner in corners[1:]:
            if all(utils.distance(corner, kept_corner) >= min_distance for kept_corner in filtered_corners):
                filtered_corners.append(corner)
        
        return np.array(filtered_corners)
    
    def _extend_obstacles(self, corners):
        # Calculate the center point of the obstacle
        center = np.mean(corners, axis=0)
        
        # For each corner point
        extended_corners = np.zeros_like(corners)
        for i in range(len(corners)):
            # Get vector from center to corner
            vector = corners[i] - center
            
            # Normalize the vector
            length = np.linalg.norm(vector)
            if length > 0:
                normalized_vector = vector / length
                
                # Move the corner point outward by the width of the thymio
                extended_corners[i] = corners[i] + normalized_vector * (self.thymio_width/2 + self.security_margin) 
                
        return extended_corners
    
    def _detect_contours(self, img, thymio_pos, goal_pos):
        img = img.copy()

        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        _, threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create mask to exclude ArUco markers
        masked_img = self._create_aruco_mask(threshold_img, thymio_pos, goal_pos)

        # Apply Canny edge detection
        edges_img = cv2.Canny(masked_img, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the threshold image for drawing
        contour_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)

        # Dictionary to store corners for each obstacle
        obstacles_corners = {}

        for i, contour in enumerate(contours):
            # Draw the contour
            
            # Filter small contours
            if cv2.contourArea(contour) <= self.obstale_min_area:  # Minimum area threshold
                continue
            
            cv2.drawContours(contour_img, [contour], -1, (0, 255, 0), 2)
            
            # Approximate the contour to find corners
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Precision parameter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get corners for this contour and filter them
            corners = approx.reshape(-1, 2).astype(np.float32)
            filtered_corners = self._filter_close_corners(corners)
            
            extended_corners = self._extend_obstacles(corners)
    
            # Store corners in dictionary
            obstacle_name = f"obstacle{i+1}"
            obstacles_corners[obstacle_name] = np.array(extended_corners)

            for corner, extended_corners in zip(filtered_corners, extended_corners):
                cv2.circle(contour_img, tuple(corner.astype(np.int32)), 5, (0, 0, 255), -1)
                cv2.circle(contour_img, tuple(extended_corners.astype(np.int32)), 5, (255, 0, 0), -1)
            
        return contour_img, obstacles_corners
    
    def _compute_trajectory(self, obstacles_corners, thymio_pos, goal_pos):
        
        path_points = None
        
        # Extract initial and goal positions
        thymio_pos = tuple(thymio_pos)
        goal_pos = tuple(goal_pos)
        
        # Convert obstacles to list format for pyvisgraph
        obstacles = []
        for obstacle_key in obstacles_corners:
            obstacle_points = obstacles_corners[obstacle_key]
            # Convert numpy array points to list of tuples
            obstacle = [tuple(point) for point in obstacle_points]
            obstacles.append(obstacle)
            
        # Convert to PyVisGraph format
        start = vg.Point(thymio_pos[0], thymio_pos[1])
        end = vg.Point(goal_pos[0], goal_pos[1])
        
        # Convert obstacles to PyVisGraph polygons
        polygon_obstacles = []
        for obstacle in obstacles:
            polygon = []
            for point in obstacle:
                polygon.append(vg.Point(point[0], point[1]))
            polygon_obstacles.append(polygon)
        
        # Build visibility graph
        graph = vg.VisGraph()
        graph.build(polygon_obstacles, status=False)
        
        # Find the shortest path
        shortest_path = graph.shortest_path(start, end)

        # Calculate path length
        path_length = 0
        for i in range(len(shortest_path) - 1):
            p1 = shortest_path[i]
            p2 = shortest_path[i + 1]
            path_length += ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
        
        path_length = int(path_length)
        
        path_points = [np.array([point.x,point.y]) for point in shortest_path]
        
        return path_points
    
    def get_trajectory(self, img, thymio_pos, goal_pos):
        # Check if image is None
        if img is None:
            return None, None, None, False
        
        thymio_pos = thymio_pos.copy()
        goal_pos = goal_pos.copy()
        
        # Transform in pixels
        thymio_pos = np.array([
                utils.mm_to_pixels(thymio_pos[0], self.scale_factor), 
                utils.mm_to_pixels(thymio_pos[1], self.scale_factor)
                ])
        
        goal_pos = np.array([
                utils.mm_to_pixels(goal_pos[0], self.scale_factor), 
                utils.mm_to_pixels(goal_pos[1], self.scale_factor)
                ])
        
        # Detect obstacles
        trajectory_img, obstacles_corners= self._detect_contours(img, thymio_pos, goal_pos)

        # Compute trajectory
        trajectory_pos = self._compute_trajectory(obstacles_corners, thymio_pos, goal_pos)
        
        # Add thymio and goal positions to the image
        thymio_pos = tuple(map(int, thymio_pos)) 
        cv2.circle(trajectory_img, thymio_pos, 5, (0, 0, 255), -1)
        text_pos = (thymio_pos[0] + 10, thymio_pos[1] - 10)  
        cv2.putText(trajectory_img, "Start", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        goal_pos = tuple(map(int, goal_pos))
        cv2.circle(trajectory_img, goal_pos, 5, (0, 0, 255), -1)
        text_pos = (goal_pos[0] + 10, goal_pos[1] - 10)  
        cv2.putText(trajectory_img, "Goal", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Convert to mm
        if obstacles_corners:
            obstacles_corners = {
                key: np.array([
                    [utils.pixels_to_mm(point[0], self.scale_factor), 
                     utils.pixels_to_mm(point[1], self.scale_factor)]
                    for point in points]) for key, points in obstacles_corners.items()
            }
        
        if len(trajectory_pos)>=0:
            # Draw trajectory
            for i in range(len(trajectory_pos) - 1):
                point1 = tuple(map(int, trajectory_pos[i]))
                point2 = tuple(map(int, trajectory_pos[i + 1]))
                cv2.line(trajectory_img, point1, point2, (0, 0, 255), 2)
            # Convert to mm
            trajectory_pos = np.array([
                [utils.pixels_to_mm(point[0], self.scale_factor), 
                 utils.pixels_to_mm(point[1], self.scale_factor)] 
                for point in trajectory_pos])
            
        return trajectory_img, trajectory_pos, obstacles_corners, True
    