import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyvisgraph as vg

class GlobalNav:
    def __init__(self):
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
        
        print("GlobalNav Initialized")
        
    def distance(self, p1, p2):
        return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    
    def convert_to_mm(self, points):
        print(points)
        points_mm = points.copy()
        points_mm[:, 0] = points[:, 0] / self.scale_factor  # x coordinates
        points_mm[:, 1] = points[:, 1] / self.scale_factor  # y coordinates
        return points_mm
    
    def convert_to_pixels(self, points):
        points_pixels = points.copy()
        points_pixels[:, 0] = np.int32(points[:, 0] * self.scale_factor)
        points_pixels[:, 1] = np.int32(points[:, 1] * self.scale_factor)
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
        #need to implement this
        return corners

    def detect_contours(self, img):
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
            
            # Approximate the contour to find corners
            epsilon = 0.02 * cv2.arcLength(contour, True)  # Precision parameter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get corners for this contour and filter them
            corners = approx.reshape(-1, 2).astype(np.float32)
            filtered_corners = self.filter_close_corners(corners)
            
            corners_mm = self.convert_to_mm(filtered_corners)
            extended_corners_mm = self.extend_obstacles(corners_mm)
            
            # Convert extended corners back to pixels for visualization
            extended_corners = self.convert_to_pixels(extended_corners_mm)
            
            # Store corners in dictionary
            obstacle_name = f"obstacle{i+1}"
            obstacles_corners[obstacle_name] = np.array(corners_mm)

            for corner, extended_corners in zip(filtered_corners, extended_corners):
                cv2.circle(contour_img, tuple(corner.astype(np.int32)), 5, (0, 0, 255), -1)
                cv2.circle(contour_img, tuple(extended_corners.astype(np.int32)), 5, (255, 0, 0), -1)
        
        # Draw the border of the world
        h, w = contour_img.shape[:2]
        border_thickness = 2
        cv2.rectangle(contour_img, (border_thickness, border_thickness), (w - border_thickness, h - border_thickness), (0, 0, 0), border_thickness)
        
        return contour_img, obstacles_corners
    
    def compute_trajectory(self, obstacles_corners, thymio_goal_positions):
        # Extract initial and goal positions
        init_pos = tuple(thymio_goal_positions['thymio'])
        goal = tuple(thymio_goal_positions['goal'])
        
        # Convert obstacles to list format for pyvisgraph
        obstacles = []
        for obstacle_key in obstacles_corners:
            obstacle_points = obstacles_corners[obstacle_key]
            # Convert numpy array points to list of tuples
            obstacle = [tuple(point) for point in obstacle_points]
            obstacles.append(obstacle)
            
        # Convert to PyVisGraph format
        start = vg.Point(init_pos[0], init_pos[1])
        end = vg.Point(goal[0], goal[1])
        
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

    def get_trajectory(self, img, thymio_goal_positions):
        
        # Detect obstacles
        trajectory_img, obstacles_corners_mm = self.detect_contours(img)
        
        # Add thymio and goal positions to the image
        for pos_name, _ in thymio_goal_positions.items():
            point = (thymio_goal_positions[pos_name][0], thymio_goal_positions[pos_name][1])
            cv2.circle(trajectory_img, point, 5, (255, 0, 0), -1)
            
        # Convert thymio and goal positions to mm
        thymio_goal_positions_mm = {
            key: self.convert_to_mm(np.array([value], dtype=np.float32))[0] # Convert single point and get first element
            for key, value in thymio_goal_positions.items()
        }
        
        # Compute trajectory
        trajectory_points_mm = self.compute_trajectory(obstacles_corners_mm, thymio_goal_positions_mm)
        trajectory_points_pixels = (self.convert_to_pixels(np.array(trajectory_points_mm))).astype(np.int32)

        # Draw lines connecting the points
        for i in range(len(trajectory_points_mm) - 1):
            cv2.line(trajectory_img, tuple(trajectory_points_pixels[i]), tuple(trajectory_points_pixels[i + 1]), (255, 255, 0), 2)
            
        return trajectory_img, trajectory_points_mm, {"thymio": thymio_goal_positions_mm["thymio"]}
    

