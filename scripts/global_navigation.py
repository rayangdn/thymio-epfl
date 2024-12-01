# Global navigation system using visibility graphs for path planning
import numpy as np
import cv2
import pyvisgraph as vg
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import utils

# Extra distance added to obstacles for collision avoidance
SECURITY_MARGIN = 60 # mm

class GlobalNav:
    def __init__(self):
        print("GlobalNav initialized correctly.")

    def _extend_obstacles(self, corners, thymio_width):
        # Calculate obstacle centroid
        center = np.mean(corners, axis=0)
        extended_corners = np.zeros_like(corners)
        
        # Extend each corner outward from center
        for i in range(len(corners)):
            vector = corners[i] - center
            length = np.linalg.norm(vector)
            
            if length > 0:
                # Normalize and scale vector by robot width plus margin
                normalized_vector = vector / length
                extended_corners[i] = corners[i] + normalized_vector * (thymio_width//2 + SECURITY_MARGIN) 
                
        return extended_corners
        
    def _compute_trajectory(self, obstacles_pos, thymio_pos, goal_pos):
        # Convert positions to tuples for pyvisgraph
        thymio_pos = tuple(thymio_pos)
        goal_pos = tuple(goal_pos)
        
        # Convert obstacle corners to list of polygons
        obstacles = []
        for obstacle_key in obstacles_pos:
            obstacle_points = obstacles_pos[obstacle_key]
            obstacle = [tuple(point) for point in obstacle_points]
            obstacles.append(obstacle)
            
        # Create pyvisgraph points for start and goal
        start = vg.Point(thymio_pos[0], thymio_pos[1])
        end = vg.Point(goal_pos[0], goal_pos[1])
        
        # Convert obstacles to pyvisgraph format
        polygon_obstacles = []
        for obstacle in obstacles:
            polygon = []
            for point in obstacle:
                polygon.append(vg.Point(point[0], point[1]))
            polygon_obstacles.append(polygon)
        
        # Create visibility graph and find shortest path
        graph = vg.VisGraph()
        graph.build(polygon_obstacles, status=False)
        shortest_path = graph.shortest_path(start, end)

        # Calculate total path length
        path_length = 0
        for i in range(len(shortest_path) - 1):
            p1, p2 = shortest_path[i], shortest_path[i + 1]
            path_length += ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
        
        # Convert path points to numpy arrays
        path_points = [np.array([point.x, point.y]) for point in shortest_path]
        
        return path_points
    
    def get_trajectory(self, img, thymio_pos, goal_pos, obstacles_pos, 
                   thymio_width, scale_factor):
        # Extend obstacles by robot width plus margin
        extended_obstacles = {}
        for obstacle, corners in obstacles_pos.items():
            extended_obstacles[obstacle] = self._extend_obstacles(corners, thymio_width)
            
            # Convert extended obstacle corners to pixel coordinates
            pts_pixels = []
            for point in extended_obstacles[obstacle]:
                px = utils.mm_to_pixels(point[0], scale_factor)
                py = utils.mm_to_pixels(point[1], scale_factor)
                pts_pixels.append(np.array([px, py]))
                
            # Draw extended obstacles
            pts_int = [p.astype(np.int32) for p in pts_pixels]
            for point in pts_int:
                cv2.circle(img, tuple(point), 3, (255, 0, 0), -1)
                
        trajectory_img = img.copy()
        found_trajectory = False
        
        # Generate path using visibility graph
        path_points = self._compute_trajectory(extended_obstacles, thymio_pos[:2], goal_pos)
        
        if path_points is not None:
            found_trajectory = True
            print("\nTrajectory found! Number of waypoints:", len(path_points)-1)
            print("\nWaypoint path [mm]:")
            for i, point in enumerate(path_points):
                if i > 0:
                    print(f"Waypoint {i}: {point}")
  
            # Convert waypoints to pixel coordinates for visualization
            path_points_pixels = []
            for point in path_points:
                px = utils.mm_to_pixels(point[0], scale_factor)
                py = utils.mm_to_pixels(point[1], scale_factor)
                path_points_pixels.append(np.array([px, py]))
            
            path_points_int = [p.astype(np.int32) for p in path_points_pixels]
            
            # Draw path lines and waypoints
            for i in range(len(path_points_int)-1):
                cv2.line(trajectory_img, tuple(path_points_int[i]), 
                        tuple(path_points_int[i+1]), (0, 255, 0), 2)
            
            for point in path_points_int:
                cv2.circle(trajectory_img, tuple(point), 3, (255, 0, 0), -1)
                
            # Label start and goal positions
            cv2.putText(trajectory_img, 'START', 
                   (path_points_int[0][0] + 10, path_points_int[0][1] + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
            cv2.putText(trajectory_img, 'GOAL', 
                    (path_points_int[-1][0] + 10, path_points_int[-1][1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
 
        return trajectory_img, path_points[1:], found_trajectory # Skip start position